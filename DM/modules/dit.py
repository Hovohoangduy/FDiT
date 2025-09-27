import math
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from einops import rearrange, repeat

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):  # x: [B]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        args = x[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim)
        self.kw = nn.Linear(dim, dim)
        self.vw = nn.Linear(dim, dim)
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        self.ow = nn.Linear(dim, dim)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, L, D), but got {x.shape}")

        q = self.qw(x) # [B, L, D]
        k = self.kw(x) # [B, L, D]
        v = self.vw(x) # [B, L, D]

        B, L, D = q.shape
        H = self.n_heads
        Dh = D // H

        q = q.view(B, L, H, Dh).transpose(1, 2) # [B, H, L, Dh]
        k = k.view(B, L, H, Dh).transpose(1, 2) # [B, H, L, Dh]
        v = v.view(B, L, H, Dh).transpose(1, 2) # [B, H, L, Dh]

        k_proj = torch.einsum('b h l d, l k -> b h d k', k, self.E[:L, :]) # [B, H, Dh, k]
        v_proj = torch.einsum('b h l d, l k -> b h k d', v, self.F[:L, :]) # [B, H, k, Dh]

        attn = torch.softmax(torch.matmul(q, k_proj) * self.scale, dim=-1)  # [B, H, L, L]
        out = torch.matmul(attn, v_proj) # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D) # [B, L, D]
        return self.ow(out)


class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1  = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2  = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim) # gate for attn branch
        self.scale_2 = nn.Linear(dim, dim) # gate for mlp branch
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.gamma_1.weight); nn.init.zeros_(self.gamma_1.bias)
        nn.init.zeros_(self.beta_1.weight);  nn.init.zeros_(self.beta_1.bias)
        nn.init.zeros_(self.gamma_2.weight); nn.init.zeros_(self.gamma_2.bias)
        nn.init.zeros_(self.beta_2.weight);  nn.init.zeros_(self.beta_2.bias)
        nn.init.zeros_(self.scale_1.weight); nn.init.ones_(self.scale_1.bias)
        nn.init.zeros_(self.scale_2.weight); nn.init.ones_(self.scale_2.bias)

    def forward(self, x, c):
        scale_msa = self.gamma_1(c) # [B*, D]
        shift_msa = self.beta_1(c) # [B*, D]
        scale_mlp = self.gamma_2(c) # [B*, D]
        shift_mlp = self.beta_2(c) # [B*, D]

        gate_msa = self.scale_1(c).unsqueeze(1) # [B*, 1, D]
        gate_mlp = self.scale_2(c).unsqueeze(1) # [B*, 1, D]

        x = self.attn(modulate(self.ln1(x), shift_msa, scale_msa)) * gate_msa + x
        x = self.mlp(modulate(self.ln2(x), shift_mlp, scale_mlp)) * gate_mlp + x
        return x


class DiT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, time_emb_dim, out_channels, in_channels, seq_k=64):
        super().__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.kernel, self.padding = (1, 7, 7), (0, 3, 3)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, dim)
        )
        self.to_patch = nn.Conv3d(in_channels, dim, self.kernel, padding=self.padding)
        self.to_out = nn.Conv3d(dim, out_channels, 1)
        self.blocks = nn.ModuleList([])
        self.seq_k = seq_k
        self.heads = heads
        self.temp_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.register_buffer('null_cond_mask', torch.tensor([], dtype=torch.bool), persistent=False)
        self.cond_proj = None

    def build_blocks(self, seq_len):
        self.blocks = nn.ModuleList([
            TransformerBlock(seq_len=seq_len, dim=self.dim, heads=self.heads, mlp_dim=self.dim * 4, k=self.seq_k)
            for _ in range(3)
        ])

    def forward(self, x, t, cond=None, *args, **kwargs):
        B, C, F, H, W = x.shape
        device = x.device

        x = self.to_patch(x) # [B, D, F, H, W]
        x = rearrange(x, 'b d f h w -> (b f) (h w) d') # [B*F, L, D]

        t = t.to(device)
        t_emb = self.time_mlp(t) # [B, D]
        t_emb = repeat(t_emb, 'b d -> (b f) d', f=F) # [B*F, D]

        if cond is not None:
            if cond.dim() == 5:
                cond = cond.mean(dim=2) # [B, C, H, W]
                cond = rearrange(cond, 'b d h w -> b d (h w)')
                cond = F_torch.adaptive_avg_pool1d(cond, 1).squeeze(-1)
            elif cond.dim() == 4:
                cond = rearrange(cond, 'b d h w -> b d (h w)')
                cond = F_torch.adaptive_avg_pool1d(cond, 1).squeeze(-1)
            elif cond.dim() == 3:
                cond = F_torch.adaptive_avg_pool1d(cond, 1).squeeze(-1)
            elif cond.dim() == 2:
                pass
            else:
                raise ValueError(f"Unsupported cond shape: {cond.shape}")
            if self.cond_proj is None or self.cond_proj.in_features != cond.shape[-1]:
                self.cond_proj = nn.Linear(cond.shape[-1], self.dim).to(cond.device)

            cond = self.cond_proj(cond) # [B, dim]
            cond = repeat(cond, 'b d -> (b f) d', f=F)
            c = t_emb + cond
        else:
            c = t_emb

        if len(self.blocks) == 0:
            self.build_blocks(seq_len=H * W)
            self.blocks.to(device)

        for blk in self.blocks:
            x = blk(x, c) # [B*F, L, D]

        x_t = rearrange(x, '(b f) n d -> (n b) f d', b=B, f=F)
        x_t, _ = self.temp_attn(x_t, x_t, x_t)
        x = rearrange(x_t, '(n b) f d -> (b f) n d', b=B, f=F)

        x = rearrange(x, '(b f) (h w) d -> b d f h w', b=B, f=F, h=H, w=W)
        return self.to_out(x)

    def forward_with_cond_scale(self, x, t, cond=None, cond_scale=1.0, **kwargs):
        if cond is None or cond_scale is None or float(cond_scale) == 1.0:
            return self.forward(x, t, cond=cond)
        out_cond = self.forward(x, t, cond=cond)
        out_uncond = self.forward(x, t, cond=None)
        return out_uncond + (out_cond - out_uncond) * float(cond_scale)