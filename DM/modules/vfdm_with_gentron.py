# Full updated code with all fixes: device alignment, FiLM broadcasting, 3D attention input, and minor cleanups

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange, repeat
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
from DM.modules.vfd import GaussianDiffusion


# -----------------------------
# Utilities
# -----------------------------
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
    """
    x:     [B*, L, D]
    shift: [B*, D]
    scale: [B*, D]
    -> broadcast over L via unsqueeze(1)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# -----------------------------
# Linformer-style Attention (fixed)
# -----------------------------
class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim)
        self.kw = nn.Linear(dim, dim)
        self.vw = nn.Linear(dim, dim)
        # Optional projection matrices (not used explicitly below but kept for future linformer proj)
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        self.ow = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, L, D]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, L, D), but got {x.shape}")

        q = self.qw(x)  # [B, L, D]
        k = self.kw(x)  # [B, L, D]
        v = self.vw(x)  # [B, L, D]

        B, L, D = q.shape
        H = self.n_heads
        Dh = D // H

        q = q.view(B, L, H, Dh).transpose(1, 2)              # [B, H, L, Dh]
        k = k.view(B, L, H, Dh).transpose(1, 2).transpose(-1, -2)  # [B, H, Dh, L]
        v = v.view(B, L, H, Dh).transpose(1, 2)              # [B, H, L, Dh]

        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)  # [B, H, L, L]
        out = torch.matmul(attn, v)                                    # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)           # [B, L, D]
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
        # FiLM parameters
        self.gamma_1 = nn.Linear(dim, dim)
        self.beta_1 = nn.Linear(dim, dim)
        self.gamma_2 = nn.Linear(dim, dim)
        self.beta_2 = nn.Linear(dim, dim)
        self.scale_1 = nn.Linear(dim, dim)
        self.scale_2 = nn.Linear(dim, dim)
        self._init_weights([self.gamma_1, self.beta_1, self.gamma_2, self.beta_2, self.scale_1, self.scale_2])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c):
        """
        x: [B*, L, D]
        c: [B*, D]  (per-frame conditioning)
        """
        scale_msa = self.gamma_1(c)  # [B*, D]
        shift_msa = self.beta_1(c)   # [B*, D]
        scale_mlp = self.gamma_2(c)  # [B*, D]
        shift_mlp = self.beta_2(c)   # [B*, D]

        gate_msa = self.scale_1(c).unsqueeze(1)  # [B*, 1, D]
        gate_mlp = self.scale_2(c).unsqueeze(1)  # [B*, 1, D]

        x = self.attn(modulate(self.ln1(x), shift_msa, scale_msa)) * gate_msa + x
        x = self.mlp(modulate(self.ln2(x), shift_mlp, scale_mlp)) * gate_mlp + x
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, time_emb_dim, out_channels, seq_k=64):
        super().__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.kernel, self.padding = (1, 7, 7), (0, 3, 3)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, dim)
        )
        self.to_patch = None
        self.to_out = nn.Conv3d(dim, out_channels, 1)
        self.blocks = nn.ModuleList([])
        self.seq_k = seq_k
        self.heads = heads
        self.temp_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.register_buffer('null_cond_mask', torch.tensor([], dtype=torch.bool), persistent=False)

    def build_blocks(self, seq_len):
        self.blocks = nn.ModuleList([
            TransformerBlock(seq_len=seq_len, dim=self.dim, heads=self.heads, mlp_dim=self.dim * 4, k=self.seq_k)
            for _ in range(3)
        ])

    def forward(self, x, t, cond=None, *args, **kwargs):
        """
        x: [B, C, F, H, W]
        t: [B]
        cond: unused here (text can be threaded in higher-level diffusion if needed)
        """
        B, C, F, H, W = x.shape
        device = x.device

        # 3D conv to feature dim
        if self.to_patch is None:
            self.to_patch = nn.Conv3d(C, self.dim, self.kernel, padding=self.padding).to(device)

        x = self.to_patch(x)                            # [B, D, F, H, W]
        x = rearrange(x, 'b d f h w -> (b f) (h w) d')  # [B*F, L, D]  (L = H*W)
        assert x.dim() == 3, f"Expected 3D after rearrange, got {x.shape}"

        t = t.to(device)
        self.time_mlp.to(device)
        self.temp_attn.to(device)

        # Per-frame conditioning c: [B*F, D]
        t_emb = self.time_mlp(t)                        # [B, D]
        c = repeat(t_emb, 'b d -> (b f) d', f=F)        # [B*F, D]

        # Build blocks lazily with correct seq_len
        if len(self.blocks) == 0:
            self.build_blocks(seq_len=H * W)
            for blk in self.blocks:
                blk.to(device)

        # Spatial transformer blocks
        for blk in self.blocks:
            x = blk(x, c)  # keep 3D [B*F, L, D]

        # Temporal attention: operate across F for each spatial token
        x_t = rearrange(x, '(b f) n d -> (n b) f d', b=B, f=F)  # [N*B, F, D]
        x_t, _ = self.temp_attn(x_t, x_t, x_t)                  # [N*B, F, D]
        x = rearrange(x_t, '(n b) f d -> (b f) n d', b=B, f=F)  # [B*F, L, D]

        # Back to 5D and project out
        x = rearrange(x, '(b f) (h w) d -> b d f h w', b=B, f=F, h=H, w=W)
        return self.to_out(x).to(device)


# -----------------------------
# Diffusion wrapper (GenTron)
# -----------------------------
class GaussianDiffusionGenTron(GaussianDiffusion):
    """
    Gaussian diffusion tailored for GenTron, using 2-channel flow inputs.
    Overrides default channel dimension to match latent flow channels.
    """
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        sampling_timesteps=250,
        ddim_sampling_eta=1.,
        timesteps=1000,
        null_cond_prob=0.1,
        loss_type='l2',
        use_dynamic_thres=True
    ):
        # explicitly set channels=2 for flow (u,v)
        super().__init__(
            denoise_fn,
            image_size=image_size,
            num_frames=num_frames,
            channels=2,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            ddim_sampling_eta=ddim_sampling_eta,
            loss_type=loss_type,
            use_dynamic_thres=use_dynamic_thres,
            null_cond_prob=null_cond_prob
        )


# -----------------------------
# Main model
# -----------------------------
class FlowDiffusionGenTron(nn.Module):
    def __init__(self, img_size, num_frames, sampling_timesteps, null_cond_prob,
                 ddim_sampling_eta, timesteps, dim, depth, heads, dim_head,
                 mlp_dim, lr, adam_betas, is_train,
                 only_use_flow, use_residual_flow,
                 pretrained_pth, config_pth):
        super().__init__()
        self.use_residual_flow = use_residual_flow
        cfg = yaml.safe_load(open(config_pth))
        ckpt = torch.load(pretrained_pth) if pretrained_pth else None

        # ----- LFAE components -----
        self.generator = Generator(
            num_regions=cfg['model_params']['num_regions'],
            num_channels=cfg['model_params']['num_channels'],
            revert_axis_swap=cfg['model_params']['revert_axis_swap'],
            **cfg['model_params']['generator_params']
        ).cuda()
        self.region_predictor = RegionPredictor(
            num_regions=cfg['model_params']['num_regions'],
            num_channels=cfg['model_params']['num_channels'],
            estimate_affine=cfg['model_params']['estimate_affine'],
            **cfg['model_params']['region_predictor_params']
        ).cuda()
        self.bg_predictor = BGMotionPredictor(
            num_channels=cfg['model_params']['num_channels'],
            **cfg['model_params']['bg_predictor_params']
        ).cuda()

        if ckpt:
            self.generator.load_state_dict(ckpt['generator']); self.generator.eval(); self.generator.requires_grad_(False)
            self.region_predictor.load_state_dict(ckpt['region_predictor']); self.region_predictor.eval(); self.region_predictor.requires_grad_(False)
            self.bg_predictor.load_state_dict(ckpt['bg_predictor']); self.bg_predictor.eval(); self.bg_predictor.requires_grad_(False)

        # ----- Denoiser + Diffusion -----
        denoiser = DiffusionTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            time_emb_dim=dim,
            out_channels=2
        )
        self.diffusion = GaussianDiffusionGenTron(
            denoise_fn=denoiser,
            image_size=img_size,
            num_frames=num_frames,
            sampling_timesteps=sampling_timesteps,
            timesteps=timesteps,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta,
            loss_type='l2',
            use_dynamic_thres=True
        ).cuda()

        if is_train:
            self.optimizer_diff = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=adam_betas)

        # placeholders for visualization
        self.real_out_vid = None
        self.real_warped_vid = None
        self.fake_out_vid = None
        self.fake_warped_vid = None
        self.real_vid_grid = None
        self.fake_vid_grid = None
        self.real_vid_conf = None
        self.fake_vid_conf = None

    # --------- Public API ---------
    def set_train_input(self, ref_img, real_vid, ref_text):
        self.ref_img = ref_img.cuda()
        self.real_vid = real_vid.cuda()
        self.ref_text = ref_text
        # initialize vis holders
        self.real_out_vid = self.real_vid
        self.real_warped_vid = self.real_vid
        self.fake_out_vid = self.real_vid
        self.fake_warped_vid = self.real_vid

    def forward(self):
        B, _, F, H, W = self.real_vid.shape
        grid, conf = [], []
        with torch.no_grad():
            src = self.region_predictor(self.ref_img)
            for i in range(F):
                drv = self.real_vid[:, :, i]
                drv_p = self.region_predictor(drv)
                bg_p = self.bg_predictor(self.ref_img, drv)
                out = self.generator(self.ref_img, src, drv_p, bg_p)
                grid.append(out['optical_flow'].permute(0, 3, 1, 2))   # [B, 2, H/4, W/4]
                conf.append(out['occlusion_map'])
        flow = torch.stack(grid, 2)              # [B, 2, F, H/4, W/4]
        conf_grid = torch.stack(conf, 2)         # [B, 1, F, H/4, W/4]

        # store for visualization
        self.real_vid_grid = flow
        self.fake_vid_grid = flow
        self.real_vid_conf = conf_grid
        self.fake_vid_conf = conf_grid

        feat = out['bottle_neck_feat'].detach()  # [B, Df, F, H', W'] (as produced by generator)

        if self.use_residual_flow:
            idg = self.get_grid(B, F, flow.shape[-2], flow.shape[-1])  # identity grid at flow res
            flow = flow - idg

        self.loss = self.diffusion(flow, feat, self.ref_text)

        # ensure null_cond_mask exists with correct batch size (no null by default)
        try:
            mask = self.diffusion.denoise_fn.null_cond_mask
            if mask.numel() == 0 or mask.shape[0] != B:
                self.diffusion.denoise_fn.null_cond_mask = torch.zeros(B, dtype=torch.bool, device=flow.device)
        except Exception:
            self.diffusion.denoise_fn.null_cond_mask = torch.zeros(B, dtype=torch.bool, device=flow.device)

        return self.loss

    def optimize_parameters(self):
        l = self.forward()
        self.optimizer_diff.zero_grad()
        l.backward()
        self.optimizer_diff.step()
        self.rec_loss = l
        self.rec_warp_loss = l

    @torch.no_grad()
    def sample_video(self, sample_img, sample_text, cond_scale=1.0):
        feat = self.generator.compute_fea(sample_img.cuda())
        vid = self.diffusion.sample(feat, cond=sample_text, cond_scale=cond_scale)
        self.real_out_vid = vid
        self.real_warped_vid = vid
        self.fake_out_vid = vid
        # reuse last computed flow grid for visualization if available
        if self.real_vid_grid is not None:
            self.fake_vid_grid = self.real_vid_grid
        return vid

    def get_grid(self, B, F, H, W):
        # Create identity grid in [-1,1] at (H, W), layout [B, 2, F, H, W]
        h = torch.linspace(-1, 1, H, device='cuda')
        w = torch.linspace(-1, 1, W, device='cuda')
        g = torch.stack(torch.meshgrid(h, w, indexing='ij'), -1).flip(2)  # [H, W, 2] -> (x,y)
        g = g.unsqueeze(0).unsqueeze(2).repeat(B, 1, F, 1, 1)              # [B, H, F, W, 2]
        return g.permute(0, 4, 2, 0+3, 1)  # [B, 2, F, H, W]
        # Note: permute arguments expanded for clarity: dims (0,4,2,3,1)

    # Optional separate sampling API kept minimal and consistent
    @torch.no_grad()
    def sample_one_video(self, cond_scale):
        self.sample_img_fea = self.generator.compute_fea(self.sample_img)
        pred = self.diffusion.sample(self.sample_img_fea, cond=self.sample_text, batch_size=1, cond_scale=cond_scale)
        if self.use_residual_flow:
            b, _, nf, h, w = pred[:, :2, :, :, :].size()
            identity_grid = self.get_grid(b, nf, h, w).cuda()
            self.sample_vid_grid = pred[:, :2, :, :, :] + identity_grid
        else:
            self.sample_vid_grid = pred[:, :2, :, :, :]
        self.sample_vid_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
        nf = self.sample_vid_grid.size(2)
        sample_out_img_list = []
        sample_warped_img_list = []
        for idx in range(nf):
            sample_grid = self.sample_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
            sample_conf = self.sample_vid_conf[:, :, idx, :, :]
            generated = self.generator.forward_with_flow(source_image=self.sample_img,
                                                          optical_flow=sample_grid,
                                                          occlusion_map=sample_conf)
            sample_out_img_list.append(generated["prediction"])
            sample_warped_img_list.append(generated["deformed"])
        self.sample_out_vid = torch.stack(sample_out_img_list, dim=2)
        self.sample_warped_vid = torch.stack(sample_warped_img_list, dim=2)

    # Convenience setters
    def set_sample_input(self, sample_img, sample_text):
        self.sample_img = sample_img.cuda()
        self.sample_text = sample_text

    def print_learning_rate(self):
        lr = self.optimizer_diff.param_groups[0]['lr']
        assert lr > 0
        print('lr= %.7f' % lr)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
