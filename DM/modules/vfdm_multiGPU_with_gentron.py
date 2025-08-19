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

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / (half - 1))
        args = x[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim)
        self.kw = nn.Linear(dim, dim)
        self.vw = nn.Linear(dim, dim)
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        self.ow = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)
        B, L, D = q.shape
        q = q.view(B, L, self.n_heads, -1).transpose(1, 2)
        k = k.view(B, L, self.n_heads, -1).transpose(1, 2).transpose(-1, -2)
        v = v.view(B, L, self.n_heads, -1).transpose(1, 2).transpose(-1, -2)
        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
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
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(2)

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
        self.temp_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.null_cond_mask = torch.tensor([], dtype=torch.bool)

    def build_blocks(self, seq_len):
        self.blocks = nn.ModuleList([
            TransformerBlock(seq_len=seq_len, dim=self.dim, heads=4, mlp_dim=self.dim * 4, k=self.seq_k)
            for _ in range(3)
        ])

    def forward(self, x, t):
        B, C, F, H, W = x.shape
        if self.to_patch is None:
            self.to_patch = nn.Conv3d(C, self.dim, self.kernel, padding=self.padding).to(x.device)
        x = self.to_patch(x)
        x = rearrange(x, 'b d f h w -> (b f) (h w) d')

        t_emb = self.time_mlp(t)
        t_emb = repeat(t_emb, 'b d -> (b f) n d', f=F, n=H * W)

        if len(self.blocks) == 0:
            self.build_blocks(seq_len=H * W)

        for blk in self.blocks:
            x = blk(x, t_emb)

        x_t = rearrange(x, '(b f) n d -> (n b) f d', b=B, f=F)
        x_t, _ = self.temp_attn(x_t, x_t, x_t)
        x = rearrange(x_t, '(n b) f d -> (b f) n d', b=B, f=F)
        x = rearrange(x, '(b f) (h w) d -> b d f h w', b=B, f=F, h=H, w=W)
        return self.to_out(x)

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

    # For GenTron, forward signature remains same as GaussianDiffusion
    # No additional overrides needed unless custom sampling required

class FlowDiffusionGenTron(nn.Module):
    def __init__(self, img_size, num_frames, sampling_timesteps, null_cond_prob,
                 ddim_sampling_eta, timesteps, dim, depth, heads, dim_head,
                 mlp_dim, lr, adam_betas, is_train,
                 use_residual_flow,
                 pretrained_pth, config_pth):
        super().__init__()
        self.use_residual_flow = use_residual_flow
        cfg = yaml.safe_load(open(config_pth))
        ckpt = torch.load(pretrained_pth) if pretrained_pth else None

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

        denoiser = DiffusionTransformer(dim, depth, heads, dim_head, mlp_dim, time_emb_dim=dim, out_channels=2)
        self.diffusion = GaussianDiffusionGenTron(
            denoiser, image_size=img_size, num_frames=num_frames,
            sampling_timesteps=sampling_timesteps, timesteps=timesteps,
            null_cond_prob=null_cond_prob, ddim_sampling_eta=ddim_sampling_eta,
            loss_type='l2', use_dynamic_thres=True
        ).cuda()

        if is_train:
            self.optimizer_diff = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=adam_betas)

                # placeholders for visualization
        self.real_out_vid = None
        self.real_warped_vid = None
        self.fake_out_vid = None
        self.fake_warped_vid = None
        # placeholder for real video grid
        self.real_vid_grid = None
        # placeholder for fake video grid
        self.fake_vid_grid = None
        # placeholder for real video confidence map
        self.real_vid_conf = None
        # placeholder for fake video confidence map
        self.fake_vid_conf = None
        self.fake_vid_grid = None

    def set_train_input(self, ref_img, real_vid, ref_text):
        self.ref_img = ref_img.cuda()
        self.real_vid = real_vid.cuda()
        self.ref_text = ref_text
        self.real_out_vid = real_vid.cuda()
        self.real_warped_vid = real_vid.cuda()
        self.fake_out_vid = real_vid.cuda()
        self.fake_warped_vid = real_vid.cuda()

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
                grid.append(out['optical_flow'].permute(0, 3, 1, 2))
                conf.append(out['occlusion_map'])
        flow = torch.stack(grid, 2)
        # store the flow grid for visualization
        self.real_vid_grid = flow
        # initialize fake video grid placeholder
        self.fake_vid_grid = flow
        # stack occlusion/confidence maps for visualization
        conf_grid = torch.stack(conf, 2)
        self.real_vid_conf = conf_grid
        self.fake_vid_conf = conf_grid
        feat = out['bottle_neck_feat'].detach().detach()
        if self.use_residual_flow:
            idg = self.get_grid(B, F, H//4, W//4)
            flow = flow - idg
        self.loss = self.diffusion(flow, feat, self.ref_text)
        # ensure null_cond_mask exists and has correct shape
        try:
            mask = self.diffusion.denoise_fn.null_cond_mask
            if mask.numel() == 0:
                # default: no null conditioning
                self.diffusion.denoise_fn.null_cond_mask = torch.zeros(B, dtype=torch.bool, device=flow.device)
        except Exception:
            self.diffusion.denoise_fn.null_cond_mask = torch.zeros(B, dtype=torch.bool, device=flow.device)

        return self.loss

    def sample_one_video(self, sample_img, sample_text, cond_scale):
        output_dict = {}
        sample_img_fea = self.generator.compute_fea(sample_img)
        bs = sample_img_fea.size(0)
        # if cond_scale = 1.0, not using unconditional model
        pred = self.diffusion.sample(sample_img_fea, cond=sample_text,
                                     batch_size=bs, cond_scale=cond_scale)
        if self.use_residual_flow:
            b, _, nf, h, w = pred[:, :2, :, :, :].size()
            identity_grid = self.get_grid(b, nf, h, w, normalize=True).cuda()
            output_dict["sample_vid_grid"] = pred[:, :2, :, :, :] + identity_grid
        else:
            output_dict["sample_vid_grid"] = pred[:, :2, :, :, :]
        output_dict["sample_vid_conf"] = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
        nf = output_dict["sample_vid_grid"].size(2)
        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(nf):
                sample_grid = output_dict["sample_vid_grid"][:, :, idx, :, :].permute(0, 2, 3, 1)
                sample_conf = output_dict["sample_vid_conf"][:, :, idx, :, :]
                # predict fake out image and fake warped image
                generated = self.generator.forward_with_flow(source_image=sample_img,
                                                             optical_flow=sample_grid,
                                                             occlusion_map=sample_conf)
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
        output_dict["sample_out_vid"] = torch.stack(sample_out_img_list, dim=2)
        output_dict["sample_warped_vid"] = torch.stack(sample_warped_img_list, dim=2)
        return output_dict

    def get_grid(self, b, nf, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid.permute(0, 3, 1, 2).unsqueeze(dim=2).repeat(1, 1, nf, 1, 1)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad