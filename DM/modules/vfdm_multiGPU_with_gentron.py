import os
import torch
import torch.nn as nn
import yaml
from einops import rearrange, repeat
from sync_batchnorm import DataParallelWithCallback
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
from DM.modules.text import tokenize, bert_embed
from DM.modules.vfd_multiGPU import GaussianDiffusion

# ----- GenTron components -----
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half, device=x.device) / (half - 1))
        args = x[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class DiffusionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, dim)
        )
        self.to_patch = None
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
            ]) for _ in range(depth)
        ])
        self.temp_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # output channels match flow(2) + conf(1) + feature(256)
        self.to_out = nn.Conv3d(dim, 2 + 1 + 256, 1)

    def forward(self, x, t, *args, **kwargs):
        B, C, F, H, W = x.shape
        if self.to_patch is None:
            self.to_patch = nn.Conv3d(C, self.to_out.in_channels, kernel_size=(1,7,7), padding=(0,3,3)).to(x.device)
        x = self.to_patch(x)
        x = rearrange(x, 'b c f h w -> (b f) (h w) c')
        t_emb = self.time_mlp(t)
        t_emb = repeat(t_emb, 'b d -> (b f) n d', f=F, n=H*W)
        x = x + t_emb
        for norm1, attn, norm2, mlp in self.blocks:
            h = norm1(x)
            x = x + attn(h, h, h)[0]
            x = x + mlp(norm2(x))
        x, _ = self.temp_attn(x, x, x)
        x = rearrange(x, '(b f) (h w) c -> b c f h w', b=B, f=F, h=H, w=W)
        return self.to_out(x)

class GaussianDiffusionGenTron(GaussianDiffusion):
    def __init__(self, denoise_fn, **kwargs):
        super().__init__(denoise_fn, **kwargs)

# ----- Multi-GPU FlowDiffusionGenTron -----
class FlowDiffusionGenTronMultiGPU(nn.Module):
    def __init__(
        self,
        img_size=32,
        num_frames=40,
        sampling_timesteps=250,
        null_cond_prob=0.1,
        ddim_sampling_eta=1.0,
        timesteps=1000,
        dim=64,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=256,
        lr=1e-4,
        learn_null_cond=False,
        padding_mode="zeros",
        pretrained_pth="",
        config_pth="",
        is_train=True
    ):
        super().__init__()
        cfg = yaml.safe_load(open(config_pth))

        # load pretrained modules
        self.generator = Generator(
            num_regions=cfg['model_params']['num_regions'],
            num_channels=cfg['model_params']['num_channels'],
            revert_axis_swap=cfg['model_params']['revert_axis_swap'],
            **cfg['model_params']['generator_params']
        ).cuda(); self.generator.eval()
        self.region_predictor = RegionPredictor(
            num_regions=cfg['model_params']['num_regions'],
            num_channels=cfg['model_params']['num_channels'],
            estimate_affine=cfg['model_params']['estimate_affine'],
            **cfg['model_params']['region_predictor_params']
        ).cuda(); self.region_predictor.eval()
        self.bg_predictor = BGMotionPredictor(
            num_channels=cfg['model_params']['num_channels'],
            **cfg['model_params']['bg_predictor_params']
        ).cuda(); self.bg_predictor.eval()

        # GenTron denoiser
        denoiser = DiffusionTransformer(dim, depth, heads, dim_head, mlp_dim, time_emb_dim=dim)
        self.diffusion = GaussianDiffusionGenTron(
            denoiser,
            image_size=img_size,
            num_frames=num_frames,
            sampling_timesteps=sampling_timesteps,
            timesteps=timesteps,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta,
            loss_type='l2',
            use_dynamic_thres=True
        ).cuda()

        self.is_train = is_train
        if is_train:
            self.optim = torch.optim.Adam(self.diffusion.parameters(), lr=lr)

    def forward(self, real_vid, ref_img, ref_text):
        # embed text
        cond = bert_embed(tokenize(ref_text), return_cls_repr=self.diffusion.text_use_bert_cls).cuda()
        flow_list, conf_list = [], []
        with torch.no_grad():
            src_params = self.region_predictor(ref_img)
            for idx in range(real_vid.size(2)):
                drv = real_vid[:, :, idx]
                drv_params = self.region_predictor(drv)
                bg_params = self.bg_predictor(ref_img, drv)
                gen = self.generator(ref_img, src_params, drv_params, bg_params)
                flow_list.append(gen['optical_flow'].permute(0,3,1,2))
                conf_list.append(gen['occlusion_map'])
        flow = torch.stack(flow_list, 2)
        loss = self.diffusion(flow, gen['bottle_neck_feat'], cond)
        return loss

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    bs, img_size, nf = 8, 128, 40
    real_vid = torch.randn(bs, 3, nf, img_size, img_size).cuda()
    ref_img = torch.randn(bs, 3, img_size, img_size).cuda()
    ref_text = ['test'] * bs

    model = FlowDiffusionGenTronMultiGPU(
        img_size=img_size,
        num_frames=nf,
        config_pth='config.yaml',
        pretrained_pth='checkpoint.pth'
    )
    model = DataParallelWithCallback(model)
    loss = model(real_vid, ref_img, ref_text)
    print('Loss:', loss.item())
