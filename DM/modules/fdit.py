import os
import math
import torch
import torch.nn as nn
import yaml
from LFAE.modules.generator import Generator
from LFAE.modules.bg_motion_predictor import BGMotionPredictor
from LFAE.modules.region_predictor import RegionPredictor
from DM.modules.vfd import GaussianDiffusion
from DM.modules.dit import DiT

class FDiT(nn.Module):
    def __init__(self, img_size, num_frames, sampling_timesteps, null_cond_prob,
                 ddim_sampling_eta, timesteps, dim, depth, heads, dim_head,
                 mlp_dim, lr, adam_betas, is_train,
                 only_use_flow, use_residual_flow,
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
            self.generator.load_state_dict(ckpt['generator']); self.generator.eval(); self.set_requires_grad(self.generator, False)
            self.region_predictor.load_state_dict(ckpt['region_predictor']); self.region_predictor.eval(); self.set_requires_grad(self.region_predictor, False)
            self.bg_predictor.load_state_dict(ckpt['bg_predictor']); self.bg_predictor.eval(); self.set_requires_grad(self.bg_predictor, False)
        else:
            self.generator.eval(); self.set_requires_grad(self.generator, False)
            self.region_predictor.eval(); self.set_requires_grad(self.region_predictor, False)
            self.bg_predictor.eval(); self.set_requires_grad(self.bg_predictor, False)

        in_channels = 259

        self.dit = DiT(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            time_emb_dim=dim,
            out_channels=259,
            in_channels=in_channels
        ).cuda()


        self.diffusion = GaussianDiffusion(
            self.dit,
            image_size=img_size,
            num_frames=num_frames,
            sampling_timesteps=sampling_timesteps,
            timesteps=timesteps,
            null_cond_prob=null_cond_prob,
            ddim_sampling_eta=ddim_sampling_eta,
            loss_type='l2',
            use_dynamic_thres=True,
            is_transformer=True
        ).cuda()

        if is_train:
            self.optimizer_diff = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=adam_betas)
            self.diffusion.train()

        # placeholders for visualization
        self.real_out_vid = None
        self.real_warped_vid = None
        self.fake_out_vid = None
        self.fake_warped_vid = None
        self.real_vid_grid = None
        self.fake_vid_grid = None
        self.real_vid_conf = None
        self.fake_vid_conf = None

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
        grid, conf, feat_list = [], [], []

        with torch.no_grad():
            src = self.region_predictor(self.ref_img)
            for i in range(F):
                drv = self.real_vid[:, :, i] # [B, 3, H, W]
                drv_p = self.region_predictor(drv)
                bg_p = self.bg_predictor(self.ref_img, drv)
                out = self.generator(self.ref_img, src, drv_p, bg_p)

                grid.append(out['optical_flow'].permute(0, 3, 1, 2)) # [B, 2, H/4, W/4]
                conf.append(out['occlusion_map'].unsqueeze(2)) # [B, 1, 1, H/4, W/4]
                feat_list.append(out['bottle_neck_feat'].unsqueeze(2)) # [B, 256, 1, H/4, W/4]

        flow = torch.stack(grid, 2) # [B, 2, F, H/4, W/4]
        conf_grid = torch.cat(conf, dim=2) # [B, 1, F, H/4, W/4]
        feat = torch.cat(feat_list, dim=2) # [B, 256, F, H/4, W/4]

        self.real_vid_grid = flow
        self.fake_vid_grid = flow
        self.real_vid_conf = conf_grid
        self.fake_vid_conf = conf_grid

        if self.use_residual_flow:
            idg = self.get_grid(B, F, flow.shape[-2], flow.shape[-1])
            flow = flow - idg

        feat_with_conf = torch.cat([feat, conf_grid], dim=1) # [B, 257, F, H, W]
        input_x = torch.cat([flow, feat_with_conf], dim=1) # [B, 259, F, H, W]

        den = self.diffusion.denoise_fn
        if len(den.blocks) == 0:
            den.build_blocks(seq_len=flow.shape[-2] * flow.shape[-1])
            den.blocks.to(flow.device)

        if not hasattr(self, '_logged_in_ch'):
            print(f"[FDiT] to_patch.in_channels = {den.to_patch.in_channels}, "
                  f"dim = {den.dim}, flowC = {flow.shape[1]}, featC = {feat.shape[1]}, confC = {conf_grid.shape[1]}")
            self._logged_in_ch = True

        fea = self.generator.compute_fea(self.ref_img) # [B, 256, H/4, W/4]
        self.loss = self.diffusion(input_x, fea, self.ref_text)

        try:
            mask = self.diffusion.denoise_fn.null_cond_mask
            if mask.numel() == 0 or mask.shape[0] != B:
                self.diffusion.denoise_fn.null_cond_mask = torch.zeros(B, dtype=torch.bool, device=flow.device)
        except Exception:
            self.diffusion.denoise_fn.null_cond_mask = torch.zeros(B, dtype=torch.bool, device=flow.device)

        return self.loss

    def optimize_parameters(self):
        self.optimizer_diff.zero_grad()
        l = self.forward()
        l.backward()
        self.optimizer_diff.step()
        self.rec_loss = float(l.detach().item())
        self.rec_warp_loss = float(l.detach().item())

    @torch.no_grad()
    def sample_video(self, sample_img, sample_text, cond_scale=1.0):
        feat = self.generator.compute_fea(sample_img.cuda())
        vid = self.diffusion.sample(feat, cond=sample_text, cond_scale=cond_scale)
        self.real_out_vid = vid
        self.real_warped_vid = vid
        self.fake_out_vid = vid
        if self.real_vid_grid is not None:
            self.fake_vid_grid = self.real_vid_grid
        return vid

    def get_grid(self, B, F, H, W):
        h = torch.linspace(-1, 1, H, device='cuda')
        w = torch.linspace(-1, 1, W, device='cuda')
        g = torch.stack(torch.meshgrid(h, w, indexing='ij'), -1).flip(2) # [H, W, 2] -> (x,y)
        g = g.unsqueeze(0).unsqueeze(2).repeat(B, 1, F, 1, 1) # [B, H, F, W, 2]
        return g.permute(0, 4, 2, 1, 3).contiguous() # [B, 2, F, H, W]
    
    @torch.no_grad()
    def sample_one_video(self, cond_scale):
        self.sample_img_fea = self.generator.compute_fea(self.sample_img)

        # [B, 2, F, H, W] (u, v)
        pred = self.diffusion.sample(
            self.sample_img_fea,
            cond=self.sample_text,
            batch_size=1,
            cond_scale=cond_scale
        )  # [B, 2, F, H, W]

        flow_uv = pred[:, :2, :, :, :] # [B, 2, F, H, W]
        if self.use_residual_flow:
            b, _, nf, h, w = flow_uv.size()
            identity_grid = self.get_grid(b, nf, h, w).cuda() # [B, 2, F, H, W]
            self.sample_vid_grid = flow_uv + identity_grid
        else:
            self.sample_vid_grid = flow_uv
        b, _, nf, h, w = self.sample_vid_grid.size()
        self.sample_vid_conf = torch.ones(b, 1, nf, h, w, device=self.sample_vid_grid.device)
        self.fake_vid_conf = self.sample_vid_conf
        sample_out_img_list = []
        sample_warped_img_list = []
        for idx in range(nf):
            sample_grid = self.sample_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1) # [B, H, W, 2]
            sample_conf = self.sample_vid_conf[:, :, idx, :, :] # [B, 1, H, W]
            generated = self.generator.forward_with_flow(
                source_image=self.sample_img,
                optical_flow=sample_grid,
                occlusion_map=sample_conf
            )
            sample_out_img_list.append(generated["prediction"])
            sample_warped_img_list.append(generated["deformed"])

        self.sample_out_vid    = torch.stack(sample_out_img_list, dim=2) # [B, 3, F, H, W]
        self.sample_warped_vid = torch.stack(sample_warped_img_list, dim=2)

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
