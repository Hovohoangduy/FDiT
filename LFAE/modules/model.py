from torch import nn
import torch
import torch.nn.functional as F
from LFAE.modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad
import clip
import torchvision.transforms as T

class CLIPEncoder(nn.Module):
    def __init__(self, requires_grad=False):
        super(CLIPEncoder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = self.model.float()
        self.visual = self.model.visual

        if not requires_grad:
            for param in self.visual.parameters():
                param.requires_grad = False
        
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
    
    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if x.min().item() < 0.0:
            x = (x + 1) / 2.0
        x = torch.clamp(x, 0, 1)
        x = self.normalize(x)
        x = x.to(self.device).float()
        return self.visual(x)

class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints.
    """

    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']),
                                                       type=self.theta.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            # TODO this part may have bugs
            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class ReconstructionModel(torch.nn.Module):
    """
    Merge all updates into single model for better multi-gpu usage
    """

    def __init__(self, region_predictor, bg_predictor, generator, train_params):
        super(ReconstructionModel, self).__init__()
        self.region_predictor = region_predictor
        self.bg_predictor = bg_predictor
        self.generator = generator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        percep_w = self.loss_weights['perceptual']

        if (percep_w if isinstance(percep_w, (int, float)) else sum(percep_w)) != 0:
            self.clip_encoder = CLIPEncoder()
            if torch.cuda.is_available():
                self.clip_encoder = self.clip_encoder.cuda()

    def forward(self, x):
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        bg_params = self.bg_predictor(x['source'], x['driving'])
        generated = self.generator(x['source'], source_region_params=source_region_params,
                                   driving_region_params=driving_region_params, bg_params=bg_params)
        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
        percep_w = self.loss_weights['perceptual']
        if (percep_w if isinstance(percep_w, (int, float)) else sum(percep_w)) != 0:
            value_total = 0
            for idx, scale in enumerate(self.scales):
                x_feat = self.clip_encoder(pyramide_generated['prediction_' + str(scale)])
                y_feat = self.clip_encoder(pyramide_real['prediction_' + str(scale)])

                value = 1 - F.cosine_similarity(x_feat, y_feat, dim=-1).mean()

                if isinstance(percep_w, (int, float)):
                    value_total = value_total + percep_w * value
                else:
                    weight = percep_w[idx] if idx < len(percep_w) else percep_w[-1]
                    value_total = value_total + weight * value

            loss_values['perceptual'] = value_total

        if (self.loss_weights['equivariance_shift'] + self.loss_weights['equivariance_affine']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_region_params = self.region_predictor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_region_params'] = transformed_region_params

            if self.loss_weights['equivariance_shift'] != 0:
                value = torch.abs(driving_region_params['shift'] -
                                  transform.warp_coordinates(transformed_region_params['shift'])).mean()
                loss_values['equivariance_shift'] = self.loss_weights['equivariance_shift'] * value

            if self.loss_weights['equivariance_affine'] != 0:
                affine_transformed = torch.matmul(transform.jacobian(transformed_region_params['shift']),
                                                  transformed_region_params['affine'])

                normed_driving = torch.inverse(driving_region_params['affine'])
                normed_transformed = affine_transformed
                value = torch.matmul(normed_driving, normed_transformed)
                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                if self.generator.pixelwise_flow_predictor.revert_axis_swap:
                    value = value * torch.sign(value[:, :, 0:1, 0:1])

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_affine'] = self.loss_weights['equivariance_affine'] * value
            for k, v in loss_values.items():
                if not torch.is_tensor(v):
                    loss_values[k] = torch.tensor(
                        v, dtype=torch.float32, device=x['source'].device
                    )

        return loss_values, generated
