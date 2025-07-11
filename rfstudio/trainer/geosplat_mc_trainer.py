from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, RelightDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model import GeoSplatterMC
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class GeoSplatMCTrainer(BaseTrainer):

    cov3d_lr: float = 3e-3
    geometry_lr: float = 3e-3
    appearance_lr: float = 1e-2
    light_lr: float = 1e-2

    base_decay: Optional[int] = 800

    base_eps: float = 1e-15

    sdf_reg_begin: float = 0.2
    sdf_reg_end: float = 0.01
    sdf_reg_decay: int = 500

    kd_grad_reg_begin: float = 0.03
    kd_grad_reg_end: float = 0.03
    kd_grad_reg_decay: Optional[int] = 250
    kd_regualr_perturb_std: float = 0.01

    ks_grad_reg_begin: float = 0.03
    ks_grad_reg_end: float = 0.03
    ks_grad_reg_decay: Optional[int] = 250
    ks_regualr_perturb_std: float = 0.01

    normal_grad_reg_begin: float = 0.03
    normal_grad_reg_end: float = 0.03
    normal_grad_reg_decay: Optional[int] = 0

    use_mask_loss: bool = True
    visual_mode: Literal['default', 'production'] = 'default'

    def setup(
        self,
        model: GeoSplatterMC,
        dataset: Union[MeshViewSynthesisDataset, RelightDataset, MultiViewDataset],
    ) -> ModuleOptimizers:
        assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset, MultiViewDataset))

        model.latlng.register_hook(lambda g: g * 64.0)
        model.occ_weight = 0.001

        optim_dict = {
            'deforms': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='deforms'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
                warm_up=50,
            ),
            'weights': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='weights'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
                warm_up=50,
            ),
            'kd': Optimizer(
                category=torch.optim.Adam,
                modules=model.field.kd_enc,
                lr=self.appearance_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'occ': Optimizer(
                category=torch.optim.Adam,
                modules=model.field.occ_enc,
                lr=self.appearance_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'ks': Optimizer(
                category=torch.optim.Adam,
                modules=model.field.ks_enc,
                lr=self.appearance_lr * 0.2,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'z': Optimizer(
                category=torch.optim.Adam,
                modules=model.field.z_enc,
                lr=self.cov3d_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'exposure': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='exposure'),
                lr=self.light_lr * 0.5,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'sdfs': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='sdfs'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
                warm_up=50,
            ),
            'light': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='light'),
                lr=self.light_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
        }
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: GeoSplatterMC,
        inputs: Cameras,
        gt_outputs: RGBAImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        bg_color = model.get_background_color().to(model.device)
        gt_pbra = gt_outputs.srgb2rgb()
        (
            pbra,
            vis,
            mesh_normal,
            num_gaussians,
            reg_loss,
        ) = model.render_report(inputs, indices=None, gt_outputs=gt_outputs)

        gt_rgba = gt_outputs

        losses = []
        assert isinstance(gt_rgba, RGBAImages)
        for pbra_item, gt_pbra_item in zip(pbra, gt_pbra, strict=True):
            train_bg_color = torch.rand_like(pbra_item[..., :3]) # [H, W, 3]
            mask = gt_pbra_item[..., 3:] # [H, W, 1]
            img1 = pbra_item[..., :3] + (1 - pbra_item[..., 3:]) * train_bg_color
            img2 = gt_pbra_item[..., :3] * mask + (1 - mask) * train_bg_color
            loss = SSIML1Loss()._impl(img1, img2)
            if self.use_mask_loss:
                loss = loss + 5 * (mask - pbra_item[..., 3:]).square().mean()
            losses.append(loss)
        loss = sum(losses) / len(losses)

        rgb = RGBImages([
            rgba_item[..., :3] + (1 - rgba_item[..., 3:]) * bg_color
            for rgba_item in pbra.detach().rgb2srgb()
        ])
        gt_rgb = gt_rgba.blend(bg_color)
        splat_psnr = PSNRLoss()(gt_rgb, rgb.clamp(0, 1)) # srgb space metrics
        metrics = {
            'loss': loss.detach(),
            'splat-psnr': splat_psnr,
            '#gaussians': num_gaussians,
            'regularization': reg_loss.detach(),
            'exposure': model.exposure_params.detach().mean().exp(),
        }

        image = None
        if visual:
            if self.visual_mode == 'production':
                image = torch.cat((
                    rgb[0].item(),
                    vis[0].item(),
                    vis[6].item(),
                    vis[5].item(),
                ), dim=1).clamp(0, 1)   # [H, 5W, 3]
            else:
                row1 = torch.cat((
                    rgb[0].item(),
                    vis[4].item(), # normal
                    mesh_normal[0].visualize(bg_color).item(),
                    gt_rgb[0].item(),
                ), dim=1)                                                  # [H, 3W, 3]
                row2 = torch.cat((
                    vis[0].item(), # kd
                    bg_color.expand_as(vis[0].item()),
                    vis[2].item(), # occ_diff
                    vis[3].item(), # occ_spec
                ), dim=1)                                                  # [H, 3W, 3]
                row3 = torch.cat((
                    vis[1].item()[..., 2:3].expand_as(vis[0].item()), # metallic
                    vis[1].item()[..., 1:2].expand_as(vis[0].item()), # roughness
                    vis[5].item(), # light
                ), dim=1)                                                  # [H, 3W, 3]
                image = torch.cat((row1, row2, row3), dim=0).clamp(0, 1)   # [3H, 3W, 3]

        return loss + reg_loss, metrics, image

    def before_update(
        self,
        model: GeoSplatterMC,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:

        if self.sdf_reg_decay > 0:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )

        if self.kd_grad_reg_decay > 0:
            model.kd_grad_weight = (
                self.kd_grad_reg_begin -
                (self.kd_grad_reg_begin - self.kd_grad_reg_end) * min(1.0, curr_step / self.kd_grad_reg_decay)
            )
            model.kd_regualr_perturb_std = self.kd_regualr_perturb_std

        if self.ks_grad_reg_decay > 0:
            model.ks_grad_weight = (
                self.ks_grad_reg_begin -
                (self.ks_grad_reg_begin - self.ks_grad_reg_end) * min(1.0, curr_step / self.ks_grad_reg_decay)
            )
            model.ks_regualr_perturb_std = self.ks_regualr_perturb_std

        if self.normal_grad_reg_decay > 0:
            model.normal_grad_weight = (
                self.normal_grad_reg_begin -
                (self.normal_grad_reg_begin - self.normal_grad_reg_end) * min(1.0, curr_step / self.normal_grad_reg_decay)
            )

    def after_update(self, model: GeoSplatterMC, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if curr_step % 5 == 0:
            torch.cuda.empty_cache()
        model.latlng.data.clamp_min_(1e-2)
