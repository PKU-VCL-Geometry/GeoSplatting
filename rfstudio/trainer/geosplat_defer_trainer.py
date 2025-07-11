from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model import GeoSplatterDefer
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class GeoSplatDeferTrainer(BaseTrainer):

    base_lr: float = 1e-3
    light_lr: float = 1e-3

    base_decay: Optional[int] = 500

    base_eps: float = 1e-15

    loss: SSIML1Loss = SSIML1Loss()

    fix_material: bool = False

    kd_reg: float = 0.2
    ks_reg: float = 0.05
    normal_reg: float = 0.0

    use_mask_loss: bool = False

    def setup(
        self,
        model: GeoSplatterDefer,
        dataset: Union[MeshViewSynthesisDataset, RelightDataset],
    ) -> ModuleOptimizers:
        assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset))

        optim_dict = {
            'light_hue': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='light_hue'),
                lr=self.light_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'light_value': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='light_value'),
                lr=self.light_lr,
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
            'means': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='means'),
                lr=self.base_lr * 0.16,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'scales': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='scales'),
                lr=self.base_lr * 5,
                eps=self.base_eps,
            ),
            'quats': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='quats'),
                lr=self.base_lr,
                eps=self.base_eps,
            ),
            'normals': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='normals'),
                lr=self.base_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'opacities': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='opacities'),
                lr=self.base_lr * 50,
                eps=self.base_eps,
            ),
        }
        if not self.fix_material:
            optim_dict['kd'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='kd'),
                lr=self.base_lr * 5,
                eps=self.base_eps,
            )
            optim_dict['ks'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.ks_enc,
                lr=self.base_lr * 0.5,
                eps=self.base_eps,
            )
            optim_dict['occ'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='occ'),
                lr=self.base_lr * 2.5,
                eps=self.base_eps,
            )

        model.latlng_value.register_hook(lambda g: g * 64)
        model.latlng_hue.register_hook(lambda g: g * 64)

        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: GeoSplatterDefer,
        inputs: Cameras,
        gt_outputs: Union[RGBAImages, RGBImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        bg_color = model.get_background_color().to(model.device)
        gt_rgba = gt_outputs.clamp(0, 1)
        gt_pbra = gt_rgba.srgb2rgb() # gt rgb space

        (
            pbra,
            vis,
            mesh_normal,
            num_gaussians,
            reg_loss,
        ) = model.render_report(
            inputs,
            indices=indices if training else None,
            gt_images=gt_pbra,
        )

        losses = []
        for pbra_item, gt_pbra_item in zip(pbra, gt_pbra, strict=True):
            train_bg_color = torch.rand_like(pbra_item[..., :3])
            mask = gt_pbra_item[..., 3:] # [H, W, 1]
            loss = self.loss._impl(
                pbra_item[..., :3] + (1 - pbra_item[..., 3:]) * train_bg_color,
                gt_pbra_item[..., :3] * mask + (1 - mask) * train_bg_color,
            )
            if self.use_mask_loss:
                loss = loss + 5 * (mask - pbra_item[..., 3:]).square().mean()
            losses.append(loss)
        loss = sum(losses) / len(losses)

        rgb = pbra.rgb2srgb().blend(model.get_background_color())
        gt_rgb = gt_rgba.blend(bg_color)
        splat_psnr = PSNRLoss()(gt_rgb, rgb.clamp(0, 1)) # srgb space metrics
        metrics = {
            'loss': loss.detach(),
            'splat-psnr': splat_psnr,
            '#gaussians': num_gaussians,
            'regularization': reg_loss.detach(),
            'exposure': model.exposure.detach().exp(),
        }

        image = None
        if visual:
            row1 = torch.cat((
                rgb[0].item(),
                mesh_normal[0].item(),
                vis[2].item(),
                gt_rgb[0].item(),
            ), dim=1)                                                  # [H, 4W, 3]
            row2 = torch.cat((
                vis[0].item(),
                vis[1].item(),
                vis[3].item(),
                vis[4].item(),
            ), dim=1)                                                  # [H, 4W, 3]
            image = torch.cat((row1, row2), dim=0).clamp(0, 1)   # [2H, 4W, 3]

        return loss + reg_loss, metrics, image

    def before_update(
        self,
        model: GeoSplatterDefer,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        model.kd_weight = self.kd_reg
        model.ks_weight = self.ks_reg
        model.normal_weight = self.normal_reg

    def after_update(self, model: GeoSplatterDefer, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if curr_step % 2 == 0:
            torch.cuda.empty_cache()
        if model.gt_envmap is None:
            model.latlng_hue.data.clamp_(0.01, 0.99)
        model.kd_params.data.clamp_(0.01, 0.99)
