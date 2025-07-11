from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, Points, RGBAImages, RGBImages, Splats
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model import GSplatter
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class GSplatTrainer(BaseTrainer):

    base_lr: float = 1e-3

    base_eps: float = 1e-15

    pos_lr_decay: int = 4500

    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    num_splits: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    stop_split_at: int = 15000
    """stop splitting at this step"""

    normal_weight: float = 5e-2
    normal_weight_start: int = 7000
    distort_weight: float = 1e-2
    distort_weight_start: int = 3000

    loss: SSIML1Loss = SSIML1Loss(ssim_lambda=0.2)

    def setup(
        self,
        model: GSplatter,
        dataset: Union[SfMDataset, MultiViewDataset, MeshViewSynthesisDataset],
    ) -> ModuleOptimizers:
        assert isinstance(dataset, (SfMDataset, MultiViewDataset, MeshViewSynthesisDataset))
        if isinstance(dataset, SfMDataset):
            model.gaussians = Splats.from_points(
                dataset.get_meta(split='train').view_as(Points),
                sh_degree=model.sh_degree,
                device=model.device,
                requires_grad=model.gaussians.requires_grad
            )

        self._dataset_size = dataset.get_size(split='train')
        self._normal_weight_enable = False
        self._distort_weight_enable = False

        optim_dict = {
            'means': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='means'),
                lr=self.base_lr * 0.16,
                eps=self.base_eps,
                lr_decay=self.pos_lr_decay
            ),
            'scales': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='scales'),
                lr=self.base_lr * 5,
                eps=self.base_eps
            ),
            'quats': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='quats'),
                lr=self.base_lr,
                eps=self.base_eps
            ),
            'colors': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='colors'),
                lr=self.base_lr * 2.5,
                eps=self.base_eps
            ),
            'opacities': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='opacities'),
                lr=self.base_lr * 50,
                eps=self.base_eps
            )
        }
        if model.sh_degree > 0:
            optim_dict['shs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='shs'),
                lr=self.base_lr * 0.125,
                eps=self.base_eps
            )
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: GSplatter,
        inputs: Cameras,
        gt_outputs: Union[RGBAImages, RGBImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        if isinstance(gt_outputs, RGBAImages):
            gt_outputs = gt_outputs.blend(model.get_background_color())
        gt_outputs = gt_outputs.clamp(0, 1)
        outputs = model.render_rgb(inputs)
        loss = self.loss(gt_outputs, outputs)
        if model.rasterize_mode == '2dgs':
            if self._normal_weight_enable:
                loss = loss + model.update_info.normal_loss * self.normal_weight
            if self._distort_weight_enable:
                loss = loss + model.update_info.distort_loss * self.distort_weight
        metrics = {
            'ssim': loss.detach(),
            'psnr': PSNRLoss()(gt_outputs, outputs.detach()),
            '#gaussians': model.gaussians.shape[0]
        }
        image = None
        if visual:
            with torch.no_grad():
                depth = model.render_depth(inputs)
                normal = depth.compute_pseudo_normals(inputs[0]).visualize(model.get_background_color())
            image = torch.cat((outputs.item(), normal.item()), dim=1).clamp(0, 1)
        return loss, metrics, image

    def before_update(
        self,
        model: GSplatter,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        model.set_max_sh_degree(curr_step // self.sh_degree_interval)
        if curr_step > self.normal_weight_start:
            self._normal_weight_enable = True
        if curr_step > self.distort_weight_start:
            self._distort_weight_enable = True

    def after_update(
        self,
        model: GSplatter,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        model.set_max_sh_degree(None)

        if curr_step < self.stop_split_at:
            model.update_grad_norm()

        if curr_step > self.warmup_length and curr_step % self.refine_every == 0:
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            reset_interval = self.reset_alpha_every * self.refine_every

            # only split/cull if we've seen every image since opacity reset
            if all([
                curr_step < self.stop_split_at,
                curr_step % reset_interval > self._dataset_size + self.refine_every,
            ]):
                # ui.log('densify and cull @ step ', curr_step)
                indices = model.gaussians.densify_and_cull(
                    densify_grad_thresh=self.densify_grad_thresh,
                    densify_size_thresh=self.densify_size_thresh,
                    num_splits=self.num_splits,
                    cull_alpha_thresh=self.cull_alpha_thresh,
                    cull_scale_thresh=(
                        self.cull_scale_thresh
                        if curr_step > self.refine_every * self.reset_alpha_every
                        else None
                    )
                )
                optimizers.mutate_params(indices=indices)

            if all([
                curr_step >= self.stop_split_at,
                self.continue_cull_post_densification,
            ]):
                # ui.log('post cull @ step ', curr_step)
                indices = model.gaussians.cull(
                    cull_alpha_thresh=self.cull_alpha_thresh,
                    cull_scale_thresh=(
                        self.cull_scale_thresh
                        if curr_step > self.refine_every * self.reset_alpha_every
                        else None
                    )
                )
                optimizers.mutate_params(indices=indices)

            if all([
                curr_step < self.stop_split_at,
                curr_step % reset_interval == self.refine_every
            ]):
                # ui.log('reset opacities @ step ', curr_step)
                model.gaussians.reset_opacities(reset_value=self.cull_alpha_thresh * 2.0)
                optimizers['opacities'].mutate_params(clear=True)

            model.gaussians.clear_extras_()
