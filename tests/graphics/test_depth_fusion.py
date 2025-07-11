from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.data import DepthSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, DepthImages, TriangleMesh
from rfstudio.model import GSplatter
from rfstudio.optim import ModuleOptimizers, Optimizer
from rfstudio.trainer import GSplatTrainer
from rfstudio.ui import console
from rfstudio.visualization import Visualizer


@dataclass
class GSplatDepthTrainer(GSplatTrainer):

    def setup(
        self,
        model: GSplatter,
        dataset: DepthSynthesisDataset,
    ) -> ModuleOptimizers:
        assert isinstance(dataset, DepthSynthesisDataset)

        self._dataset_size = dataset.get_size(split='train')

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
        assert model.sh_degree == 0
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: GSplatter,
        inputs: Cameras,
        gt_outputs: DepthImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        outputs = model.render_depth(inputs)
        losses = [
            torch.nn.functional.l1_loss(
                output[..., :1] * gt_output[..., 1:],
                gt_output[..., :1] * gt_output[..., 1:],
            )
            for output, gt_output in zip(outputs, gt_outputs, strict=True)
        ]
        loss = sum(losses) / len(losses)
        metrics = {
            'l1': loss.detach(),
            '#gaussians': model.gaussians.shape[0],
        }
        image = None
        if visual:
            with torch.no_grad():
                normal = outputs.compute_pseudo_normals(inputs[0]).visualize(model.get_background_color())
                gt_normal = gt_outputs.compute_pseudo_normals(inputs[0]).visualize(model.get_background_color())
            image = torch.cat((normal.item(), gt_normal.item()), dim=1).clamp(0, 1)
        return loss, metrics, image

@dataclass
class GS2Mesh(Task):

    load: Path = ...
    viser: Visualizer = Visualizer(port=6789)
    resolution: int = 256
    alpha_threshold: float = 0.1

    def filter_rendering(self, cameras: Cameras, model: GSplatter) -> DepthImages:
        depths = []
        for camera in cameras.view(-1, 1):
            depth = model.render_depth(camera).item()
            filtered_alpha = depth[..., 1:].clamp_min(self.alpha_threshold) - self.alpha_threshold
            depths.append(torch.cat((
                depth[..., :1],
                filtered_alpha / (1 - self.alpha_threshold),
            ), dim=-1))
        return DepthImages(depths)

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        cameras = Cameras.from_sphere(
            center=(0, 0, 0),
            up=(0, 0, 1),
            radius=3,
            num_samples=100,
            resolution=(800, 800),
            device=self.device
        )

        with console.progress('TSDF Fusion') as handle:
            fused = TriangleMesh.from_depth_fusion(
                self.filter_rendering(cameras, model),
                cameras=cameras,
                progress_handle=handle,
                sdf_trunc=10. / self.resolution,
                voxel_size=2. / self.resolution,
            )
        with self.viser.customize() as handle:
            handle['fused'].show(fused).configurate(normal_size=0.02)

@dataclass
class Syn2Mesh(Task):

    dataset: DepthSynthesisDataset = DepthSynthesisDataset(path=...)
    viser: Visualizer = Visualizer(port=6789)
    resolution: int = 256

    def run(self) -> None:
        self.dataset.to(self.device)
        cameras = self.dataset.get_inputs(split='train')[...]
        depths = self.dataset.get_gt_outputs(split='train')[...]
        mesh = self.dataset.get_meta(split='train')

        with console.progress('TSDF Fusion') as handle:
            fused = TriangleMesh.from_depth_fusion(
                depths,
                cameras=cameras,
                progress_handle=handle,
                sdf_trunc=10. / self.resolution,
                voxel_size=2. / self.resolution,
            )
        with self.viser.customize() as handle:
            if mesh is not None:
                handle['gt'].show(mesh)
            handle['fused'].show(fused)


if __name__ == '__main__':
    TaskGroup(
        lego=TrainTask(
            dataset=DepthSynthesisDataset(
                path=Path('data') / 'lego',
            ),
            model=GSplatter(
                background_color='white',
                sh_degree=0,
                prepare_densification=True,
            ),
            experiment=Experiment(name='gsplat_lego_depth'),
            trainer=GSplatDepthTrainer(
                num_steps_per_val=500,
                num_steps=30000,
                batch_size=1,
                mixed_precision=False,
                full_test_after_train=False,
                densify_grad_thresh=2e-5,
                densify_size_thresh=1e-2,
            ),
            cuda=0,
            seed=1,
        ),
        s2m=Syn2Mesh(cuda=0),
        g2m=GS2Mesh(cuda=0),
    ).run()
