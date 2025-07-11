from __future__ import annotations

import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.data.dataparser import RFMaskedRealDataparser
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, DepthImages, RGBAImages, TriangleMesh
from rfstudio.io import dump_float32_image
from rfstudio.loss import LPIPSLoss, PSNRLoss, SSIML1Loss
from rfstudio.model import GSplatter
from rfstudio.trainer import GSplatTrainer
from rfstudio.ui import console
from rfstudio.utils.pretty import P

tasks = {}
for scene in ['chair', 'lego', 'ficus', 'mic', 'ship', 'materials', 'drums', 'hotdog']:
    tasks[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=pathlib.Path('data') / 'blender' / scene,
        ),
        model=GSplatter(
            background_color='white',
            sh_degree=3,
            prepare_densification=True,
        ),
        experiment=Experiment(name='3dgs', timestamp=scene),
        trainer=GSplatTrainer(
            num_steps=30000,
            batch_size=1,
            num_steps_per_val=1000,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1,
    )

mip360_tasks = {}
for scene in ['garden']:
    tasks[scene] = TrainTask(
        dataset=SfMDataset(
            path=pathlib.Path('data') / 'mip360' / scene,
        ),
        model=GSplatter(
            background_color='white',
            sh_degree=3,
            prepare_densification=True,
        ),
        experiment=Experiment(name='3dgs', timestamp=scene),
        trainer=GSplatTrainer(
            num_steps=30000,
            batch_size=1,
            num_steps_per_val=1000,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1,
    )

for scene in ['chair', 'lego', 'ficus', 'mic', 'ship', 'materials', 'drums', 'hotdog']:
    tasks[scene + '-2d'] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=pathlib.Path('data') / 'blender' / scene,
        ),
        model=GSplatter(
            background_color='white',
            sh_degree=3,
            prepare_densification=True,
            rasterize_mode='2dgs',
        ),
        experiment=Experiment(name='2dgs', timestamp=scene),
        trainer=GSplatTrainer(
            num_steps=30000,
            batch_size=1,
            num_steps_per_val=1000,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1,
    )


@dataclass
class NVS(Task):

    load: Path = ...

    view: int = ...

    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        dataset = train_task.dataset
        test_view = dataset.get_inputs(split='test')[self.view]
        rgb = model.render_rgb(test_view.view(-1))
        dump_float32_image(self.output, rgb.item().clamp(0, 1))


@dataclass
class Masker(Task):

    load: Path = ...

    preview: Optional[int] = None

    export: Path = ...

    scale: float = 1

    center: Tuple[float, float, float] = (0, 0, 0)

    floor: Optional[float] = None

    alpha_theshold: float = 0.8

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        dataset = train_task.dataset
        assert isinstance(model, GSplatter)
        assert isinstance(dataset, (SfMDataset, MultiViewDataset))
        train_views = dataset.get_inputs(split='train')
        center = torch.tensor(self.center).float().to(self.device)
        selected = (model.gaussians.means - center).square().sum(-1) <= self.scale ** 2
        if self.floor is not None:
            selected &= model.gaussians.means[..., 2] > self.floor
        model.gaussians = model.gaussians[selected].contiguous()
        self.export.mkdir(exist_ok=True, parents=True)
        if self.preview is not None:
            rgb = model.render_rgb(train_views[self.preview].view(-1))
            dump_float32_image(self.export / Path('preview.png'), rgb.item().clamp(0, 1))
        else:
            with console.status('Dumping'):
                imgs = []
                all_cameras = []
                for split in ['train', 'val', 'test']:
                    cameras = dataset.get_inputs(split=split)[...].clone()
                    for camera, rgb in zip(cameras, dataset.get_gt_outputs(split=split)[...], strict=True):
                        alpha = (model.render_rgba(camera.view(-1)).item()[..., 3:] > self.alpha_theshold).float()
                        imgs.append(torch.cat((rgb * alpha, alpha), dim=-1))
                    cameras.c2w[..., :3, 3] = (cameras.c2w[..., :3, 3] - center) / self.scale + center
                    cameras.near.mul_(1 / self.scale)
                    cameras.far.mul_(1 / self.scale)
                    all_cameras.append(cameras)
                RFMaskedRealDataparser.dump(
                    Cameras.cat(all_cameras, dim=0),
                    RGBAImages(imgs),
                    None,
                    path=self.export,
                    split='all',
                )


@dataclass
class MeshExport(Task):

    load: Path = ...

    output: Path = ...

    resolution: int = 512

    fusing_alpha_threshold: float = 0.5

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        with console.progress(desc='TSDF Fusion') as handle:
            cameras = Cameras.from_sphere(
                center=(0, 0, 0),
                up=(0, 0, 1),
                radius=3,
                num_samples=100,
                resolution=(800, 800),
                device=self.device
            )
            fused_mesh = TriangleMesh.from_depth_fusion(
                DepthImages([model.render_depth(camera).item() for camera in cameras.view(-1, 1)]),
                cameras=cameras,
                progress_handle=handle,
                sdf_trunc=10. / self.resolution,
                voxel_size=2. / self.resolution,
                alpha_trunc=self.fusing_alpha_threshold,
            )
        fused_mesh.export(self.output, only_geometry=True)


@dataclass
class Evaler(Task):

    load: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GSplatter)
        dataset = train_task.dataset
        psnrs = []
        ssims = []
        lpipss = []
        with console.progress(scene) as ptrack:
            for inputs, gt_outputs, _ in ptrack(dataset.get_test_iter(1), total=dataset.get_size(split='test')):
                gt_outputs = gt_outputs.blend(model.get_background_color())
                rgb = model.render_rgb(inputs).clamp(0, 1)
                psnrs.append(PSNRLoss()(rgb, gt_outputs))
                ssims.append(1 - SSIML1Loss()(rgb, gt_outputs))
                lpipss.append(LPIPSLoss()(rgb, gt_outputs))

        psnr = torch.stack(psnrs).mean()
        ssim = torch.stack(ssims).mean()
        lpips = torch.stack(lpipss).mean()
        console.print(P@'PSNR: {psnr:.3f}')
        console.print(P@'SSIM: {ssim:.4f}')
        console.print(P@'LPIPS: {lpips:.4f}')

if __name__ == '__main__':
    TaskGroup(
        **tasks,
        **mip360_tasks,
        nvs=NVS(cuda=0),
        eval=Evaler(cuda=0),
        export=MeshExport(cuda=0),
        masker=Masker(cuda=0),
    ).run()
