from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, RGBAImages, TextureLatLng
from rfstudio.graphics.shaders import PrettyShader
from rfstudio.model import GeoSplatterPrior
from rfstudio.trainer import GeoSplatPriorTrainer
from rfstudio.ui import console

shiny_blender_task = {}

for scene in ['car', 'coffee', 'ball', 'helmet', 'teapot', 'toaster']:
    shiny_blender_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'refnerf' / scene,
        ),
        model=GeoSplatterPrior(
            load=Path('exports') / 'prior' / f'{scene}.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_prior', timestamp=scene),
        trainer=GeoSplatPriorTrainer(
            num_steps=1000,
            batch_size=6,
            num_steps_per_val=25,
            normal_grad_reg_decay=250,
            mixed_precision=False,
            full_test_after_train=False,
            hold_after_train=False,
        ),
        cuda=0,
        seed=1
    )

tsir_task = {}

for scene in ['lego', 'armadillo', 'ficus', 'hotdog']:
    if scene == 'armadillo':
        desc = 'tsir_arm'
    else:
        desc = f'tsir_{scene}'
    tsir_task[desc] = TrainTask(
        dataset=RelightDataset(
            path=Path('data') / 'tensoir' / scene,
        ),
        model=GeoSplatterPrior(
            load=Path('exports') / 'prior' / f'{desc}.ply',
            background_color='white',
            z_up_to_y_up=True,
        ),
        experiment=Experiment(name='geosplat_prior', timestamp=desc),
        trainer=GeoSplatPriorTrainer(
            num_steps=500,
            batch_size=6,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
            hold_after_train=False,
        ),
        cuda=0,
        seed=1
    )

truck_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'tnt' / 'blender' / 'Truck',
    ),
    model=GeoSplatterPrior(
        load=Path('exports') / 'prior' / 'truck_adjusted.ply',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_prior', timestamp='truck'),
    trainer=GeoSplatPriorTrainer(
        geometry_lr=4e-5,
        num_steps=500,
        batch_size=6,
        num_steps_per_val=25,
        mixed_precision=False,
        full_test_after_train=False,
        hold_after_train=False,
    ),
    cuda=0,
    seed=1
)

courthouse_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'tnt' / 'blender' / 'Courthouse',
    ),
    model=GeoSplatterPrior(
        load=Path('exports') / 'prior' / 'courthouse_adjusted.ply',
        background_color='white',
        smooth_type='grad',
    ),
    experiment=Experiment(name='geosplat_prior', timestamp='courthouse'),
    trainer=GeoSplatPriorTrainer(
        geometry_lr=1e-5,
        cov3d_lr=1e-5,
        kd_grad_reg_begin=0.01,
        kd_grad_reg_end=0.01,
        ks_grad_reg_begin=0.002,
        ks_grad_reg_end=0.002,
        num_steps=500,
        batch_size=6,
        num_steps_per_val=5,
        mixed_precision=False,
        full_test_after_train=False,
        hold_after_train=False,
    ),
    cuda=0,
    seed=1
)

@dataclass
class Export(Task):

    load: Path = ...
    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        self.output.parent.mkdir(exist_ok=True, parents=True)
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GeoSplatterPrior)
        model.export_model(self.output)

@dataclass
class MeshRenderer(Task):

    load: Path = ...

    step: Optional[int] = None

    z_up: bool = True

    wireframe: bool = True

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            dataset = train_task.dataset
            assert isinstance(model, GeoSplatterPrior)
            assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset))
            mesh, _ = model.get_geometry()
        with console.progress(desc='Rendering Test View', transient=True) as ptrack:
            bg_color = model.get_background_color()
            idx = 0
            for inputs, gt_outputs, _ in ptrack(dataset.get_test_iter(1), total=dataset.get_size(split='test')):
                pbra, _, _, _, _ = model.render_report(inputs, indices=None, gt_outputs=gt_outputs)
                rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                mesh_rgb = mesh.render(
                    inputs[0].resize(2),
                    shader=PrettyShader(occlusion_type='none', z_up=self.z_up, wireframe=self.wireframe),
                ).rgb2srgb().blend(bg_color).resize_to(800, 800).item().clamp(0, 1)
                train_task.experiment.dump_image('pbr', index=idx, image=rgb)
                train_task.experiment.dump_image('mesh', index=idx, image=mesh_rgb)
                idx += 1

@dataclass
class Relighter(Task):

    load: Path = ...

    envmap: Path = ...

    hfov_degree: float = 60.

    z_up: bool = False

    envmap_as_bg: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            dataset = train_task.dataset
            assert isinstance(model, GeoSplatterPrior)
            assert isinstance(dataset, MeshViewSynthesisDataset)
        cameras = []
        for radius in [0.4, 0.6]:
            for pitch in [17]:
                cameras.append(Cameras.from_orbit(
                    center=(-0.2, -0.2, 0),
                    up=(0, 0, 1) if self.z_up else (0, 1, 0),
                    radius=radius,
                    pitch_degree=pitch,
                    num_samples=80,
                    resolution=(960, 540),
                    hfov_degree=self.hfov_degree,
                    device=self.device,
                ))
        cameras = Cameras.cat(cameras, dim=0)[49:50]
        cameras = cameras.replace(c2w=dataset.get_inputs(split='train')[...][4:5].c2w)
        bg_color = model.get_background_color()

        if self.envmap.exists():
            envmap = TextureLatLng.from_image_file(self.envmap, device=self.device).as_cubemap(resolution=512)
            if self.z_up:
                envmap.z_up_to_y_up_()
            envmap = envmap.as_latlng(
                width=model.latlng.shape[1],
                height=model.latlng.shape[0],
                apply_transform=True,
            )
            model.latlng.data.copy_(envmap.data)
            name = self.envmap.stem
        else:
            envmap = None
            name = 'nvs'

        with console.progress(desc='Rendering') as ptrack:
            for i, camera in enumerate(ptrack(cameras)):
                pbra, vis = model.render_report(
                    camera.view(-1),
                    gt_outputs=RGBAImages(torch.ones(960, 540, 4)).to(self.device),
                    indices=None,
                )[:2]
                if not self.envmap_as_bg or envmap is None:
                    rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                else:
                    bg = envmap.as_cubemap(resolution=512).render(camera)
                    rgb = pbra.rgb2srgb().clamp(0, 1).blend_background(bg).item().clamp(0, 1)
                # train_task.experiment.dump_image(name, index=i, image=rgb[145:-115, 80:-160])
                # if envmap is None:
                #     train_task.experiment.dump_image('albedo', index=i, image=vis[0].item()[145:-115, 80:-160])
                train_task.experiment.dump_image(name, index=i, image=rgb[100:-115-145+100, 80:-160])
                if envmap is None:
                    train_task.experiment.dump_image('albedo', index=i, image=vis[0].item()[100:-115-145+100, 80:-160])

if __name__ == '__main__':
    TaskGroup(
        **tsir_task,
        **shiny_blender_task,
        truck=truck_task,
        courthouse=courthouse_task,
        export=Export(cuda=0),
        relit=Relighter(cuda=0),
        mesh=MeshRenderer(cuda=0),
    ).run()
