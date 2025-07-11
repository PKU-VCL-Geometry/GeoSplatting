from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics.shaders import PrettyShader
from rfstudio.model import GeoSplatterMC
from rfstudio.trainer import GeoSplatMCTrainer
from rfstudio.ui import console

shiny_blender_task = {}

for scene in ['car', 'coffee', 'ball', 'helmet', 'teapot', 'toaster']:
    shiny_blender_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'refnerf' / scene,
        ),
        model=GeoSplatterMC(
            load=Path('exports') / f'{scene}.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_mc', timestamp=scene),
        trainer=GeoSplatMCTrainer(
            num_steps=1000,
            batch_size=8,
            num_steps_per_val=25,
            normal_grad_reg_decay=250,
            sdf_reg_decay=1000,
            mixed_precision=False,
            full_test_after_train=False,
            hold_after_train=False,
        ),
        cuda=0,
        seed=1
    )

s4r_task = {}

for scene in ['air_baloons', 'jugs', 'chair', 'hotdog']:
    if scene == 'air_baloons':
        desc = 's4r_air'
    else:
        desc = f's4r_{scene}'
    s4r_task[desc] = TrainTask(
        dataset=RelightDataset(
            path=Path('data') / 'Synthetic4Relight' / scene,
        ),
        model=GeoSplatterMC(
            load=Path('exports') / f'{desc}.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_mc', timestamp=desc),
        trainer=GeoSplatMCTrainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
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
        model=GeoSplatterMC(
            load=Path('exports') / f'{desc}.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_mc', timestamp=desc),
        trainer=GeoSplatMCTrainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
            hold_after_train=False,
        ),
        cuda=0,
        seed=1
    )

tsir_task['lego_highres'] = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'lego',
    ),
    model=GeoSplatterMC(
        load=Path('exports') / 'lego_highres.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_mc', timestamp='lego_highres'),
    trainer=GeoSplatMCTrainer(
        num_steps=500,
        batch_size=8,
        num_steps_per_val=25,
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
        assert isinstance(model, GeoSplatterMC)
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
            assert isinstance(model, GeoSplatterMC)
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

if __name__ == '__main__':
    TaskGroup(
        **s4r_task,
        **tsir_task,
        **shiny_blender_task,
        export=Export(cuda=0),
        mesh=MeshRenderer(cuda=0),
    ).run()
