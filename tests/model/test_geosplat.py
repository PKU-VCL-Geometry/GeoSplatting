from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model import GeoSplatter
from rfstudio.trainer import GeoSplatTrainer

car_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'car',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='car'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

coffee_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'coffee',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='coffee'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

ball_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'ball',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='ball'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

helmet_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'helmet',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='helmet'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

teapot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'teapot',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.7,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='teapot'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

toaster_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'toaster',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='toaster'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

lego_highres_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'lego',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=128,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='lego_highres'),
    trainer=GeoSplatTrainer(
        num_steps=1500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        sdf_reg_begin=0.4,
        sdf_reg_end=0.2,
        sdf_reg_decay=1000,
        num_steps_per_val=50,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_air_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'air_baloons',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.95,
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_air'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_chair_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'chair',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_chair'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_hotdog_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'hotdog',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_hotdog'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_jugs_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'jugs',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.9,
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_jugs'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        num_steps_per_val=100,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_arm_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'armadillo',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        initial_guess='diffuse',
        scale=0.85,
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_arm'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_lego_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'lego',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_lego'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_hotdog_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'hotdog',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_hotdog'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_ficus_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'ficus',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=120,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_ficus'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        hold_after_train=False,
        full_test_after_train=False,
        mixed_precision=False
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
        assert isinstance(model, GeoSplatter)
        model.export_model(self.output)

if __name__ == '__main__':
    TaskGroup(

        # Shiny Blender
        ball=ball_task,
        car=car_task,
        coffee=coffee_task,
        helmet=helmet_task,
        teapot=teapot_task,
        toaster=toaster_task,

        # Synthetic4Relight dataset
        s4r_air=s4r_air_task,
        s4r_chair=s4r_chair_task,
        s4r_hotdog=s4r_hotdog_task,
        s4r_jugs=s4r_jugs_task,

        # TensoIR dataset
        tsir_arm=tsir_arm_task,
        tsir_lego=tsir_lego_task,
        tsir_ficus=tsir_ficus_task,
        tsir_hotdog=tsir_hotdog_task,

        export=Export(cuda=0),
    ).run()
