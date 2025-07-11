from __future__ import annotations

import pathlib

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.train import TrainTask
from rfstudio.model import GSplatter
from rfstudio.trainer import GSplatTrainer

train_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=pathlib.Path('data') / 'spot',
    ),
    model=GSplatter(
        background_color='black',
        sh_degree=0
    ),
    experiment=Experiment(name='spot'),
    trainer=GSplatTrainer(
        num_steps=30000,
        batch_size=1,
        num_steps_per_val=300,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

if __name__ == '__main__':
    train_task.run()
