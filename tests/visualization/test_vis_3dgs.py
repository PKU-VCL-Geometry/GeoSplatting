from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rfstudio.engine.task import Task
from rfstudio.engine.train import TrainTask
from rfstudio.model import GSplatter, MeshSplatter
from rfstudio.visualization import vis_3dgs


@dataclass
class Vis3DGS(Task):

    load: Path = ...

    port: int = ...

    backend: Literal['viser', 'custom'] = 'custom'

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, (GSplatter, MeshSplatter))
        vis_3dgs(model, port=self.port, backend=self.backend)


if __name__ == '__main__':
    Vis3DGS(port=6789).run()
