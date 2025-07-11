from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from rfstudio.engine.task import Task
from rfstudio.visualization import vis_colmap


@dataclass
class VisColmap(Task):

    path: Path = ...

    port: int = ...

    pcd: Optional[Path] = None

    backend: Literal['viser', 'custom'] = 'custom'

    def run(self) -> None:
        vis_colmap(self.path, port=self.port, pcd_source=self.pcd, backend=self.backend)


if __name__ == '__main__':
    VisColmap(port=6789).run()
