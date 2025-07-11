from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.graphics import Points
from rfstudio.visualization import Visualizer


@dataclass
class VisColmap(Task):

    input: Path = ...

    viser: Visualizer = Visualizer(port=6789)

    def run(self) -> None:
        points = Points.from_file(self.input)
        self.viser.show(pcd=points)


if __name__ == '__main__':
    VisColmap().run()
