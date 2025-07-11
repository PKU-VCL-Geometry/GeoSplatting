from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.graphics import TriangleMesh
from rfstudio.visualization import Visualizer


@dataclass
class Tester(Task):

    model: Path = ...
    viser: Visualizer = Visualizer(port=6789)
    subdivide: bool = True

    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.model).normalize()
        with self.viser.customize() as handle:
            handle['mesh'].show(mesh).configurate(normal_size=0.05)
            handle['wireframe'].show(mesh).configurate(shade='wireframe')
            if self.subdivide:
                handle['subdivide'].show(mesh.subdivide()).configurate(normal_size=0.05)


if __name__ == '__main__':
    Tester(model=Path('data') / 'spot' / 'spot_triangulated.obj').run()
