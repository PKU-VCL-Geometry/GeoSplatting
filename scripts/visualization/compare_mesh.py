from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from rfstudio.engine.task import Task
from rfstudio.graphics import TriangleMesh
from rfstudio.visualization import Visualizer


@dataclass
class Compare(Task):

    input: Tuple[str, ...] = ...
    viser: Visualizer = Visualizer(port=6789)
    show_normal: bool = False

    def run(self) -> None:
        inputs = sum([
            list(Path('.').glob(p)) if '*' in p else [Path(p)]
            for p in self.input
        ], [])
        meshes = [TriangleMesh.from_file(m) for m in inputs]
        with self.viser.customize() as handle:
            for filename, mesh in zip(inputs, meshes):
                handle[filename.stem].show(mesh).configurate(normal_size=0.05 if self.show_normal else None)
            handle['aabb'].show(TriangleMesh.create_cube(size=2)).configurate(shade='wireframe')

if __name__ == '__main__':
    Compare().run()
