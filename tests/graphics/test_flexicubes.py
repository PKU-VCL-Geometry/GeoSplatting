from __future__ import annotations

from dataclasses import dataclass

from rfstudio.engine.task import Task
from rfstudio.graphics import FlexiCubes
from rfstudio.visualization import Visualizer


@dataclass
class TestFlexiCubes(Task):

    viser: Visualizer = Visualizer(port=6789)

    def run(self) -> None:
        flexicubes = FlexiCubes.from_resolution(32)
        sphere_sdfs = flexicubes.vertices.norm(dim=-1, keepdim=True) - 0.8 # [..., 1]
        cube_sdfs = (flexicubes.vertices.abs() - 0.9).max(-1, keepdim=True).values
        with self.viser.customize() as handle:
            mesh = flexicubes.replace(sdf_values=sphere_sdfs).dual_marching_cubes()[0]
            handle['sphere'].show(mesh).configurate(normal_size=0.05)
            mesh = flexicubes.replace(sdf_values=cube_sdfs).dual_marching_cubes()[0]
            handle['cube'].show(mesh).configurate(normal_size=0.05)


if __name__ == '__main__':
    TestFlexiCubes(cuda=0).run()
