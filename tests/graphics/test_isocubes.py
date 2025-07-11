from dataclasses import dataclass

from rfstudio.engine.task import Task
from rfstudio.graphics import IsoCubes
from rfstudio.visualization import Visualizer


@dataclass
class Test(Task):

    viser: Visualizer = Visualizer(port=6789)

    def run(self) -> None:
        isocubes = IsoCubes.from_resolution(32, device=self.device)
        sphere_sdfs = isocubes.vertices.norm(dim=-1, keepdim=True) - 0.8 # [..., 1]
        cube_sdfs = (isocubes.vertices.abs() - 0.9).max(-1, keepdim=True).values
        with self.viser.customize() as handle:
            mesh = isocubes.replace(sdf_values=sphere_sdfs).marching_cubes()
            handle['sphere'].show(mesh).configurate(normal_size=0.05)
            mesh = isocubes.replace(sdf_values=cube_sdfs).marching_cubes()
            handle['cube'].show(mesh).configurate(normal_size=0.05)


if __name__ == '__main__':
    Test(cuda=0).run()
