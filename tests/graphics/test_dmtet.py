from dataclasses import dataclass

from rfstudio.engine.task import Task
from rfstudio.graphics import DMTet
from rfstudio.visualization import Visualizer


@dataclass
class TestDMTet(Task):

    viser: Visualizer = Visualizer(port=6789)

    def run(self) -> None:
        dmtet = DMTet.from_predefined(resolution=32, device=self.device)
        sphere_sdfs = dmtet.vertices.norm(dim=-1, keepdim=True) - 0.8 # [..., 1]
        cube_sdfs = (dmtet.vertices.abs() - 0.9).max(-1, keepdim=True).values
        with self.viser.customize() as handle:
            mesh = dmtet.replace(sdf_values=sphere_sdfs).marching_tets()
            handle['sphere'].show(mesh).configurate(normal_size=0.05)
            mesh = dmtet.replace(sdf_values=cube_sdfs).marching_tets()
            handle['cube'].show(mesh).configurate(normal_size=0.05)


if __name__ == '__main__':
    TestDMTet(cuda=0).run()
