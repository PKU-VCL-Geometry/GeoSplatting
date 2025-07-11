from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, TriangleMesh
from rfstudio.graphics.shaders import DepthShader
from rfstudio.visualization import Visualizer


@dataclass
class Tester(Task):

    model: Path = ...
    viser: Visualizer = Visualizer(port=6789)

    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.model).to(self.device).normalize()
        camera = Cameras.from_lookat(
            eye=(1.5, 1.5, 1.5),
            resolution=(512, 512),
            near=0.5,
            far=4,
            device=mesh.device,
        ).to(mesh.device)
        depth = mesh.render(camera, shader=DepthShader(culling=False))
        with self.viser.customize() as handle:
            handle['deprojected'].show(depth.deproject(camera, alpha_threshold=1))
            handle['mesh'].show(mesh)


if __name__ == '__main__':
    Tester(
        model=Path('data') / 'spot' / 'spot_triangulated.obj',
        cuda=0,
    ).run()
