from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, TriangleMesh
from rfstudio.io import dump_float32_image


@dataclass
class Tester(Task):

    model: Path = ...
    output: Path = ...

    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.model).to(self.device)
        mesh = mesh.normalize().compute_vertex_normals()
        camera = Cameras.from_lookat(
            eye=(1.5, 1.5, 1.5),
            resolution=(512, 512),
            near=0.5,
            far=4,
            device=mesh.device,
        ).to(mesh.device)
        depth = mesh.render_depth(camera)
        pseudo_normal = depth.compute_pseudo_normals(camera)
        normal = mesh.render_normals(camera)
        img = torch.cat((
            depth.visualize(max_bound=camera.far.item()).item(),
            pseudo_normal.visualize((0, 0, 0)).item(),
            normal.visualize((0, 0, 0)).item(),
        ), dim=-2)
        dump_float32_image(self.output, img)


if __name__ == '__main__':
    Tester(
        model=Path('data') / 'spot' / 'spot_triangulated.obj',
        output=Path('temp.png'),
        cuda=0,
    ).run()
