from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, TriangleMesh
from rfstudio.graphics.shaders import DepthShader, FlatShader, NormalShader, PrettyShader, SSAOShader
from rfstudio.io import dump_float32_image
from rfstudio.utils.colormap import IntensityColorMap


@dataclass
class Tester(Task):

    model: Path = ...
    output: Path = ...
    z_up: bool = False

    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.model).normalize().to(self.device)
        occlusion = mesh.compute_ambient_occlusion() # [F]
        camera = Cameras.from_lookat(
            eye=(-1.2, -1.2, 1.2) if self.z_up else (-1.2, 1.2, -1.2),
            resolution=(1024, 1024),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            hfov_degree=60,
            near=1e-2,
            far=1e2,
            device=mesh.device,
        ).to(mesh.device)
        output = torch.cat((
            mesh.render(camera, shader=PrettyShader(antialias=True, z_up=self.z_up)).rgb2srgb().blend((1, 1, 1)).item(),
            mesh.render(camera, shader=SSAOShader(antialias=True)).visualize(IntensityColorMap('gray')).item(),
            mesh.render(camera, shader=NormalShader()).visualize((1, 1, 1)).item(),
            mesh.render(
                camera,
                shader=FlatShader(face_colors=IntensityColorMap('gray').from_scaled(occlusion.view(-1, 1))),
            ).blend((1, 1, 1)).item(),
            mesh.render(camera, shader=DepthShader()).visualize().item(),
        ), dim=1)
        dump_float32_image(self.output, output.clamp(0, 1))


if __name__ == '__main__':
    Tester(
        model=Path('data/spot/spot.obj'),
        output=Path('temp.png'),
        cuda=0,
    ).run()
