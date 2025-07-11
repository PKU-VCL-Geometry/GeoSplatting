from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from torch.nn.functional import avg_pool2d

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, TriangleMesh
from rfstudio.graphics.shaders import (
    PrettyShader,
)
from rfstudio.io import dump_float32_image


@dataclass
class Tester(Task):

    model: Optional[Path] = ...
    output: Path = ...
    aa: bool = True
    z_up: bool = False
    flip: bool = False
    wireframe: bool = True

    def run(self) -> None:
        if self.model is None:
            mesh = TriangleMesh.create_sphere().to(self.device)
        else:
            mesh = TriangleMesh.from_file(self.model).normalize().to(self.device)
        flip_coeff = -1.0 if self.flip else 1.0
        camera = Cameras.from_lookat(
            eye=(-1.7 * flip_coeff, 1.7 * flip_coeff, 1.7) if self.z_up else (1.7 * flip_coeff, 1.7, -1.7 * flip_coeff),
            resolution=(1600, 1600),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            hfov_degree=45,
            near=1e-2,
            far=1e2,
            device=mesh.device,
        ).to(mesh.device)
        output = mesh.render(
            camera,
            shader=PrettyShader(antialias=self.aa, wireframe=self.wireframe, normal_type='flat'),
        ).rgb2srgb().item()
        output = avg_pool2d(output[None].permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).squeeze(0)
        dump_float32_image(self.output, output.clamp(0, 1))


if __name__ == '__main__':
    Tester(
        model=Path('data') / 'spot' / 'spot.obj',
        output=Path('temp.png'),
        cuda=0,
    ).run()
