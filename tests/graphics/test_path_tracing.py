from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, Texture2D, TextureCubeMap, TriangleMesh
from rfstudio.graphics.shaders import DiffusePBRShader, NormalShader, PathTraceShader
from rfstudio.io import dump_float32_image
from rfstudio.visualization import TabularFigures


@dataclass
class Tester(Task):

    model: Optional[Path] = ...
    output: Path = ...
    envmap: Path = ...
    aa: bool = False
    z_up: bool = False

    roughness: Optional[float] = None
    metallic: Optional[float] = None

    def run(self) -> None:
        if self.model is None:
            mesh = TriangleMesh.create_sphere().to(self.device)
            mesh.annotate_(
                uvs=mesh.vertices.new_zeros(*mesh.indices.shape, 2),
                kd=Texture2D.from_constants((1.0, 1.0, 1.0), device=self.device),
                ks=Texture2D.from_constants((0.0, self.roughness, self.metallic), device=self.device),
            )
        else:
            mesh = TriangleMesh.from_file(self.model).normalize().to(self.device)
            if mesh.uvs is None:
                mesh.replace_(uvs=mesh.vertices.new_zeros(*mesh.indices.shape, 2))
            if mesh.kd is None:
                mesh.replace_(kd=Texture2D.from_constants((0.8, 0.7, 0.2), device=self.device))
            if mesh.ks is None:
                mesh.replace_(ks=Texture2D.from_constants((0, 0.25, 0), device=self.device))
            if self.roughness is not None:
                mesh.ks.data[..., 1] = self.roughness
            if self.metallic is not None:
                mesh.ks.data[..., 2] = self.metallic
        envmap = TextureCubeMap.from_image_file(self.envmap, device=self.device)
        if self.z_up:
            envmap.z_up_to_y_up_()
        camera = Cameras.from_lookat(
            eye=(-1.2, -1.2, 1.2) if self.z_up else (-1.2, 1.2, -1.2),
            resolution=(400, 400),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            hfov_degree=60,
            near=1e-2,
            far=1e2,
            device=mesh.device,
        ).to(mesh.device)
        figures = TabularFigures(1, 3, device=mesh.device)
        figures[0, 0].load(mesh.render(camera, shader=PathTraceShader(envmap=envmap)).rgb2srgb(), info='PT[full]')
        figures[0, 1].load(mesh.render(camera, shader=DiffusePBRShader(envmap=envmap)).rgb2srgb(), info='PT[diffuse]')
        figures[0, 2].load(mesh.render(camera, shader=NormalShader()).visualize((1, 1, 1)), info='Normal')
        img = figures.draw(text_bg_color=(0.4, 0.4, 0.4, 0.4), text_fg_color=(1, 1, 1, 1)).clamp(0, 1)
        dump_float32_image(self.output, img)


if __name__ == '__main__':
    Tester(
        model=Path('data') / 'spot' / 'spot.obj',
        envmap=Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr',
        output=Path('temp.png'),
        cuda=0,
    ).run()
