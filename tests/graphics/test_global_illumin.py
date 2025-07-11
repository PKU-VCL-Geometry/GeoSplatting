from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, Texture2D, TextureLatLng, TriangleMesh
from rfstudio.graphics.shaders import MCShader, PBRShader
from rfstudio.io import dump_float32_image


@dataclass
class Tester(Task):

    output: Path = ...
    envmap: Path = ...
    aa: bool = False

    def run(self) -> None:
        meshes = []
        colors_i = [(1.0, 1.0, 1.0), (0.8, 0.2, 0.2), (0.1, 0.6, 0.1), (0.2, 0.3, 0.6)]
        metallic_j = [0.0, 0.3, 0.9]
        roughness_j = [0.4, 0.2, 0.1]
        pos_i = [-1.5, -0.5, 0.5, 1.5]
        pos_j = [-1, 0, 1]
        for i in range(4):
            for j in range(3):
                mesh = TriangleMesh.create_sphere(radius=0.45).translate(pos_i[i], 0, pos_j[j]).to(self.device)
                mesh.annotate_(
                    uvs=mesh.vertices.new_zeros(*mesh.indices.shape, 2),
                    kd=Texture2D.from_constants(colors_i[i], device=self.device),
                    ks=Texture2D.from_constants((0.0, roughness_j[j], metallic_j[j]), device=self.device),
                )
                meshes.append(mesh)
        scene = TriangleMesh.merge(*meshes)
        envmap = TextureLatLng.from_image_file(self.envmap, device=self.device)
        camera = Cameras.from_lookat(
            eye=(0.5, 4, -2),
            resolution=(512, 512),
            up=(0, 1, 0),
            hfov_degree=60,
            near=1e-2,
            far=1e2,
            device=scene.device,
        ).to(scene.device)
        output = torch.cat((
            scene.render(camera, shader=PBRShader(
                antialias=self.aa,
                envmap=envmap.as_cubemap(resolution=512).as_splitsum()),
            ).rgb2srgb().blend((1, 1, 1)).item(),
            scene.render(camera, shader=MCShader(
                antialias=self.aa,
                envmap=envmap,
                num_samples_per_ray=16,
            )).rgb2srgb().blend((1, 1, 1)).item(),
        ), dim=1)
        dump_float32_image(self.output, output.clamp(0, 1))


if __name__ == '__main__':
    Tester(
        envmap=Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr',
        output=Path('temp.png'),
        cuda=0,
    ).run()
