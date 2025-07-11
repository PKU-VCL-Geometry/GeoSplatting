from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, Points, Texture2D, TriangleMesh
from rfstudio.io import dump_float32_image
from rfstudio.visualization import Visualizer


def test_render(
    mesh: TriangleMesh,
    output: Path,
    *,
    W: int = 1280,
    H: int = 720,
    antialias: bool = False,
) -> Dict:
    camera = Cameras.from_lookat(
        eye=(1.5, 1.5, 1.5),
        resolution=(W, H),
        near=0.5,
        far=4,
        device=mesh.device,
    ).to(mesh.device)
    images = torch.cat((
        mesh.render(camera).blend((0, 0, 0)).item(),
        mesh.render_depth(camera).visualize(max_bound=camera.far.item()).item(),
    ), dim=1)
    dump_float32_image(output, images)
    return { 'view': camera }


def test_sample(mesh: TriangleMesh, *, num_samples: int) -> Dict[str, Points]:
    return {
        'uniform': mesh.view(1, 1).uniformly_sample(
            num_samples,
            samples_per_face='uniform',
        ),
        'poisson': mesh.poisson_disk_sample(num_samples),
        'r2': mesh.uniformly_sample(
            num_samples,
            samples_per_face='uniform',
            samples_in_face='r2',
        )
    }


def test_subdivide(mesh: TriangleMesh) -> Dict[str, TriangleMesh]:
    subx1 = mesh.subdivide()
    subx2 = subx1.subdivide()
    return {
        'subdivide(x1)': subx1,
        'subdivide(x2)': subx2
    }


@dataclass
class Tester(Task):

    model: Optional[Path] = ...
    output: Path = ...
    texture: Optional[Path] = None
    viser: Visualizer = Visualizer(port=6789)
    aa: bool = False

    test_render: bool = True
    test_sample: bool = True
    test_subdivide: bool = True

    def run(self) -> None:
        if self.model is None:
            vertices = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).float() # [3, 3]
            mesh = TriangleMesh(
                vertices=vertices,
                indices=torch.tensor([[0, 1, 2]]).long(),
                uvs=torch.zeros((1, 3, 2)),
                textures=torch.zeros((1, 1, 3)),
            ).to(self.device)
            N = 5000
        else:
            texture = None if self.texture is None else Texture2D.from_image_file(self.texture)
            mesh = TriangleMesh.from_file(self.model, kd_texture=texture).to(self.device).normalize()
            N = 50000
        with self.viser.customize() as handle:
            if self.test_render:
                for key, value in test_render(mesh, self.output, antialias=self.aa).items():
                    handle[key].show(value)
            if self.test_sample:
                for key, value in test_sample(mesh, num_samples=N).items():
                    handle[key].show(value)
            if self.test_subdivide:
                for key, value in test_subdivide(mesh).items():
                    handle[key].show(value).configurate(shade='wireframe')
            handle['mesh'].show(mesh)


if __name__ == '__main__':
    Tester(
        model=Path('data') / 'spot' / 'spot_triangulated.obj',
        output=Path('temp.png'),
        texture=Path('data') / 'spot' / 'spot_texture.png',
        cuda=0,
    ).run()
