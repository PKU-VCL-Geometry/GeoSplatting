from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Tuple

import torch

from rfstudio.graphics import Cameras, RGBAImages, Texture2D, TriangleMesh
from rfstudio.graphics.shaders import LambertianShader
from rfstudio.utils.context import create_random_seed_context
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser


class _KnownModel(NamedTuple):
    mesh_name: str
    texture_name: Optional[str]
    val_pitch_degree: float
    view_radius: float


_KNOWN_MODELS = {
    'spot': _KnownModel(
        mesh_name='spot_triangulated.obj',
        texture_name='spot_texture.png',
        val_pitch_degree=45.,
        view_radius=3.,
    ),
    'cube': _KnownModel(
        mesh_name='cube.obj',
        texture_name=None,
        val_pitch_degree=45.,
        view_radius=3.,
    ),
    'damicornis': _KnownModel(
        mesh_name='usnm_93379-150k.obj',
        texture_name='usnm_93379-100k-2048-diffuse.jpg',
        val_pitch_degree=15.,
        view_radius=3.,
    ),
}


@dataclass
class MeshViewSynthesisDataparser(BaseDataparser[Cameras, RGBAImages, TriangleMesh]):

    resolution: Tuple[int, int] = (512, 512)

    sampling_heuristic: Literal['hemisphere', 'sphere'] = 'sphere'

    num_train_views: int = 192

    num_val_views: int = 64

    num_test_views: int = 128

    antialias: bool = True

    view_sampling_seed: int = 123

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], TriangleMesh]:
        model = _KNOWN_MODELS[path.name]
        if model.texture_name is None:
            texture = None
        else:
            texture = Texture2D.from_image_file(path / model.texture_name)
        mesh = TriangleMesh.from_file(path / model.mesh_name, kd_texture=texture).normalize().to(device)

        with create_random_seed_context(self.view_sampling_seed):
            if split == 'val':
                cameras = Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 1, 0),
                    radius=model.view_radius,
                    pitch_degree=model.val_pitch_degree,
                    num_samples=self.num_val_views,
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=45,
                    device=device
                )
            else:
                if self.sampling_heuristic == 'hemisphere':
                    sampling_heuristic = Cameras.from_hemisphere
                elif self.sampling_heuristic == 'sphere':
                    sampling_heuristic = Cameras.from_sphere
                else:
                    raise ValueError(self.sampling_heuristic)
                cameras = sampling_heuristic(
                    center=(0, 0, 0),
                    up=(0, 1, 0),
                    radius=model.view_radius,
                    num_samples=(self.num_train_views + self.num_test_views),
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=45,
                    device=device
                )

                if split == 'train':
                    cameras = cameras[:self.num_train_views]
                elif split == 'test':
                    cameras = cameras[-self.num_test_views:]
                else:
                    raise ValueError('Argument `split` must be any of train, val or test.')

        images = mesh.render(cameras, shader=LambertianShader(antialias=self.antialias))
        return cameras, images, mesh

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        if path.exists() and path.name in _KNOWN_MODELS:
            model = _KNOWN_MODELS[path.name]
            if not (path / model.mesh_name).exists():
                return False
            if (model.texture_name is not None) and not (path / model.texture_name).exists():
                return False
            return True
        return False
