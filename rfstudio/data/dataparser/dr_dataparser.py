from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Tuple

import torch

from rfstudio.graphics import Cameras, DepthImages, RGBAImages, TriangleMesh
from rfstudio.graphics.shaders import DepthShader
from rfstudio.utils.context import create_random_seed_context
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser


class _KnownModel(NamedTuple):
    mesh_name: str
    val_pitch_degree: float
    view_radius: float
    z_up: bool


_KNOWN_MODELS = {
    'spot': _KnownModel(
        mesh_name='spot.obj',
        val_pitch_degree=45.,
        view_radius=3.,
        z_up=False,
    ),
    'inputmodels': _KnownModel(
        mesh_name='block.obj',
        val_pitch_degree=45.,
        view_radius=3.,
        z_up=False,
    ),
    'damicornis': _KnownModel(
        mesh_name='usnm_93379-150k.obj',
        val_pitch_degree=15.,
        view_radius=3.,
        z_up=False,
    ),
    'lego': _KnownModel(
        mesh_name='lego.ply',
        val_pitch_degree=45.,
        view_radius=3.,
        z_up=True,
    ),
}


@dataclass
class MeshDRDataparser(BaseDataparser[Cameras, DepthImages, TriangleMesh]):

    resolution: Tuple[int, int] = (800, 800)

    sampling_heuristic: Literal['hemisphere', 'sphere'] = 'sphere'

    num_train_views: int = 100

    num_val_views: int = 100

    num_test_views: int = 200

    antialias: bool = True

    view_sampling_seed: int = 123

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], TriangleMesh]:
        model = _KNOWN_MODELS[path.name]
        mesh = TriangleMesh.from_file(path / model.mesh_name).normalize().to(device)

        with create_random_seed_context(self.view_sampling_seed):
            if split == 'val':
                cameras = Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.z_up else (0, 1, 0),
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
                    up=(0, 0, 1) if model.z_up else (0, 1, 0),
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

        images = mesh.render(
            cameras,
            shader=DepthShader(
                antialias=self.antialias,
                culling=False,
            ),
        )
        return cameras, images, mesh

    @staticmethod
    def recognize(path: Path) -> bool:
        if path.exists() and path.name in _KNOWN_MODELS:
            model = _KNOWN_MODELS[path.name]
            return (path / model.mesh_name).exists()
        return False
