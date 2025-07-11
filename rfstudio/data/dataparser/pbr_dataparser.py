from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Tuple

import torch

from rfstudio.graphics import Cameras, RGBAImages, Texture2D, TextureCubeMap, TriangleMesh
from rfstudio.graphics.shaders import PBRShader
from rfstudio.utils.context import create_random_seed_context
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser


class _KnownModel(NamedTuple):
    mesh_name: str
    envmap: Path
    val_pitch_degree: float
    view_radius: float
    ks: Optional[Tuple[float, float, float]]


_KNOWN_MODELS = {
    'spot': _KnownModel(
        mesh_name='spot.obj',
        envmap=Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr',
        val_pitch_degree=45.,
        view_radius=3.,
        ks=None,
    ),
    'damicornis': _KnownModel(
        mesh_name='usnm_93379-150k.obj',
        envmap=Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr',
        val_pitch_degree=15.,
        view_radius=3.,
        ks=(0.0, 0.25, 0.0),
    ),
}


@dataclass
class MeshPBRDataparser(BaseDataparser[Cameras, RGBAImages, TriangleMesh]):

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
        assert model.envmap.exists()
        mesh = TriangleMesh.from_file(path / model.mesh_name).normalize().to(device)
        assert mesh.kd is not None
        if mesh.ks is None:
            assert model.ks is not None
            mesh.replace_(ks=Texture2D.from_constants(model.ks, device=device))
        else:
            assert model.ks is None
        envmap = TextureCubeMap.from_image_file(model.envmap, device=device).as_splitsum()

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

        images = mesh.render(
            cameras,
            shader=PBRShader(
                antialias=self.antialias,
                envmap=envmap,
            ),
        ).clamp(0, 1).rgb2srgb()
        return cameras, images, mesh

    @staticmethod
    def recognize(path: Path) -> bool:
        if path.exists() and path.name in _KNOWN_MODELS:
            model = _KNOWN_MODELS[path.name]
            return (path / model.mesh_name).exists()
        return False
