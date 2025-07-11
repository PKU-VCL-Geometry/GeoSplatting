from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch

from rfstudio.graphics import Cameras, RGBImages, TriangleMesh
from rfstudio.graphics.shaders import PureShader
from rfstudio.utils.context import create_random_seed_context
from rfstudio.utils.typing import Indexable


@dataclass
class ShapeNetDataparser:

    alpha_color: Literal['white', 'black'] = 'black'
    """
    alpha color of background
    """

    sampling_heuristic: Literal['hemisphere', 'sphere'] = 'sphere'

    num_train_views: int = 100

    num_val_views: int = 20

    num_test_views: int = 20

    view_sampling_seed: int = 1

    resolution: Tuple[int, int] = (800, 800)

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBImages], Any]:

        with create_random_seed_context(self.view_sampling_seed):
            if split == 'val':
                cameras = Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 1, 0),
                    radius=3.,
                    pitch_degree=45,
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
                    radius=3.,
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

        mesh = TriangleMesh.from_file(path / 'models' / 'model_normalized.obj').to(device).normalize()

        assert mesh.face_normals is not None

        images = mesh.render(cameras, shader=PureShader()).clamp(0, 1).rgb2srgb()

        return cameras, images, mesh

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'models' / 'model_normalized.obj',
            path / 'models' / 'model_normalized.mtl',
        ]
        return all([p.exists() for p in paths])
