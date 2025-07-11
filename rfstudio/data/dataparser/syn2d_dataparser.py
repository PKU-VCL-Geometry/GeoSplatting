
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Tuple

import torch

from rfstudio.graphics._2d import Cameras2D, CircleShape2D, RGBA2DImages
from rfstudio.utils.context import create_random_seed_context
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser


@dataclass
class Synthetic2DDataparser(BaseDataparser[Cameras2D, RGBA2DImages, Any]):

    num_circles: int = 3

    num_train_views: int = 8192

    num_val_views: int = 8192

    num_test_views: int = 200

    data_creation_seed: int = 123

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras2D], Indexable[RGBA2DImages], Any]:

        assert path.name == 'circle'

        with create_random_seed_context(self.data_creation_seed):
            shape = CircleShape2D.random(self.num_circles, device=device)
            num_samples = self.num_train_views + self.num_val_views + self.num_test_views
            cameras = Cameras2D.from_orbit(
                center=(0, 0),
                radius=1.,
                num_samples=num_samples,
                width=800,
                near=1e-3,
                far=2.,
                hfov_degree=60,
                device=device,
            )[torch.randperm(num_samples, device=device)].contiguous()

            if split == 'train':
                cameras = cameras[:self.num_train_views]
            elif split == 'test':
                cameras = cameras[-self.num_test_views:]
            elif split == 'val':
                cameras = cameras[self.num_train_views:-self.num_test_views]
            else:
                raise ValueError('Argument `split` must be any of train, val or test.')

        images = shape.render(cameras)
        return cameras, images, shape

    @staticmethod
    def recognize(path: Path) -> bool:
        return str(path).startswith('synthetic:/2d/')
