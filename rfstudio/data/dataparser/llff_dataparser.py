from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import numpy as np
import torch

from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.graphics.math import get_rotation_from_relative_vectors
from rfstudio.io import load_float32_image
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_image_batch_lazy, load_masked_image_batch_lazy


@dataclass
class LLFFDataparser(BaseDataparser[Cameras, RGBImages, Any]):

    train_split_ratio: int = 8

    val_split_ratio: int = 1

    test_split_ratio: int = 1

    auto_orient: bool = False

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBImages], Any]:

        poses_bounds = torch.from_numpy(np.load(path / 'poses_bounds.npy')).float() # [N, 17]
        poses = poses_bounds[:, :15].view(-1, 3, 5) # [N, 3, 5]
        hwf = poses[:, :, 4] # [N, 3]
        c2w = poses[:, :, :4].clone() # [N, 3, 4]
        c2w[:, :, 0] = poses[:, :, 1]
        c2w[:, :, 1] = -poses[:, :, 0]
        bounds = poses_bounds[:, 15:] # [N, 2]
        split_ratio_sum = self.train_split_ratio + self.val_split_ratio + self.test_split_ratio
        if split == 'train':
            split_range = (0, self.train_split_ratio)
        elif split == 'test':
            split_range = (self.train_split_ratio, self.train_split_ratio + self.test_split_ratio)
        elif split == 'val':
            split_range = (self.train_split_ratio + self.test_split_ratio, split_ratio_sum)
        else:
            raise ValueError(
                "Invalid value for argument 'split':"
                f"'train', 'test', 'val' expected, but {repr(split)} received"
            )
        indices = [i for i in range(poses.shape[0]) if split_range[0] <= (i % split_ratio_sum) < split_range[1]]
        image_filenames = list((path / 'images').glob("*.JPG"))
        image_filenames.sort(key=lambda p: p.name)
        assert all([p.exists() for p in image_filenames])
        height, width = load_float32_image(image_filenames[0]).shape[:2]

        offset = c2w[:, :, 3].mean(0) # [3]
        c2w[:, :, 3] -= offset
        if self.auto_orient:
            up = c2w[:, :, 1].mean(0)
            up = up / torch.linalg.norm(up)
            rotation = get_rotation_from_relative_vectors(up, torch.tensor([0, 0, 1]).to(up))
            c2w[:, :, :3] = rotation[None, :, :] @ c2w[:, :, :3]
            c2w[:, :, 3:] = rotation[None, :, :] @ c2w[:, :, 3:]
        rescale = 1.1 / c2w[:, :, 3].max()
        c2w[:, :, 3] *= rescale # scale to bbox [-1, 1]^3

        N = len(indices)
        cameras = Cameras(
            c2w=c2w[indices, :, :],
            fx=(hwf[:, 2] / hwf[:, 1] * width)[indices],
            fy=(hwf[:, 2] / hwf[:, 0] * height)[indices],
            cx=torch.empty(N).fill_(width / 2),
            cy=torch.empty(N).fill_(height / 2),
            width=torch.empty(N, dtype=torch.long).fill_(width),
            height=torch.empty(N, dtype=torch.long).fill_(height),
            near=bounds[indices, 0] * rescale,
            far=bounds[indices, 1] * rescale,
        )

        images = load_image_batch_lazy(
            [image_filenames[i] for i in indices],
            device=device,
        )

        return cameras.to(device), images, None

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'images',
            path / 'poses_bounds.npy',
        ]
        return all([p.exists() for p in paths])


@dataclass
class MaskedLLFFDataparser(BaseDataparser[Cameras, RGBAImages, Any]):

    train_split_ratio: int = 8

    val_split_ratio: int = 1

    test_split_ratio: int = 1

    auto_orient: bool = False

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], Any]:

        poses_bounds = torch.from_numpy(np.load(path / 'poses_bounds.npy')).float() # [N, 17]
        poses = poses_bounds[:, :15].view(-1, 3, 5) # [N, 3, 5]
        hwf = poses[:, :, 4] # [N, 3]
        c2w = poses[:, :, :4].clone() # [N, 3, 4]
        c2w[:, :, 0] = poses[:, :, 1]
        c2w[:, :, 1] = -poses[:, :, 0]
        bounds = poses_bounds[:, 15:] # [N, 2]
        split_ratio_sum = self.train_split_ratio + self.val_split_ratio + self.test_split_ratio
        if split == 'train':
            split_range = (0, self.train_split_ratio)
        elif split == 'test':
            split_range = (self.train_split_ratio, self.train_split_ratio + self.test_split_ratio)
        elif split == 'val':
            split_range = (self.train_split_ratio + self.test_split_ratio, split_ratio_sum)
        else:
            raise ValueError(
                "Invalid value for argument 'split':"
                f"'train', 'test', 'val' expected, but {repr(split)} received"
            )
        indices = [i for i in range(poses.shape[0]) if split_range[0] <= (i % split_ratio_sum) < split_range[1]]
        image_filenames = list((path / 'images').glob("*.JPG"))
        image_filenames.sort(key=lambda p: p.name)
        assert all([p.exists() for p in image_filenames])
        height, width = load_float32_image(image_filenames[0]).shape[:2]

        offset = c2w[:, :, 3].mean(0) # [3]
        c2w[:, :, 3] -= offset
        if self.auto_orient:
            up = c2w[:, :, 1].mean(0)
            up = up / torch.linalg.norm(up)
            rotation = get_rotation_from_relative_vectors(up, torch.tensor([0, 0, 1]).to(up))
            c2w[:, :, :3] = rotation[None, :, :] @ c2w[:, :, :3]
            c2w[:, :, 3:] = rotation[None, :, :] @ c2w[:, :, 3:]
        rescale = 1.1 / c2w[:, :, 3].max()
        c2w[:, :, 3] *= rescale # scale to bbox [-1, 1]^3

        N = len(indices)
        cameras = Cameras(
            c2w=c2w[indices, :, :],
            fx=(hwf[:, 2] / hwf[:, 1] * width)[indices],
            fy=(hwf[:, 2] / hwf[:, 0] * height)[indices],
            cx=torch.empty(N).fill_(width / 2),
            cy=torch.empty(N).fill_(height / 2),
            width=torch.empty(N, dtype=torch.long).fill_(width),
            height=torch.empty(N, dtype=torch.long).fill_(height),
            near=bounds[indices, 0] * rescale,
            far=bounds[indices, 1] * rescale,
        )

        masks = { p.stem: p for p in (path / 'masks').iterdir() }
        images = load_masked_image_batch_lazy(
            [image_filenames[i] for i in indices],
            masks=[masks[image_filenames[i].stem] for i in indices],
            device=device,
        )

        return cameras.to(device), images, None

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'images',
            path / 'masks',
            path / 'poses_bounds.npy',
        ]
        return all([p.exists() for p in paths])
