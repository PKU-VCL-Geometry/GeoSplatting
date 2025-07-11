from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch

from rfstudio.graphics import Cameras, RGBAImages
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_masked_image_batch_lazy


@dataclass
class Syn4RelightDataparser(BaseDataparser[
    Cameras,
    RGBAImages,
    Tuple[Indexable[RGBAImages], Indexable[RGBAImages], Tuple[Indexable[RGBAImages], ...], Tuple[Path, ...]],
]):

    scale_factor: Optional[float] = None
    """
    scale factor for resizing image
    """

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[
        Indexable[Cameras],
        Indexable[RGBAImages],
        Tuple[Indexable[RGBAImages], Indexable[RGBAImages], Tuple[Indexable[RGBAImages], ...], Tuple[Path, ...]],
    ]:

        IMAGE_WIDTH = 800
        IMAGE_HEIGHT = 800

        if self.scale_factor is not None:
            IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
            IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)

        split = 'train' if split == 'val' else split

        with open(path / f"transforms_{split}.json", 'r') as f:
            meta = json.loads(f.read())

        poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)

        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

        c2w = torch.from_numpy(poses[:, :3, :])            # camera to world transform
        c2w = torch.stack((-c2w[:, 1, :], c2w[:, 2, :], -c2w[:, 0, :]), dim=-2)
        c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

        N = c2w.shape[0]
        cameras = Cameras(
            c2w=c2w,
            fx=torch.ones(N) * focal_length,
            fy=torch.ones(N) * focal_length,
            cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
            cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
            width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
            height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
            near=torch.ones(N) * (4 / 3),
            far=torch.ones(N) * 4,
        ).to(device)

        if split == 'test':
            images = load_masked_image_batch_lazy(
                [path / (frame['file_path'] + "_rgba.png") for frame in meta['frames']],
                device=device,
                scale_factor=self.scale_factor,
            )
            with open(path / "transforms_test.json", 'r') as f:
                test_meta = json.loads(f.read())
            albedo = load_masked_image_batch_lazy(
                [path / (frame['file_path'] + "_albedo.png") for frame in test_meta['frames']],
                device=device,
                scale_factor=self.scale_factor,
            )
            roughness = load_masked_image_batch_lazy(
                [path / (frame['file_path'] + "_rough.png") for frame in test_meta['frames']],
                device=device,
                scale_factor=self.scale_factor,
            )
            relight1 = load_masked_image_batch_lazy(
                [
                    path / 'test_rli' / ('envmap6_' + frame['file_path'].rsplit('/', 1)[1] + ".png")
                    for frame in test_meta['frames']
                ],
                device=device,
                scale_factor=self.scale_factor,
            )
            relight2 = load_masked_image_batch_lazy(
                [
                    path / 'test_rli' / ('envmap12_' + frame['file_path'].rsplit('/', 1)[1] + ".png")
                    for frame in test_meta['frames']
                ],
                device=device,
                scale_factor=self.scale_factor,
            )
            envmap1 = path.parent / 'envmap6.exr'
            envmap2 = path.parent / 'envmap12.exr'
            meta = (albedo, roughness, (relight1, relight2), (envmap1, envmap2))
        else:
            images = load_masked_image_batch_lazy(
                [path / (frame['file_path'] + "_rgb.exr") for frame in meta['frames']],
                masks=[path / (frame['file_path'] + "_mask.png") for frame in meta['frames']],
                device=device,
                scale_factor=self.scale_factor,
                pbra=True,
            )
            meta = None

        return cameras, images, meta

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'test',
            path / 'train',
            path / 'transforms_test.json',
            path / 'transforms_train.json',
            path.parent / 'envmap3.exr',
            path.parent / 'envmap6.exr',
            path.parent / 'envmap12.exr',
        ]
        return all([p.exists() for p in paths])
