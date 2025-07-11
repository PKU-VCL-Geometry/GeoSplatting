from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import torch

from rfstudio.graphics import Cameras, RGBAImages
from rfstudio.io import dump_float32_image
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_masked_image_batch_lazy


@dataclass
class RFMaskedRealDataparser(BaseDataparser[Cameras, RGBAImages, Any]):

    train_split_ratio: int = 7

    val_split_ratio: int = 1

    test_split_ratio: int = 2

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
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], Any]:

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

        image_filenames = list((path / 'images').glob("*.png"))
        indices = [i for i in range(len(image_filenames)) if split_range[0] <= (i % split_ratio_sum) < split_range[1]]
        image_filenames = [path / 'images' / f'{i:04d}.png' for i in indices]

        camera_data = torch.load(path / 'cameras.pkl', map_location='cpu')
        cameras = Cameras(
            c2w=camera_data['c2w'],
            fx=camera_data['fx'],
            fy=camera_data['fy'],
            cx=camera_data['cx'],
            cy=camera_data['cy'],
            width=camera_data['width'],
            height=camera_data['height'],
            near=camera_data['near'],
            far=camera_data['far'],
        )[indices].contiguous().to(device)

        images = load_masked_image_batch_lazy(
            image_filenames,
            device=device,
            scale_factor=self.scale_factor,
        )

        return cameras, images, None

    @staticmethod
    def dump(
        inputs: Cameras,
        gt_outputs: RGBAImages,
        meta: Any,
        *,
        path: pathlib.Path,
        split: Literal['all'],
    ) -> None:

        assert inputs.ndim == 1
        assert split == 'all'
        assert len(inputs) == len(gt_outputs)
        inputs = inputs.detach().cpu()

        assert path.exists()
        (path / 'images').mkdir(exist_ok=True)

        camera_data = {
            'c2w': inputs.c2w,
            'fx': inputs.fx,
            'fy': inputs.fy,
            'cx': inputs.cx,
            'cy': inputs.cy,
            'width': inputs.width,
            'height': inputs.height,
            'near': inputs.near,
            'far': inputs.far,
        }
        torch.save(camera_data, path / 'cameras.pkl')
        for idx, image in enumerate(gt_outputs):
            dump_float32_image(path / 'images' / f'{idx:04d}.png', image)

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'images' / '0000.png',
            path / 'cameras.pkl',
        ]
        return all([p.exists() for p in paths])
