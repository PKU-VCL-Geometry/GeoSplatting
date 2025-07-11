from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch

from rfstudio.graphics import Cameras, RGBAImages, TriangleMesh
from rfstudio.io import dump_float32_image
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_masked_image_batch_lazy


@dataclass
class StanfordORBDataparser(BaseDataparser[Cameras, RGBAImages, TriangleMesh]):

    scale_factor: Optional[float] = 1024 / 2048
    """
    scale factor for resizing image
    """

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], TriangleMesh]:

        IMAGE_WIDTH = 2048
        IMAGE_HEIGHT = 2048

        if split == 'val':
            split = 'train'

        if self.scale_factor is not None:
            IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
            IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)

        with open(path / f"transforms_{split}.json", 'r') as f:
            meta = json.loads(f.read())
        image_filenames = [path / (frame['file_path'] + ".png") for frame in meta['frames']]
        mask_filenames = [
            path / (frame['file_path'].replace(split, split + '_mask') + ".png")
            for frame in meta['frames']
        ]
        poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)

        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

        c2w = torch.from_numpy(poses[:, :3, :])            # camera to world transform
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
            far=torch.ones(N) * 4
        ).to(device)

        images = load_masked_image_batch_lazy(
            image_filenames,
            device=device,
            scale_factor=self.scale_factor,
            masks=mask_filenames,
        )

        mesh = TriangleMesh.from_file(
            path.parent.parent / 'ground_truth' / path.name / 'mesh_blender' / 'mesh.obj'
        ).to(device)
        mesh.vertices.mul_(2 / 3)

        return cameras, images, mesh

    @staticmethod
    def dump(
        inputs: Cameras,
        gt_outputs: RGBAImages,
        meta: TriangleMesh,
        *,
        path: pathlib.Path,
        split: Literal['train', 'test', 'val'],
    ) -> None:

        assert inputs.ndim == 1
        focal_length = inputs.fx[0].item()
        image_width = inputs.width[0].item()
        assert (inputs.fx == focal_length).all() and (inputs.fy == focal_length).all()
        assert (inputs.width == image_width).all() and (inputs.height == image_width).all()

        assert path.exists()
        (path / split).mkdir(exist_ok=True)

        transforms = {
            'frames': [],
            'camera_angle_x': np.arctan2(image_width, 2 * focal_length) * 2,
        }
        c2w = torch.cat((inputs.c2w, torch.tensor([0, 0, 0, 1]).to(inputs.c2w).expand(inputs.shape[0], 1, 4)), dim=1)
        c2w[:, :3, 3] *= 1.5
        for idx, pose, image in zip(range(len(c2w)), c2w, gt_outputs[...].clamp(0, 1), strict=True):
            frame = {
                'transform_matrix': pose.detach().cpu().numpy().tolist(),
                'file_path': str(pathlib.Path('.') / split / f'r_{idx}'),
            }
            transforms['frames'].append(frame)
            dump_float32_image(path / split / f'r_{idx}.png', image)
        with open(path / f"transforms_{split}.json", 'w') as f:
            json.dump(transforms, f, indent=4)

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'test',
            path / 'test_mask',
            path / 'train',
            path / 'train_mask',
            path / 'transforms_test.json',
            path / 'transforms_train.json',
            path / 'transforms_novel.json',
            path.parent,
        ]
        return (
            all([p.exists() for p in paths]) and
            path.parent.name == 'blender_LDR' and
            (path.parent.parent / 'ground_truth' / path.name).exists()
        )
