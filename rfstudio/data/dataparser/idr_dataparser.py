from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Literal, Tuple

import cv2
import numpy as np
import torch

from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.io import dump_float32_image, load_float32_image
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_image_batch_lazy, load_masked_image_batch_lazy


@dataclass
class IDRDataparser(BaseDataparser[Cameras, RGBImages, Any]):

    scale_factor: float = 0.4

    auto_orient: Literal['+y', '-y', '+z', 'none'] = '+y'

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBImages], Any]:

        image_filenames = []
        image_dirs = path / 'image'
        assert image_dirs.exists()
        total_num_images = len(list(image_dirs.glob("*.png")))
        for i in range(total_num_images):
            image_filenames.append(image_dirs / f'{i:06d}.png')
        IMAGE_HEIGHT, IMAGE_WIDTH = load_float32_image(image_filenames[0]).shape[:2]

        c2w = torch.empty((total_num_images, 3, 4)) # [N, 3, 4]
        fxfycxcy = torch.empty((total_num_images, 4)) # [N, 4]
        cam_file = np.load(path / 'cameras_large.npz')

        for i in range(total_num_images):
            P = torch.from_numpy(cam_file[f'world_mat_{i}'] @ cam_file[f'scale_mat_{i}']).float()
            ixt, inv_ext = _load_K_Rt_from_P(P[:3, :4])
            c2w[i, :3, :4] = inv_ext
            fxfycxcy[i] = ixt[[0, 4, 2, 5]] * self.scale_factor
        c2w[:, :, 1:3] *= -1 # from COLMAP to Blender

        N = c2w.shape[0]
        cameras = Cameras(
            c2w=c2w,
            fx=fxfycxcy[:, 0],
            fy=fxfycxcy[:, 1],
            cx=fxfycxcy[:, 2],
            cy=fxfycxcy[:, 3],
            width=torch.empty(N, dtype=torch.long).fill_(int(self.scale_factor * IMAGE_WIDTH)),
            height=torch.empty(N, dtype=torch.long).fill_(int(self.scale_factor * IMAGE_HEIGHT)),
            near=torch.empty(N).fill_(1e-2),
            far=torch.empty(N).fill_(1e2),
        ).transform_to_fit_sphere(radius=(3 ** 0.5), auto_orient=self.auto_orient)

        cameras = cameras.clone()

        images = load_image_batch_lazy(
            image_filenames,
            scale_factor=self.scale_factor,
            device=device,
        )

        return cameras.to(device), images, None

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'image' / '000000.png',
            path / 'cameras_large.npz',
        ]
        return all([p.exists() for p in paths])

    @staticmethod
    def dump(
        inputs: Cameras,
        gt_outputs: RGBImages,
        meta: Any,
        *,
        path: pathlib.Path,
        split: Literal['train', 'val', 'test'],
    ) -> None:

        if split != 'train':
            return

        assert inputs.ndim == 1
        assert path.exists()
        (path / 'image').mkdir(exist_ok=True)
        (path / 'mask').mkdir(exist_ok=True)

        c2w = torch.cat((inputs.c2w, torch.tensor([0, 0, 0, 1]).to(inputs.c2w).expand(inputs.shape[0], 1, 4)), dim=1)
        c2w[:, :, 1:3] *= -1 # from Blender to COLMAP
        K = inputs.intrinsic_matrix # [N, 3, 3]
        P = torch.eye(4).to(K).view(1, 4, 4).repeat(inputs.shape[0], 1, 1)
        P[:, :3, :3] = K @ c2w[:, :3, :3].transpose(-1, -2)
        P[:, :3, 3:] = P[:, :3, :3] @ -c2w[:, :3, 3:]
        cam_file = {}
        for idx, image in enumerate(gt_outputs[...].clamp(0, 1)):
            cam_file[f'world_mat_{idx}'] = P[idx].detach().cpu().numpy().astype(np.float64)
            cam_file[f'scale_mat_{idx}'] = np.eye(4)
            dump_float32_image(path / 'image' / f'{idx:06d}.png', image)
            dump_float32_image(path / 'mask' / f'{idx:03d}.png', torch.zeros_like(image))
        np.savez_compressed(path / 'cameras_large.npz', **cam_file)


def _load_K_Rt_from_P(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # This function is borrowed from IDR: https://github.com/lioryariv/idr

    K, R, t, *_ = cv2.decomposeProjectionMatrix(P.detach().cpu().numpy())

    K = K / K[2, 2]

    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return torch.from_numpy(K).to(P).flatten(), torch.from_numpy(pose).to(P)[:3, :4]


@dataclass
class MaskedIDRDataparser(BaseDataparser[Cameras, RGBAImages, Any]):

    scale_factor: float = 0.4

    auto_orient: Literal['+y', '-y', '+z', 'none'] = '+y'

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], Any]:

        image_filenames = []
        mask_filenames = []
        image_dirs = path / 'image'
        mask_dirs = path / 'mask'
        assert image_dirs.exists() and mask_dirs.exists()
        total_num_images = len(list(image_dirs.glob("*.png")))
        for i in range(total_num_images):
            image_filenames.append(image_dirs / f'{i:06d}.png')
            mask_filenames.append(mask_dirs / f'{i:03d}.png')
        IMAGE_HEIGHT, IMAGE_WIDTH = load_float32_image(image_filenames[0]).shape[:2]

        c2w = torch.empty((total_num_images, 3, 4)) # [N, 3, 4]
        fxfycxcy = torch.empty((total_num_images, 4)) # [N, 4]
        cam_file = np.load(path / 'cameras_large.npz')

        for i in range(total_num_images):
            P = torch.from_numpy(cam_file[f'world_mat_{i}'] @ cam_file[f'scale_mat_{i}']).float()
            ixt, inv_ext = _load_K_Rt_from_P(P[:3, :4])
            c2w[i, :3, :4] = inv_ext
            fxfycxcy[i] = ixt[[0, 4, 2, 5]] * self.scale_factor
        c2w[:, :, 1:3] *= -1 # from COLMAP to Blender

        N = c2w.shape[0]
        cameras = Cameras(
            c2w=c2w,
            fx=fxfycxcy[:, 0],
            fy=fxfycxcy[:, 1],
            cx=fxfycxcy[:, 2],
            cy=fxfycxcy[:, 3],
            width=torch.empty(N, dtype=torch.long).fill_(int(self.scale_factor * IMAGE_WIDTH)),
            height=torch.empty(N, dtype=torch.long).fill_(int(self.scale_factor * IMAGE_HEIGHT)),
            near=torch.empty(N).fill_(1e-2),
            far=torch.empty(N).fill_(1e2),
        ).transform_to_fit_sphere(radius=3, auto_orient=self.auto_orient)

        cameras = cameras.clone()

        images = load_masked_image_batch_lazy(
            image_filenames,
            scale_factor=self.scale_factor,
            device=device,
            masks=mask_filenames,
        )

        return cameras.to(device), images, None

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'image' / '000000.png',
            path / 'mask' / '000.png',
            path / 'cameras_large.npz',
        ]
        return all([p.exists() for p in paths])

    @staticmethod
    def dump(
        inputs: Cameras,
        gt_outputs: RGBAImages,
        meta: Any,
        *,
        path: pathlib.Path,
        split: Literal['train', 'val', 'test'],
    ) -> None:

        if split != 'train':
            return

        assert inputs.ndim == 1
        assert path.exists()
        (path / 'image').mkdir(exist_ok=True)
        (path / 'mask').mkdir(exist_ok=True)

        c2w = torch.cat((inputs.c2w, torch.tensor([0, 0, 0, 1]).to(inputs.c2w).expand(inputs.shape[0], 1, 4)), dim=1)
        c2w[:, :, 1:3] *= -1 # from Blender to COLMAP
        K = inputs.intrinsic_matrix # [N, 3, 3]
        P = torch.eye(4).to(K).view(1, 4, 4).repeat(inputs.shape[0], 1, 1)
        P[:, :3, :3] = K @ c2w[:, :3, :3].transpose(-1, -2)
        P[:, :3, 3:] = P[:, :3, :3] @ -c2w[:, :3, 3:]
        cam_file = {}
        for idx, image in enumerate(gt_outputs[...].clamp(0, 1)):
            cam_file[f'world_mat_{idx}'] = P[idx].detach().cpu().numpy().astype(np.float64)
            cam_file[f'scale_mat_{idx}'] = np.eye(4)
            dump_float32_image(path / 'image' / f'{idx:06d}.png', image[..., :3])
            dump_float32_image(path / 'mask' / f'{idx:03d}.png', image[..., 3:].repeat(1, 1, 3))
        np.savez_compressed(path / 'cameras_large.npz', **cam_file)
