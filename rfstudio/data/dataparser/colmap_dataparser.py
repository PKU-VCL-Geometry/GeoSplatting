# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from rfstudio.graphics import Cameras, RGBImages, SfMPoints
from rfstudio.graphics.math import get_rotation_from_relative_vectors
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_image_batch_lazy


def skip_next_bytes(fid: BinaryIO, num_bytes: int) -> None:
    new_pos = fid.seek(num_bytes, 1)
    assert new_pos != -1


def read_next_bytes(
    fid: BinaryIO,
    num_bytes: int,
    format_char_sequence: Iterable[Literal['c', 'e', 'f', 'd', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q']],
    endian_character: Literal['@', '=', '<', '>', '!'] = "<"
) -> Tuple[Any, ...]:
    """
    Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file: Path) -> Dict[int, Tuple[float, float, float, float, int, int]]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    results = {}
    assert path_to_model_file.exists()
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            assert model_id in [0, 1], (
                "Only support SIMPLE_PINHOLE (model_id = 0) and PINHOLE (model_id = 1) as camera models, "
                f"but model_id = {model_id} received"
            )
            width = camera_properties[2]
            height = camera_properties[3]
            if model_id == 0:
                f, cx, cy = read_next_bytes(fid, num_bytes=24, format_char_sequence="ddd")
                fx, fy = f, f
            else:
                fx, fy, cx, cy = read_next_bytes(fid, num_bytes=32, format_char_sequence="dddd")
            results[camera_id] = (fx, fy, cx, cy, width, height)
        return results


def read_images_binary(path_to_model_file: Path) -> List[Tuple[int, str, Tensor, int]]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        images = {}
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            camera_id = binary_image_properties[8]
            # assert image_id >= 0 and image_id < num_reg_images and images[image_id] is None
            assert image_id not in images
            c2w = np.zeros((3, 4))
            R = qvec2rotmat(binary_image_properties[1:5])
            T = np.array(binary_image_properties[5:8])
            c2w[:, :3] = R.T
            c2w[:, 3] = R.T @ -T
            c2w[:, 1:3] *= -1
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":     # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            skip_next_bytes(fid, num_bytes=24 * num_points2D)
            images[image_id] = (image_name, torch.from_numpy(c2w).float(), camera_id)
    assert len(images) == num_reg_images
    g = ((image_id, Path(image_name).name, c2w, camera_id) for image_id, (image_name, c2w, camera_id) in images.items())
    return list(sorted(g, key=lambda x: x[1]))


def read_points3D_binary(path_to_model_file: Path) -> Tuple[Tensor, Tensor, Tensor]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    xyz = []
    rgb = []
    ind = []
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for i in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz.append(np.asarray(binary_point_line_properties[1:4]))
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            rgb.append(np.asarray(binary_point_line_properties[4:7]) / 255.)
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.asarray(tuple(set(map(int, track_elems[0::2]))))[:, None] # [S, 1]
            ind.append(np.pad(image_ids, ((0, 0), (1, 0)), 'constant', constant_values=i))
    return (
        torch.from_numpy(np.stack(xyz)).float(),
        torch.from_numpy(np.stack(rgb)).float(),
        torch.from_numpy(np.concatenate(ind, axis=0))
    )


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ])


@dataclass
class ColmapDataparser(BaseDataparser[Cameras, RGBImages, SfMPoints]):

    image_path: str = 'images'

    downsample: Optional[int] = None

    train_split_ratio: int = 7

    val_split_ratio: int = 1

    test_split_ratio: int = 2

    auto_orient: bool = True

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBImages], SfMPoints]:

        recon_path = path / 'sparse' / '0'
        camera_lst = read_cameras_binary(recon_path / 'cameras.bin')
        meta = read_images_binary(recon_path / 'images.bin')
        assert len(meta) >= 10
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
        indices = [i for i in range(len(meta)) if split_range[0] <= (i % split_ratio_sum) < split_range[1]]
        c2w = torch.stack([c2w for _, _, c2w, _ in meta])  # [N, 3, 4]

        positions, colors, image_ids = read_points3D_binary(recon_path / 'points3D.bin') # [S, 3], [S, 3], [K, 2]
        image_ids = image_ids.to(device)
        image_indices = image_ids.clone()
        map_image_id_to_image_idx = { image_id: i for i, (image_id, _, _, _) in enumerate(meta) }
        for image_id, image_idx in map_image_id_to_image_idx.items():
            mask = (image_ids[:, 1] == image_id)
            image_indices[mask, 1] = image_idx
        visibilities = torch.zeros((positions.shape[0], c2w.shape[0]), dtype=torch.bool, device=device)
        visibilities[image_indices[:, 0], image_indices[:, 1]] = True
        offset = positions.mean(0)
        c2w[:, :, 3] -= offset
        positions -= offset
        if self.auto_orient:
            up = c2w[:, :, 1].mean(0)
            up = up / torch.linalg.norm(up)
            rotation = get_rotation_from_relative_vectors(up, torch.tensor([0, 0, 1]).to(up))
            c2w[:, :, :3] = rotation[None, :, :] @ c2w[:, :, :3]
            c2w[:, :, 3:] = rotation[None, :, :] @ c2w[:, :, 3:]
            positions = (rotation[None, :, :] @ positions[:, :, None]).squeeze(-1)
        rescale = 0.9 / torch.quantile(positions.view(-1).abs(), 0.9)
        c2w[:, :, 3] *= rescale                                                          # scale to bbox [-1, 1]^3
        positions *= rescale

        N = len(indices)
        cameras = Cameras(
            c2w=c2w[indices, :, :],
            fx=torch.empty(N),
            fy=torch.empty(N),
            cx=torch.empty(N),
            cy=torch.empty(N),
            width=torch.empty(N, dtype=torch.long),
            height=torch.empty(N, dtype=torch.long),
            near=torch.ones(N) * 1e-3,
            far=torch.ones(N) * 1e3
        )
        max_size = 0
        for i in range(N):
            camera_id = meta[indices[i]][3]
            fx, fy, cx, cy, width, height = camera_lst[camera_id]
            max_size = max(max_size, width, height)
        if self.downsample is None:
            downsample = 1
            while max_size > 1600:
                max_size /= 2
                downsample *= 2
        else:
            downsample = self.downsample
        for i in range(N):
            camera_id = meta[indices[i]][3]
            fx, fy, cx, cy, width, height = camera_lst[camera_id]
            cameras.fx[i] = fx / downsample
            cameras.fy[i] = fy / downsample
            cameras.cx[i] = cx / downsample
            cameras.cy[i] = cy / downsample
            cameras.width[i] = round(width / downsample)
            cameras.height[i] = round(height / downsample)

        image_path = f'{self.image_path}_{downsample}' if downsample > 1 else self.image_path
        downsample_online = not (path / image_path).exists()
        if downsample_online:
            image_path = self.image_path
        image_filenames = [path / image_path / file_path for _, file_path, _, _ in meta]
        assert all([p.exists() for p in image_filenames])

        images = load_image_batch_lazy(
            [image_filenames[i] for i in indices],
            scale_factor=1 / downsample if downsample_online else None,
            device=device,
        )

        points = SfMPoints(
            positions=positions.to(device),
            colors=colors.to(device),
            visibilities=visibilities,
        ).seen_by(torch.tensor(indices))

        return cameras.to(device), images, points

    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'sparse' / '0' / 'cameras.bin',
            path / 'sparse' / '0' / 'images.bin',
            path / 'sparse' / '0' / 'points3D.bin',
        ]
        return all([p.exists() for p in paths])
