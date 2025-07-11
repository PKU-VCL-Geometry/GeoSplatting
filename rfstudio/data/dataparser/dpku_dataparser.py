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

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch

from rfstudio.graphics import Cameras, RGBImages, SfMPoints
from rfstudio.utils.filesystem import get_last_modified_time
from rfstudio.utils.process import run_command
from rfstudio.utils.typing import Indexable

from .colmap_dataparser import ColmapDataparser


@dataclass
class DPKUDataparser(ColmapDataparser):

    max_image_size: int = 1280

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBImages], SfMPoints]:
        dense_path = path / 'dense'
        if ColmapDataparser.recognize(dense_path):
            sparse_time = get_last_modified_time(path / "sparse" / "0" / "cameras.bin")
            dense_time = get_last_modified_time(dense_path / "sparse" / "0" / "cameras.bin")
            if sparse_time <= dense_time:
                return super().parse(dense_path, split=split, device=device)
        if dense_path.exists():
            shutil.rmtree(dense_path)
        dense_path.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cmd = [
                'colmap',
                'image_undistorter',
                f'--image_path {path / "images"}',
                f'--input_path {path / "sparse" / "0"}',
                f'--output_path {tmpdir}',
                f'--max_image_size {self.max_image_size}'
            ]
            run_command(" ".join(cmd), verbose=False)
            (dense_path / 'sparse').mkdir(exist_ok=True, parents=True)
            (dense_path / 'images').mkdir(exist_ok=True, parents=True)
            shutil.move(tmpdir / 'sparse', dense_path / 'sparse' / '0')
            for p in (tmpdir / 'images').glob("**/*.jpg"):
                shutil.move(p, dense_path / 'images' / p.name)
        return super().parse(dense_path, split=split, device=device)

    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'sparse' / '0' / 'cameras.bin',
            path / 'sparse' / '0' / 'images.bin',
            path / 'sparse' / '0' / 'points3D.bin',
            path / 'database.db',
        ]
        return all([p.exists() for p in paths])
