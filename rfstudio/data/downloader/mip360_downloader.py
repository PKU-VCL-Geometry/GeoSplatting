from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

from .base_downloader import BaseDownloader


@dataclass
class Mip360Downloader(BaseDownloader):

    scene: Literal['all', 'bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump'] = 'all'

    @property
    def hf_id(self) -> str:
        return "illusive-chase/rfdata"

    @property
    def name(self) -> str:
        return 'mip360'

    @property
    def files(self) -> Dict[str, Path]:
        return {
            'bicycle': Path('mip360') / 'bicycle.zip',
            'bonsai': Path('mip360') / 'bonsai.zip',
            'counter': Path('mip360') / 'counter.zip',
            'garden': Path('mip360') / 'garden.zip',
            'kitchen': Path('mip360') / 'kitchen.zip',
            'room': Path('mip360') / 'room.zip',
            'stump': Path('mip360') / 'stump.zip',
        }

    def process(self, source: Path, target: Path, mutable: bool) -> None:
        with zipfile.ZipFile(source, "r") as zip_ref:
            zip_ref.extractall(str(target))
