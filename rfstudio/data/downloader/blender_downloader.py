from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

from .base_downloader import BaseDownloader


@dataclass
class BlenderDownloader(BaseDownloader):

    scene: Literal['all', 'chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'] = 'all'

    @property
    def hf_id(self) -> str:
        return "jkulhanek/nerfbaselines-data"

    @property
    def name(self) -> str:
        return 'blender'

    @property
    def files(self) -> Dict[str, Path]:
        return {
            'chair': Path('blender') / 'chair.zip',
            'drums': Path('blender') / 'drums.zip',
            'ficus': Path('blender') / 'ficus.zip',
            'hotdog': Path('blender') / 'hotdog.zip',
            'lego': Path('blender') / 'lego.zip',
            'materials': Path('blender') / 'materials.zip',
            'mic': Path('blender') / 'mic.zip',
            'ship': Path('blender') / 'ship.zip',
        }

    def process(self, source: Path, target: Path, mutable: bool) -> None:
        with zipfile.ZipFile(source, "r") as zip_ref:
            zip_ref.extractall(str(target))
