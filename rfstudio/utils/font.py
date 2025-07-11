from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float32
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

_ASSETS_DIR: Path = files('rfstudio') / 'assets' / 'font'

@lru_cache(maxsize=64)
def _cached_font(file: Path, font_size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(file, font_size)

@lru_cache(maxsize=64)
def _baked(
    text: str,
    *,
    file: Path,
    line_height: int,
) -> Float32[Tensor, "H W 1"]:
    font = _cached_font(file, line_height * 0.9)
    length = font.getlength(text, mode='L')
    image = Image.new(mode="RGBA", size=(int(length + 0.99), line_height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text(
        (0, line_height * 0.75),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        anchor='ls',
    )
    return torch.from_numpy(np.array(image)[..., 3:]) / 255.

@dataclass
class Font:

    load: Path = ...

    @staticmethod
    def from_name(name: Literal['LinLibertine', 'LinLibertine_B', 'TimesNewRoman', 'TimesNewRoman_B']) -> Font:
        if name == 'LinLibertine':
            filename = 'LinLibertine_R.ttf'
        elif name == 'LinLibertine_B':
            filename = 'LinLibertine_RB.ttf'
        elif name == 'TimesNewRoman':
            filename = 'TimesNewRoman_R.ttf'
        elif name == 'TimesNewRoman_B':
            filename = 'TimesNewRoman_RB.ttf'
        else:
            raise ValueError(name)
        return Font(load=_ASSETS_DIR / filename)

    def write(self, text: str, *, line_height: int) -> Float32[Tensor, "H W 1"]:
        assert text.isascii()
        return _baked(text, file=self.load, line_height=line_height)
