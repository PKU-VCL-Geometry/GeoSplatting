from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32, Int32
from torch import Tensor

from rfstudio.graphics.math import (
    get_polar_from_rect_2d,
    get_projection,
    get_radian_distance,
)


@dataclass
class BaseSpatialSelector:

    def filter(self, positions: Float32[Tensor, "N 3"]) -> Bool[Tensor, "N 1"]:
        raise NotImplementedError


@dataclass
class BaseSequentialSelector:

    def filter(self, indices: Int32[Tensor, "N 1"]) -> Bool[Tensor, "N 1"]:
        raise NotImplementedError


@dataclass
class FanSelector(BaseSpatialSelector):

    fan_index: int

    num_fans: int

    num_fans_per_cover: float = 1.0

    plane: Literal['xy', 'yz', 'xz', 'pca'] = 'pca'

    @property
    def radian_per_cover(self) -> float:
        return self.radian_per_fan * self.num_fans_per_cover

    @property
    def radian_per_fan(self) -> float:
        return 2 * np.pi / self.num_fans

    def filter(self, positions: Float32[Tensor, "N 3"]) -> Bool[Tensor, "N 1"]:
        coords_2d, _ = get_projection(positions, plane=self.plane)     # [N, 2]
        fan_center = self.fan_index * self.radian_per_fan
        _, theta = get_polar_from_rect_2d(coords_2d)                   # [N], [N]
        return (get_radian_distance(fan_center, theta) * 2 < self.radian_per_cover).unsqueeze(-1)


@dataclass
class SliceSelector(BaseSequentialSelector):

    slice_index: int

    num_slices: int

    cover_factor: float = 1.0

    def filter(self, indices: Int32[Tensor, "N 1"]) -> Bool[Tensor, "N 1"]:
        assert self.num_slices > 1
        padding = 0.5 * self.cover_factor / self.num_slices
        assert padding < 0.5
        max_index = indices.max() + 1
        normalized = (indices.float() + 0.5) / max_index   # [N, 1]
        slice_center = padding + (1 - 2 * padding) * self.slice_index / (self.num_slices - 1)
        return (normalized - slice_center).abs() < padding
