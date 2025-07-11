from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import torch
from jaxtyping import Float32, Shaped
from matplotlib import colormaps
from torch import Tensor


def _float2ratio(
    value: Float32[Tensor, "..."],
    *,
    eps: float = 1e-5,
    auto_scaling: bool = True,
) -> np.ndarray:
    np_array = value.detach().cpu().numpy()
    if not auto_scaling:
        return np_array
    return (np_array - np_array.min()) / np.clip(np_array.max() - np_array.min(), a_min=eps, a_max=None)


def _color2tensor(value: Union[np.ndarray, tuple], *, reference: Shaped[Tensor, "... 1"]) -> Float32[Tensor, "... 3"]:
    np_value = np.asarray(value)[..., :3]
    return torch.from_numpy(np_value).to(reference.device).float().view(*reference.shape[:-1], 3)


@dataclass
class BaseColorMap:

    def __call__(self, value: Float32[Tensor, "... 1"]) -> Float32[Tensor, "... 3"]:
        ratios = _float2ratio(value.flatten(), auto_scaling=True)
        return _color2tensor(colormaps[self.style](ratios), reference=value).view(*value.shape[:-1], 3)

    def from_scaled(self, value: Float32[Tensor, "... 1"]) -> Float32[Tensor, "... 3"]:
        assert value.min().item() >= 0 and value.max().item() <= 1
        ratios = _float2ratio(value.flatten(), auto_scaling=False)
        return _color2tensor(colormaps[self.style](ratios), reference=value).view(*value.shape[:-1], 3)

    def discretize(self, N: int, *, device: Optional[torch.device] = None) -> Float32[Tensor, "N 3"]:
        queries = (torch.arange(N, device=device) + 0.5) / N
        return self.from_scaled(queries)

@dataclass
class BinaryColorMap(BaseColorMap):

    style: Literal[
        'PiYG',
        'PRGn',
        'BrBG',
        'RdGy',
        'RdBu',
        'RdYlBu',
        'RdYlGn',
        'Spectral',
        'coolwarm',
        'bwr',
        'seismic',
    ] = 'RdBu'
    '''
    https://matplotlib.org/stable/users/explain/colors/colormaps.html#diverging
    '''


@dataclass
class IntensityColorMap(BaseColorMap):

    style: Literal[
        'binary',
        'gist_gray',
        'gray',
        'bone',
        'pink',
        'hot',
        'afmhot',
        'gist_heat',
    ] = 'hot'
    '''
    https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential2
    '''


@dataclass
class UniformColorMap(BaseColorMap):

    style: Literal['viridis', 'plasma', 'inferno', 'magma', 'cividis'] = 'viridis'
    '''
    https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential
    '''


@dataclass
class CyclicColorMap(BaseColorMap):

    style: Literal['twilight', 'twilight_shifted', 'hsv'] = 'twilight_shifted'
    '''
    https://matplotlib.org/stable/users/explain/colors/colormaps.html#cyclic
    '''


@dataclass
class RainbowColorMap(BaseColorMap):

    style: Literal[
        'CMRmap',
        'gist_earth',
        'terrain',
        'gnuplot',
        'gnuplot2',
        'cubehelix',
        'brg',
        'gist_rainbow',
        'rainbow',
        'jet',
        'turbo',
        'nipy_spectral',
        'gist_ncar',
    ] = 'rainbow'
    '''
    https://matplotlib.org/stable/users/explain/colors/colormaps.html#miscellaneous
    '''
