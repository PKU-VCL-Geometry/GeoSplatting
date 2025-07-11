from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Type, TypeVar

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from ._points import Points

T = TypeVar('T', bound=TensorDataclass)

@dataclass
class Rays(TensorDataclass):

    origins: torch.Tensor = Float[..., 3]

    directions: torch.Tensor = Float[..., 3]

    pixel_indices: Optional[torch.Tensor] = Long[..., 3]             # [..., CamIdx/H/W]

    near: torch.Tensor = Float[..., 1]

    far: torch.Tensor = Float[..., 1]

    def get_samples(
        self,
        t: Float32[torch.Tensor, "... S+1"],
        *,
        spacing: Literal['uniform', 'linear_disparity'] = 'uniform',
    ) -> RaySamples:
        if spacing == 'uniform':
            bins = (self.near + (self.far - self.near) * t).unsqueeze(-1) # [..., S+1, 1]
        elif spacing == 'linear_disparity':
            bins = 1 / (1 / self.near + (1 / self.far - 1 / self.near) * t).unsqueeze(-1)
        else:
            raise ValueError(spacing)
        return RaySamples(
            origins=self.origins,
            directions=self.directions,
            bins=bins[..., :-1, :],
            max_bin=bins[..., -1, :],
        )


@dataclass
class RaySamples(TensorDataclass):

    num_samples: int = Size.Dynamic

    origins: torch.Tensor = Float[..., 3]

    directions: torch.Tensor = Float[..., 3]

    bins: torch.Tensor = Float[..., num_samples, 1]

    max_bin: torch.Tensor = Float[..., 1]

    colors: Optional[torch.Tensor] = Float[..., num_samples, 3]

    densities: Optional[torch.Tensor] = Float[..., num_samples, 1]

    alphas: Optional[torch.Tensor] = Float[..., num_samples, 1]

    weights: Optional[torch.Tensor] = Float[..., num_samples, 1]

    @property
    def positions(self) -> Float32[Tensor, "*bs S 3"]:
        bins = torch.cat((self.bins, self.max_bin.unsqueeze(-1)), dim=-2) # [..., S+1, 1]
        return self.origins[..., None, :] + (bins[..., :-1, :] + bins[..., 1:, :]) / 2 * self.directions[..., None, :]

    @property
    def distances(self) -> Float32[Tensor, "*bs S 1"]:
        return torch.cat((
            self.bins[..., 1:, :] - self.bins[..., :-1, :],
            self.max_bin.unsqueeze(-1) - self.bins[..., -1:, :],
        ), dim=-2) # [..., S, 1]

    def get_weighted(self) -> RaySamples:
        assert self.densities is not None or self.alphas is not None, "Densities/alphas required"
        assert self.densities is None or self.alphas is None, "Densities/alphas cannot be given both"
        assert self.weights is None, "Weights have already been computed"
        if self.densities is not None:
            distances = (self.distances * self.directions.norm(dim=-1, keepdim=True).unsqueeze(-1)) # [..., S, 1]
            delta_density = distances * self.densities                 # [..., S, 1]
            alphas = 1 - (-delta_density).exp()                        # [..., S, 1]
            transmittance = delta_density.cumsum(dim=-2)               # [..., S, 1]
            transmittance = torch.cat((
                torch.zeros_like(transmittance[..., :1, :]),
                delta_density.cumsum(dim=-2)[..., :-1, :],
            ), dim=-2)
            transmittance = (-transmittance).exp()                     # [..., S, 1]
        else:
            alphas = self.alphas
            transmittance = torch.cat((
                torch.zeros_like(alphas[..., :1, :]),
                (1.0 - alphas + 1e-7).cumprod(dim=-2)[..., :-1, :],
            ), dim=-2)
        weights = torch.nan_to_num(alphas * transmittance)
        return self.annotate(weights=weights)

    def view_as(self, target_type: Type[T]) -> T:
        if target_type is Points:
            return Points(
                positions=self.positions,
                colors=self.colors,
            )
        raise NotImplementedError
