from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..module import Module


@dataclass
class CurveMapping(Module):

    feature_dim: Optional[int] = None

    num_control_points: int = 10

    point_distribution: Literal['uniform', 'log', 'exp'] = 'uniform'

    def __setup__(self) -> None:
        assert self.num_control_points > 0
        if self.feature_dim is not None:
            self.control_points = nn.Parameter(torch.randn(self.num_control_points, self.feature_dim) * 0.1)
        else:
            self.control_points = nn.UninitializedParameter()

    def __call__(self, inputs: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs in_dim"]:
        assert (inputs.min().item() >= 0 and inputs.max().item() <= 1), "Inputs must be normalized to [0, 1)"
        shape = inputs.shape
        inputs = inputs.detach().view(-1, shape[-1]) * (1 - 1e-6)
        if nn.parameter.is_lazy(self.control_points):
            self.control_points.materialize((self.num_control_points, shape[-1]))
            self.feature_dim = shape[-1]
            self.control_points.data.copy_(torch.randn(self.num_control_points, self.feature_dim) * 0.1)
        curve_points = nn.functional.pad(
            self.control_points.exp().cumsum(0),
            (0, 0, 1, 0),
            mode='constant',
            value=0,
        )
        curve_points = curve_points / curve_points[-1:, :]             # [CP+1, FD]

        if self.point_distribution == 'uniform':
            pass
        elif self.point_distribution == 'log':
            inputs = (inputs + 1).log2()
        elif self.point_distribution == 'exp':
            inputs = 2 ** inputs - 1
        else:
            raise NotImplementedError

        x = inputs * self.num_control_points
        indices = x.floor().long()                                     # [*BS, FD]
        weights = x - indices                                          # [*BS, FD]
        return torch.add(
            curve_points.gather(dim=0, index=indices) * (1 - weights), # [*BS, FD]
            curve_points.gather(dim=0, index=indices + 1) * weights    # [*BS, FD]
        ).view(shape)

    def get_bins(self) -> List[Any]:
        assert not nn.parameter.is_lazy(self.control_points)
        cp = self.control_points.detach().exp().cumsum(0)
        return (cp / cp[-1:, :]).tolist()
