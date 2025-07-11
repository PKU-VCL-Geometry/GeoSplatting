from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from jaxtyping import Float
from torch import Tensor

from rfstudio.graphics import Points, TriangleMesh

from ._chamfer import ChamferDistance


def _convert(x: Union[Points, TriangleMesh], *, target_num_points: Optional[int]) -> Points:
    if isinstance(x, TriangleMesh):
        num_samples = max(x.num_faces * 10, 100000) if target_num_points is None else target_num_points
        return x.uniformly_sample(num_samples, samples_per_face='uniform').to(x.device)
    if isinstance(x, Points):
        assert x.ndim == 1
        return x
    raise TypeError(x.__class__)

@dataclass
class ChamferDistanceMetric:

    target_num_points: Optional[int] = None

    def __call__(self, a: Union[Points, TriangleMesh], b: Union[Points, TriangleMesh]) -> Float[Tensor, "1"]:
        points_a = _convert(a, target_num_points=self.target_num_points) # [S]
        points_b = _convert(b, target_num_points=self.target_num_points) # [S]
        dist_ab, dist_ba = ChamferDistance()(points_a.positions, points_b.positions)
        return dist_ab.mean() + dist_ba.mean()

@dataclass
class ChamferDistanceFscoreMetric:

    threshold: float = 0.001
    target_num_points: Optional[int] = None

    def __call__(
        self,
        a: Union[Points, TriangleMesh],
        b: Union[Points, TriangleMesh],
    ) -> Tuple[Float[Tensor, "1"], Float[Tensor, "1"]]:
        points_a = _convert(a, target_num_points=self.target_num_points) # [S]
        points_b = _convert(b, target_num_points=self.target_num_points) # [S]
        dist_ab, dist_ba = ChamferDistance()(points_a.positions, points_b.positions)
        precision_ab = (dist_ab < self.threshold).float().mean()
        precision_ba = (dist_ba < self.threshold).float().mean()
        fscore = 2 * precision_ab * precision_ba / (precision_ab + precision_ba).clamp(min=1e-10)
        return dist_ab.mean() + dist_ba.mean(), fscore
