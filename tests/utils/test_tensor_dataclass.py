from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Size, TensorDataclass


@dataclass
class Cameras(TensorDataclass):

    c2w: Tensor = Float[..., 3, 4]

    fx: Tensor = Float[..., 1]

    width: Tensor = Float[1]

    def some_method(self) -> None:
        '''some method'''
        pass

@dataclass
class RaySamples(TensorDataclass):

    num_rays_per_batch: int = Size.Dynamic

    num_channels: int = Size.Dynamic

    positions: Tensor = Float[..., num_rays_per_batch, 3]

    colors: Tensor = Float.Trainable[..., num_rays_per_batch, num_channels]

    densities: Optional[Tensor] = Float[..., num_rays_per_batch, 1]

@dataclass
class CameraGroup(TensorDataclass):

    cameras1: Cameras = Cameras[...]

    cameras2: Optional[Cameras] = Cameras[3]

    samples: RaySamples = RaySamples.Trainable[1, 2, 5]

@dataclass
class WeightedRaySamples(RaySamples):

    weights: Tensor = Float[..., RaySamples.num_rays_per_batch, 1]


def test_basic_functions():

    cameras = Cameras(
        c2w=torch.zeros(1, 2, 3, 4),
        fx=torch.zeros(1, 2, 1),
        width=torch.zeros(1)
    )
    assert cameras.shape == (1, 2)
    assert cameras.size() == (1, 2)
    assert cameras.size(0) == 1
    assert cameras.dim() == 2
    assert cameras.c2w.shape == (1, 2, 3, 4)

    cameras = Cameras.zeros((1, 2))
    assert cameras.c2w.shape == (1, 2, 3, 4)
    assert cameras.view(-1).shape == (2,)
    assert cameras.view(-1).width.shape == (1,)
    assert cameras.view(-1).c2w.shape == (2, 3, 4)
    cameras = cameras[:, 0]
    assert cameras.c2w.shape == (1, 3, 4)

    assert Cameras.cat([
        Cameras.ones((3, 2)),
        Cameras.ones((3, 3)),
        Cameras.ones((3, 7)),
    ], dim=1).shape == (3, 12)

def test_extra_features():

    samples = RaySamples.zeros((1, 2), num_rays_per_batch=5, num_channels=4)
    assert samples.num_rays_per_batch == 5 and samples.num_channels == 4
    assert samples.positions.shape == (1, 2, 5, 3)
    assert samples.colors.shape == (1, 2, 5, 4)
    assert samples[0, :, None].colors.shape == (2, 1, 5, 4)
    assert RaySamples.cat([
        samples,
        samples,
        samples
    ], dim=0).shape == (3, 2)
    assert samples.densities is None
    assert samples.annotate(
        densities=torch.ones(samples.shape + (samples.num_rays_per_batch, 1))
    ).densities.shape == (1, 2, 5, 1)

    error = None
    try:
        samples.densities = None
    except RuntimeError as e:
        error = e
    finally:
        pass
    assert error is not None and error.args == ("Attribute assignment is not supported",)

def test_nested():

    group = CameraGroup.zeros((2, 3), num_rays_per_batch=3, num_channels=2)
    assert group.cameras2 is None
    assert group.cameras1.c2w.shape == (2, 3, 3, 4)
    assert group.samples.colors.shape == (1, 2, 5, 3, 2)

def test_inherit():
    samples = WeightedRaySamples.zeros((1, 2), num_rays_per_batch=5, num_channels=4)
    assert samples.num_rays_per_batch == 5 and samples.num_channels == 4
    assert samples.positions.shape == (1, 2, 5, 3)
    assert samples.colors.shape == (1, 2, 5, 4)
    assert samples.weights.shape == (1, 2, 5, 1)

def test_pickle():
    samples = RaySamples.zeros((1, 2), num_rays_per_batch=5, num_channels=4)
    samples.replace_(
        positions=torch.rand_like(samples.positions),
        colors=torch.rand_like(samples.colors),
    )
    samples.serialize('temp.pkl')
    new_samples = RaySamples.deserialize('temp.pkl')
    assert new_samples.num_rays_per_batch == samples.num_rays_per_batch
    assert new_samples.num_channels == samples.num_channels
    assert new_samples.positions.allclose(samples.positions)
    assert new_samples.colors.allclose(samples.colors)
    assert new_samples.densities is None

if __name__ == '__main__':
    test_basic_functions()
    test_extra_features()
    test_nested()
    test_inherit()
    test_pickle()
