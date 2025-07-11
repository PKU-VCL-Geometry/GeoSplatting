from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, TensorDataclass

from .math import safe_normalize


@dataclass
class SphericalGaussians(TensorDataclass):

    axis: Tensor = Float[..., 3]
    sharpness: Tensor = Float[..., 1]
    amplitude: Tensor = Float[..., 3]

    @staticmethod
    def from_brdf_lobe(
        *,
        normals: Float32[Tensor, "... 3"],
        wo: Float32[Tensor, "... 3"],
        roughness: Float32[Tensor, "... 1"],
    ) -> SphericalGaussians:
        inv_roughness_pow4 = 2. / roughness ** 4 # [..., 1]
        NoV = (normals * wo).sum(dim=-1, keepdim=True).clamp_min(1e-4) # [..., 1]
        sharpness = inv_roughness_pow4 / (4 * NoV + 1e-6)
        amplitude = (inv_roughness_pow4 / torch.pi).expand_as(normals)
        axis = safe_normalize(2 * NoV * normals - wo)
        return SphericalGaussians(
            axis=axis,
            sharpness=sharpness,
            amplitude=amplitude,
        )

    def __matmul__(self, rhs: SphericalGaussians) -> SphericalGaussians:
        assert (self.sharpness < rhs.sharpness).float().mean().item() > 0.5
        ratio = self.sharpness / rhs.sharpness
        dot = (self.axis * rhs.axis).sum(dim=-1, keepdim=True)
        tmp = (ratio * ratio + 1. + 2. * ratio * dot).sqrt()
        tmp = torch.min(tmp, ratio + 1.)
        diff = rhs.sharpness * (tmp - ratio - 1.)
        return SphericalGaussians(
            axis=safe_normalize((ratio * self.axis + rhs.axis) / tmp),
            sharpness=rhs.sharpness * tmp,
            amplitude=self.amplitude * rhs.amplitude * diff.exp()
        )

    def integral(self, normals: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 3"]:
        cosine = (self.axis * normals).sum(-1, keepdim=True)
        sharpness = self.sharpness + 1e-6
        inv_sharpness = 1. / sharpness
        t = torch.divide(
            sharpness.sqrt() * (1.6988 + 10.8438 * inv_sharpness),
            1. + 6.2201 * inv_sharpness + 10.2415 * inv_sharpness * inv_sharpness,
        )

        inv_a = torch.exp(-t)
        mask = (cosine >= 0).float()
        inv_b = torch.exp(-t * torch.clamp(cosine, min=0.))
        s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
        b = torch.exp(t * torch.clamp(cosine, max=0.))
        s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
        s = mask * s1 + (1. - mask) * s2

        A_b = 2. * torch.pi / sharpness * (torch.exp(-sharpness) - torch.exp(-2. * sharpness))
        A_u = 2. * torch.pi / sharpness * (1. - torch.exp(-sharpness))
        return (A_b * (1. - s) + A_u * s) * self.amplitude

    def cosine_integral(self, normals: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 3"]:
        cosine_sg = SphericalGaussians(
            axis=normals,
            sharpness=torch.empty_like(normals[..., :1]).fill_(0.0315),
            amplitude=torch.empty_like(normals).fill_(32.708),
        )
        return ((cosine_sg @ self).integral(normals) - 31.7003 * self.integral(normals)).sum(-2).clamp_min(0)
