from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from rfstudio.graphics import Points, TriangleMesh
from rfstudio.graphics.math import get_random_quaternion, quat2rot, sh_deg2dim, sh_dim2deg
from rfstudio.utils.decorator import chains
from rfstudio.utils.tensor_dataclass import Float, Size, TensorDataclass


@dataclass
class Splats(TensorDataclass):

    sh_dim: int = Size.Dynamic

    means: torch.Tensor = Float.Trainable[..., 3]
    scales: torch.Tensor = Float.Trainable[..., 3]
    quats: torch.Tensor = Float.Trainable[..., 4]
    colors: torch.Tensor = Float.Trainable[..., 3]
    shs: torch.Tensor = Float.Trainable[..., sh_dim, 3]
    opacities: torch.Tensor = Float.Trainable[..., 1]

    last_width: Optional[torch.Tensor] = Float[1]
    last_height: Optional[torch.Tensor] = Float[1]
    xys_grad_norm: Optional[torch.Tensor] = Float[...]
    vis_counts: Optional[torch.Tensor] = Float[...]

    @property
    def sh_degree(self) -> Literal[0, 1, 2, 3, 4]:
        return sh_dim2deg(self.sh_dim + 1)

    @classmethod
    def from_points(
        cls,
        points: Points,
        *,
        sh_degree: int,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Splats:
        assert points.colors is not None
        distances = points.k_nearest(k=3)[0]   # [N, K]
        size = distances.shape[0]

        return cls(
            means=points.positions.clone().to(device),
            scales=distances.mean(dim=-1, keepdim=True).log().repeat(1, 3),
            quats=get_random_quaternion(size, device=device),
            opacities=torch.logit(0.1 * torch.ones((size, 1), device=device)),
            colors=points.colors.clone().to(device),
            shs=torch.zeros((size, sh_deg2dim(sh_degree) - 1, 3), device=device)
        ).requires_grad_(requires_grad)

    @classmethod
    def random(
        cls,
        size: int,
        *,
        sh_degree: int,
        random_scale: float,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
    ) -> Splats:
        points = Points.rand((size, ), device=device).translate(-0.5).scale(2 * random_scale)
        distances = points.k_nearest(k=3)[0]   # [N, K]

        return cls(
            means=points.positions,
            scales=distances.mean(dim=-1, keepdim=True).log().repeat(1, 3),
            quats=get_random_quaternion(size, device=device),
            opacities=torch.logit(0.1 * torch.ones((size, 1), device=device)),
            colors=(torch.ones((size, 3), device=device) * 0.5),
            shs=torch.zeros((size, sh_deg2dim(sh_degree) - 1, 3), device=device)
        ).requires_grad_(requires_grad)

    @torch.no_grad()
    def reset_opacities(self, *, reset_value: float) -> None:
        # Reset value is set to be twice of the cull_alpha_thresh
        self.opacities.data.clamp_(max=torch.logit(torch.tensor(reset_value)).item())

    @torch.no_grad()
    def split(self, num_splits: int, scale_factor: float = 1 / 1.6) -> Splats:
        expaneded_shape = (num_splits, *self.shape)
        # sample new means
        randn = torch.randn((*expaneded_shape, 3), device=self.device)      # [S, ..., 3]
        scaled_offsets = self.scales.exp() * randn                          # [S, ..., 3]
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)          # [..., 4]
        rots = quat2rot(quats.view(-1, 4)).view(1, *self.shape, 3, 3) # [1, ..., 3, 3]
        rotated_offsets = (rots @ scaled_offsets[..., None])[..., 0]        # [S, ..., 3]
        new_means = rotated_offsets + self.means                            # [S, ..., 3]

        # sample new colors
        new_colors = self.colors.expand(*expaneded_shape, 3).contiguous()        # [S, ..., 3]
        new_shs = self.shs.expand(*expaneded_shape, self.sh_dim, 3).contiguous() # [S, ..., SH_DIM, 3]

        # sample new opacities
        new_opacities = self.opacities.expand(*expaneded_shape, 1).contiguous() # [S, ..., 1]

        # sample new scales
        new_scales = (self.scales.exp() * scale_factor).log()
        new_scales = new_scales.expand(*expaneded_shape, 3).contiguous() # [S, ..., 3]

        # sample new quats
        new_quats = self.quats.expand(*expaneded_shape, 4).contiguous() # [S, ..., 4]

        return Splats(
            means=new_means,
            scales=new_scales,
            quats=new_quats,
            colors=new_colors,
            shs=new_shs,
            opacities=new_opacities
        )

    @torch.no_grad()
    def densify_and_cull(
        self,
        *,
        densify_grad_thresh: float,
        densify_size_thresh: float,
        num_splits: int,
        cull_alpha_thresh: float,
        cull_scale_thresh: Optional[float]
    ) -> Int32[Tensor, "N ndim"]:
        assert self.xys_grad_norm is not None and self.vis_counts is not None
        scale_max = self.scales.exp().max(dim=-1).values               # [...]
        avg_grad_norm = 0.5 * max(
            self.last_width.item(),
            self.last_height.item(),
        ) * (self.xys_grad_norm / self.vis_counts)                     # [...]
        high_grads = (avg_grad_norm > densify_grad_thresh)             # [...]
        splits = (scale_max > densify_size_thresh)                     # [...]
        dups = high_grads & (scale_max <= densify_size_thresh)         # [...]
        splits = high_grads & splits                                   # [...]

        culls = (self.opacities.sigmoid()[..., 0] < cull_alpha_thresh) # [...]
        if cull_scale_thresh is not None:
            toobigs = (scale_max > cull_scale_thresh)                  # [...]
            culls = culls | toobigs
        selected = ~(culls | splits)                                   # [...]

        new_gaussians = Splats.cat([
            self[selected].clear_extras_(),
            self[splits].split(num_splits).view(-1),
            self[dups].clear_extras_(),
        ], dim=0)
        self.swap_(new_gaussians.requires_grad_(self.requires_grad))
        indices = selected.nonzero() # [N, ndim]

        return torch.cat([
            indices,
            -indices.new_ones(np.prod(self.shape) - indices.shape[0], indices.shape[1]),
        ], dim=0) # [N', ndim]

    @torch.no_grad()
    def cull(
        self,
        *,
        cull_alpha_thresh: float,
        cull_scale_thresh: Optional[float],
    ) -> Int32[Tensor, "N ndim"]:
        culls = (self.opacities.sigmoid()[..., 0] < cull_alpha_thresh) # [...]
        if cull_scale_thresh is not None:
            scale_max = self.scales.exp().max(dim=-1).values           # [...]
            toobigs = (scale_max > cull_scale_thresh)                  # [...]
            culls = culls | toobigs
        selected = ~culls
        self.swap_(self[selected].requires_grad_(self.requires_grad))
        return selected.nonzero()

    def clear_extras_(self) -> Splats:
        self.replace_(
            xys_grad_norm=None,
            vis_counts=None,
            last_width=None,
            last_height=None,
        )
        return self

    @chains
    def as_module(self, *, field_name: str) -> Any:

        def parameters(self) -> Any:
            return [getattr(self, field_name)]

        return parameters

    def get_cov3d_half(self) -> Float32[Tensor, "*bs 3 3"]:
        R = quat2rot(self.quats / self.quats.norm(dim=-1, keepdim=True)) # [..., 3, 3]
        S = self.scales.exp()                  # [..., 3]
        M = R * S[..., None, :]                # [..., 3, 3]
        return M

    def get_cov3d_inv_half(self) -> Float32[Tensor, "*bs 3 3"]:
        R = quat2rot(self.quats / self.quats.norm(dim=-1, keepdim=True)) # [..., 3, 3]
        S_inv = (-self.scales).exp()           # [..., 3]
        M_inv = R * S_inv[..., None, :]        # R @ (1/S) -> [..., 3, 3]
        # Cov3d.-1 = R.-T @ S.-T @ S.-1 @ R.-1 = R @ (1/S) @ (1/S).T @ R.T = M_inv @ M_inv.T
        return M_inv

    @torch.no_grad()
    def get_bounding_box(self, threshold: float) -> Float32[Tensor, "*bs 2 3"]:
        M_inv = self.get_cov3d_inv_half()      # [..., 3, 3]

        # solve equation:
        #     scaling_coeff * exp(-0.5 * x.T @ cov3d_inv @ x) < threshold
        #  -> scaling_coeff_log - 0.5 * x.T @ cov3d_inv @ x < threshold_log
        #  -> x.T @ cov3d_inv @ x > 2 * (scaling_coeff_log - threshold_log)
        # where:
        #       [scaling_coeff_log]
        #     = log(1 / (2pi ** 1.5 * gs.scales.exp()))
        #     = -1.5 * log(2pi) - gs.scales
        #       [cov3d_inv]
        #     = M_inv @ M_inv.T
        # thus:
        #       (x.T @ M_inv) @ (x.T @ M_inv).T > 2 * (scaling_coeff_log - threshold_log)

        xcxt_min_bound = -2 * (1.5 * np.log(2 * np.pi) + np.log(threshold) + self.scales) # [..., 1]
        assert (xcxt_min_bound > 0).all()
        offset_min_bound = (xcxt_min_bound / M_inv.square().sum(-1)).sqrt()               # [..., 3]
        return torch.stack((
            self.means - offset_min_bound,
            self.means + offset_min_bound,
        ), dim=-2)                                                                        # [..., 2, 3]

    @torch.no_grad()
    def get_cov3d_shape(self, *, iso_values: float = 1.0) -> TriangleMesh:
        eye = torch.eye(3, device=self.device)                              # [3, 3]
        scales = self.scales.exp().view(-1, 3) * iso_values                 # [N, 3]
        local_vertices = scales[:, None, :] * torch.cat((eye, -eye), dim=0) # [N, 6, 3]
        offsets = torch.matmul(
            quat2rot((self.quats / self.quats.norm(dim=-1, keepdim=True)).view(-1, 1, 4)),
            local_vertices[..., None],
        ).squeeze(-1)                                                       # [N, 6, 3]
        vertices = self.means.view(-1, 3)[:, None, :] + offsets             # [N, 6, 3]
        local_indices = torch.tensor([
            [0, 1, 2],
            [1, 3, 2],
            [3, 4, 2],
            [4, 0, 2],
            [0, 5, 1],
            [1, 5, 3],
            [3, 5, 4],
            [4, 5, 0],
        ]).long().to(self.device)                                           # [8, 3]

        indices = local_indices + (torch.arange(vertices.shape[0], device=self.device) * 6).view(-1, 1, 1) # [N, 8, 3]
        return TriangleMesh(vertices=vertices.view(-1, 3), indices=indices.view(-1, 3))

    @torch.no_grad()
    def as_points(self, num_samples: int) -> Points:
        volumes = self.scales.sum(-1).exp().view(-1)                                     # [N]
        probs = volumes / volumes.sum()
        indices = torch.multinomial(probs, num_samples, replacement=True)                # [S]
        randn = torch.randn(num_samples, 3, device=self.device)
        offsets = randn * self.scales.view(-1, 3)[indices, :].exp()                      # [S, 3]
        rotated_offsets = quat2rot(
            (self.quats / self.quats.norm(dim=-1, keepdim=True)).view(-1, 4)[indices]
        ) @ offsets[..., None] # [S, 3, 1]
        positions = self.means.view(-1, 3)[indices, :] + rotated_offsets.squeeze(-1)     # [S, 3]
        colors = self.colors.view(-1, 3)[indices, :]
        return Points(positions=positions, colors=colors)
