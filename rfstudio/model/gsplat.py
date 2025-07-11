from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import open3d as o3d
import torch
from gsplat import rasterization, rasterization_2dgs
from jaxtyping import Float32
from torch import Tensor

from rfstudio.graphics import Cameras, DepthImages, RGBAImages, RGBImages, Splats
from rfstudio.graphics.math import rgb2sh
from rfstudio.nn import Module
from rfstudio.utils.tensor_dataclass import Float, Int, TensorDataclass


@dataclass
class GSplatter(Module):

    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 1.0
    "Size of the cube to initialize random gaussians within"
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""

    block_width: int = 16

    background_color: Literal["random", "black", "white"] = "random"

    prepare_densification: bool = False

    rasterize_mode: Literal["classic", "antialiased", "2dgs"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel.
    This approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured,
    which results "aliasing-like" artifacts.
    The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers
    that were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """

    def __setup__(self) -> None:
        self.gaussians = Splats.random(
            size=self.num_random,
            sh_degree=self.sh_degree,
            random_scale=self.random_scale,
            device=self.device,
            requires_grad=True
        )
        self.update_info = None
        self.max_sh_degree = None

    def state_dict(self) -> Dict[str, Tensor]:
        return {
            'means': self.gaussians.means,
            'colors': self.gaussians.colors,
            'shs': self.gaussians.shs,
            'opacities': self.gaussians.opacities,
            'scales': self.gaussians.scales,
            'quats': self.gaussians.quats
        }

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        self.gaussians = Splats(
            means=state_dict['means'],
            colors=state_dict['colors'],
            shs=state_dict['shs'],
            opacities=state_dict['opacities'],
            scales=state_dict['scales'],
            quats=state_dict['quats']
        ).to(self.device).requires_grad_(self.gaussians.requires_grad)

    @torch.no_grad()
    def export_point_cloud(self, path: pathlib.Path) -> None:
        map_to_tensors = {}
        map_to_tensors["positions"] = self.gaussians.means.cpu().numpy()
        map_to_tensors["normals"] = np.zeros_like(map_to_tensors["positions"], dtype=np.float32)

        features_dc = rgb2sh(self.gaussians.colors.cpu().numpy())      # [N, 3]
        for i in range(3):
            map_to_tensors[f"f_dc_{i}"] = features_dc[:, i, None]
        map_to_tensors["opacity"] = self.gaussians.opacities.cpu().numpy()
        scales = self.gaussians.scales.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = (self.gaussians.quats / self.gaussians.quats.norm(dim=-1, keepdim=True)).cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        pcd = o3d.t.geometry.PointCloud(map_to_tensors)
        o3d.t.io.write_point_cloud(str(path), pcd)

    def get_background_color(self) -> Float32[Tensor, "3 "]:
        if self.background_color == 'black':
            return torch.zeros(3)
        if self.background_color == 'white':
            return torch.ones(3)
        if self.training:
            return torch.rand(3)
        return torch.tensor([0.1490, 0.1647, 0.2157])

    def set_max_sh_degree(self, value: Optional[int] = None) -> None:
        self.max_sh_degree = value

    def render_depth(self, inputs: Cameras) -> DepthImages:

        # TODO: fix the device
        with torch.no_grad():
            if self.gaussians.device != self.device:
                requires_grad = self.gaussians.requires_grad
                self.gaussians.swap_(self.gaussians.to(self.device))
                self.gaussians.requires_grad_(requires_grad)

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())

        if self.rasterize_mode not in ["antialiased", "classic", "2dgs"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        if self.rasterize_mode == "2dgs":
            render, alpha, _, _, _, _, info = rasterization_2dgs(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=self.gaussians.colors.detach(),
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='ED',
                sh_degree=None,
                sparse_grad=False,
                absgrad=False,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
        else:
            render, alpha, info = rasterization(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=self.gaussians.colors.detach(),
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='ED',
                sh_degree=None,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )

        key = "gradient_2dgs" if self.rasterize_mode == '2dgs' else "means2d"
        if self.training and self.prepare_densification and info[key].requires_grad:
            info[key].retain_grad()
            self.update_info = UpdateInfo(
                xys=info[key],
                radii=info["radii"],
                indices=info["gaussian_ids"],
                last_width=camera.width[None],
                last_height=camera.height[None]
            )

        depth = torch.cat((render, alpha), dim=-1) # [1, H, W, 2]
        return DepthImages([depth.squeeze(0)])

    def render_rgb(self, inputs: Cameras) -> RGBImages:

        # TODO: fix the device
        with torch.no_grad():
            if self.gaussians.device != self.device:
                requires_grad = self.gaussians.requires_grad
                self.gaussians.swap_(self.gaussians.to(self.device))
                self.gaussians.requires_grad_(requires_grad)

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())
        background_color = self.get_background_color().to(camera.device)
        sh_degree = (
            self.gaussians.sh_degree
            if self.max_sh_degree is None
            else min(self.max_sh_degree, self.gaussians.sh_degree)
        )

        if self.rasterize_mode not in ["antialiased", "classic", "2dgs"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        if sh_degree == 0:
            colors = self.gaussians.colors
            sh_degree = None
        else:
            colors = torch.cat((rgb2sh(self.gaussians.colors[..., None, :]), self.gaussians.shs), dim=-2)

        if self.rasterize_mode == "2dgs":
            render, alpha, normal, pseudo_normal, distort, _, info = rasterization_2dgs(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=colors,
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='RGB+ED',
                sh_degree=sh_degree,
                distloss=self.training,
                sparse_grad=False,
                absgrad=False,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
        else:
            render, alpha, info = rasterization(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=colors,
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='RGB',
                sh_degree=sh_degree,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )

        key = "gradient_2dgs" if self.rasterize_mode == '2dgs' else "means2d"
        if self.training and self.prepare_densification and info[key].requires_grad:
            normal_loss = None
            distort_loss = None
            if self.rasterize_mode == '2dgs':
                normal_loss = (1 - (normal * (pseudo_normal * alpha)).sum(-1)).mean()[None]
                distort_loss = distort.mean()[None]
            info[key].retain_grad()
            self.update_info = UpdateInfo(
                xys=info[key],
                radii=info["radii"],
                indices=info["gaussian_ids"],
                last_width=camera.width[None],
                last_height=camera.height[None],
                distort_loss=distort_loss,
                normal_loss=normal_loss,
            )

        rgb = render[..., :3] + (1 - alpha) * background_color
        return RGBImages([rgb.squeeze(0)])

    def render_rgba(self, inputs: Cameras) -> RGBAImages:

        # TODO: fix the device
        with torch.no_grad():
            if self.gaussians.device != self.device:
                requires_grad = self.gaussians.requires_grad
                self.gaussians.swap_(self.gaussians.to(self.device))
                self.gaussians.requires_grad_(requires_grad)

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())
        sh_degree = (
            self.gaussians.sh_degree
            if self.max_sh_degree is None
            else min(self.max_sh_degree, self.gaussians.sh_degree)
        )

        if self.rasterize_mode not in ["antialiased", "classic", "2dgs"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        if sh_degree == 0:
            colors = self.gaussians.colors
            sh_degree = None
        else:
            colors = torch.cat((rgb2sh(self.gaussians.colors[..., None, :]), self.gaussians.shs), dim=-2)

        if self.rasterize_mode == "2dgs":
            render, alpha, _, _, _, _, info = rasterization_2dgs(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=colors,
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='RGB',
                sh_degree=sh_degree,
                sparse_grad=False,
                absgrad=False,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
        else:
            render, alpha, info = rasterization(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=colors,
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='RGB',
                sh_degree=sh_degree,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )

        rgba = torch.cat((render[..., :3], alpha), dim=-1)
        return RGBAImages([rgba.squeeze(0)])

    @torch.no_grad()
    def update_grad_norm(self) -> None:
        assert self.update_info is not None
        # keep track of a moving average of grad norms
        grads = self.update_info.xys.grad.norm(dim=-1)                 # [V]
        if self.gaussians.xys_grad_norm is None:
            self.gaussians.annotate_(
                xys_grad_norm=grads.new_zeros(self.gaussians.shape),
                vis_counts=grads.new_ones(self.gaussians.shape),
                last_width=self.update_info.last_width,
                last_height=self.update_info.last_height
            )
        assert self.gaussians.vis_counts is not None
        self.gaussians.vis_counts[self.update_info.indices] += 1
        self.gaussians.xys_grad_norm[self.update_info.indices] += grads


@dataclass
class UpdateInfo(TensorDataclass):
    xys: torch.Tensor = Float[..., 2]
    radii: torch.Tensor = Int[...]
    indices: torch.Tensor = Int[...]
    last_width: torch.Tensor = Float[1]
    last_height: torch.Tensor = Float[1]
    normal_loss: Optional[torch.Tensor] = Float[1]
    distort_loss: Optional[torch.Tensor] = Float[1]
