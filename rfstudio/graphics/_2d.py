from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import cv2
import numpy as np
import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.io import dump_float32_image
from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass
from rfstudio.utils.typing import IntArrayLike, IntLike

from ._images import RGBAImages


class RGBA2DImages:

    def __init__(self, tensors: Union[Tensor, Iterator[Tensor]]) -> None:
        C = 4
        if not isinstance(tensors, Tensor):
            tensors = torch.stack(tensors)
        assert tensors.ndim in [2, 3] and tensors.shape[-1] == C, (
            f"Tensors must have shape (B, W, {C}) or (W, {C})."
        )
        if tensors.ndim == 2:
            tensors = tensors.unsqueeze(0)
        self._tensors = tensors

    def __getitem__(self, indices: Union[IntArrayLike, IntLike]) -> RGBA2DImages:
        return RGBA2DImages(self._tensors[indices])

    def get(self, idx: int) -> Tensor:
        return self._tensors[idx]

    def __len__(self) -> int:
        return len(self._tensors)

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self._tensors)

    def to(self, device: torch.device) -> RGBA2DImages:
        return RGBA2DImages(self._tensors.to(device))

    def detach(self) -> RGBA2DImages:
        return RGBA2DImages(self._tensors.detach())

    def cpu(self) -> RGBA2DImages:
        return RGBA2DImages(self._tensors.cpu())

    @property
    def device(self) -> torch.device:
        return self._tensors.device

    def item(self) -> Tensor:
        assert len(self._tensors) == 1
        return self._tensors[0]

    def visualize(
        self,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> RGBAImages:
        N = len(self._tensors)
        try_height = (width if height is None else height) // N * N
        images = RGBAImages([self._tensors.repeat_interleave(N, dim=0)])
        if width is None and height is None:
            return images
        return images.resize_to(
            width=self._tensors.shape[1] if width is None else width,
            height=try_height if height is None else height,
        )

def _shading2D(x: Float32[Tensor, "... 2"], *, scale: float = 1) -> Float32[Tensor, "... 3"]:
    colors = (x / (2 * scale) + 0.5).clamp(0, 1) # [..., 2]
    return torch.cat((colors, 1 - colors[..., 0:1] * colors[..., 1:2]), dim=-1)

@runtime_checkable
class Visualizable(Protocol):

    def visualize(self, *, width: int, height: int, scale: float = 1.) -> RGBAImages:
        ...

@dataclass
class CircleShape2D(TensorDataclass):

    num_circles: int = Size.Dynamic
    origins: Tensor = Float[num_circles, 2]
    radius: Tensor = Float[num_circles, 1]

    def render(self, cameras: Cameras2D) -> RGBA2DImages:
        rays = cameras.flatten().generate_rays() # [N, W]
        desired_shape = rays.shape # [N, W]
        rays = rays.flatten() # [R]
        oc = rays.origins - self.origins[:, None, :] # [C, R, 2]
        a = rays.directions.square().sum(-1) # [R]
        b = 2 * (rays.directions * oc).sum(-1) # [C, R]
        c = oc.square().sum(-1) - self.radius.square() # [C, R]
        discriminant = b.square() - 4 * a * c
        valid = ~(discriminant < 0)
        sqrt_discriminant = discriminant.clamp_min(0).sqrt()
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        t1 = torch.where((t1 > cameras.near) & valid, t1, 2 * cameras.far)
        t2 = torch.where((t2 > cameras.near) & valid, t2, 2 * cameras.far)
        ts = torch.min(t1, t2).min(0).values.unsqueeze(-1) # [R, 1]
        intersections = rays.origins + ts * rays.directions # [R, 2]
        alphas = (ts < cameras.far).float() # [R, 1]
        rgba = torch.cat((_shading2D(intersections) * alphas, alphas), dim=-1).view(*desired_shape, 4) # [N, W, 4]
        return RGBA2DImages(rgba)

    def visualize(self, *, width: int, height: int, scale: float = 1.) -> RGBAImages:
        xs = torch.linspace(-scale, scale, width, device=self.device) # [W]
        ys = torch.linspace(-scale, scale, height, device=self.device) # [H]
        Ys, Xs = torch.meshgrid(ys, xs, indexing='ij') # [H, W]
        queries = torch.stack((Xs, Ys), dim=-1).flip(0) # [H, W, 2]
        alphas = torch.less(
            (queries - self.origins[:, None, None, :]).square().sum(-1, keepdim=True),
            self.radius[:, None, None, :].square(),
        ).any(0).float() # [H, W, 1]
        rgba = torch.cat((_shading2D(queries) * alphas, alphas), dim=-1) # [H, W, 4]
        return RGBAImages([rgba])

    @staticmethod
    def random(size: int, *, device: Optional[torch.device] = None) -> CircleShape2D:
        radius = torch.rand(size, 1, device=device) * 0.2 + 0.1 # radius \in [0.1, 0.3]
        origins = (torch.rand(size, 2, device=device) * 2 - 1) * ((1 - radius) * 0.8)
        return CircleShape2D(radius=radius, origins=origins)

@dataclass
class Rays2D(TensorDataclass):

    origins: Tensor = Float[..., 2]

    directions: Tensor = Float[..., 2]

    near: Tensor = Float[1]

    far: Tensor = Float[1]

    def get_samples(self, t: Float32[Tensor, "... S+1"]) -> RaySamples2D:
        bins = (self.near + (self.far - self.near) * t).unsqueeze(-1) # [..., S+1, 1]
        return RaySamples2D(
            origins=self.origins,
            directions=self.directions,
            bins=bins[..., :-1, :],
            max_bin=bins[..., -1, :],
        )

@dataclass
class RaySamples2D(TensorDataclass):

    num_samples: int = Size.Dynamic

    origins: Tensor = Float[..., 2]

    directions: Tensor = Float[..., 2]

    bins: Tensor = Float[..., num_samples, 1]

    max_bin: Tensor = Float[..., 1]

    colors: Optional[Tensor] = Float[..., num_samples, 3]

    densities: Optional[Tensor] = Float[..., num_samples, 1]

    weights: Optional[Tensor] = Float[..., num_samples, 1]

    @property
    def positions(self) -> Float32[Tensor, "*bs S 2"]:
        bins = torch.cat((self.bins, self.max_bin.unsqueeze(-1)), dim=-2) # [..., S+1, 1]
        return self.origins[..., None, :] + (bins[..., :-1, :] + bins[..., 1:, :]) / 2 * self.directions[..., None, :]

    def get_weighted(self) -> RaySamples2D:
        assert self.densities is not None, "Densities should be given to compute weights"
        assert self.weights is None, "Weights have already been computed"

        distances = torch.cat((
            self.bins[..., 1:, :] - self.bins[..., :-1, :],
            self.max_bin.unsqueeze(-1) - self.bins[..., -1:, :],
        ), dim=-2) # [..., S, 1]
        distances = (distances * self.directions.norm(dim=-1, keepdim=True).unsqueeze(-1)) # [..., S, 1]
        delta_density = distances * self.densities                # [..., S, 1]
        alphas = 1 - torch.exp(-delta_density)                         # [..., S, 1]
        transmittance = delta_density.cumsum(dim=-2)                   # [..., S, 1]
        transmittance = torch.cat((
            torch.zeros_like(transmittance[..., :1, :]),
            delta_density.cumsum(dim=-2)[..., :-1, :],
        ), dim=-2)
        transmittance = torch.exp(-transmittance)                      # [..., S, 1]
        weights = torch.nan_to_num(alphas * transmittance)
        return self.annotate(weights=weights)


@dataclass
class Cameras2D(TensorDataclass):

    c2w: Tensor = Float[..., 2, 3]

    focal: Tensor = Float[1]

    center: Tensor = Float[1]

    width: Tensor = Long[1]

    near: Tensor = Float[1]

    far: Tensor = Float[1]

    @classmethod
    def from_lookat(
        cls,
        *,
        eye: Union[Tuple[float, float], Float32[Tensor, "*bs 2"]] = (1., 0.),
        target: Union[Tuple[float, float], Float32[Tensor, "*bs 2"]] = (0., 0.),
        width: int = 800,
        hfov_degree: float = 90.,
        near: float = 1e-3,
        far: float = 1e3,
        device: Optional[torch.device] = None
    ) -> Cameras2D:
        shape = None
        params = (eye, target)
        dims = (2, 2)
        types = (torch.float32, torch.float32)
        for p, D in zip(params, dims):
            if not isinstance(p, Tensor):
                continue
            if device is None:
                device = p.device
            else:
                assert p.device == device
            if D is not None:
                assert p.shape[-1] == D
                sl = slice(None, -1)
            else:
                sl = slice(None)
            if shape is None:
                shape = p.shape[sl]
            else:
                assert shape == p.shape[sl]
        if device is None:
            device = torch.empty(0).device
        if shape is None:
            shape = ()
        shapes = (((*shape, D) if D is not None else shape) for D in dims)
        params = ((p if isinstance(p, Tensor) else torch.tensor(p, device=device)) for p in params)
        params = (p.to(t).expand(shape).contiguous() for p, shape, t in zip(params, shapes, types))
        eye, target = params
        width = torch.tensor([width], device=device).long()
        hfov_degree = torch.tensor([hfov_degree], device=device)
        near = torch.tensor([near], device=device)
        far = torch.tensor([far], device=device)

        forward = target - eye                                              # [..., 2]
        right = torch.stack((forward[..., 1], -forward[..., 0]), dim=-1)    # [..., 2]
        R = torch.stack((forward, -right), dim=-1)                          # [..., 2, 2]
        T = eye[..., None]                                                  # [..., 2, 1]
        cx = width * 0.5                                                    # [...]
        focal_length = cx / torch.tan(hfov_degree * (0.5 * torch.pi / 180)) # [...]
        return Cameras2D(
            c2w=torch.cat((R / R.norm(dim=-2, keepdim=True), T), dim=-1),
            focal=focal_length,
            center=cx,
            width=width,
            near=near,
            far=far
        )

    @classmethod
    def from_orbit(
        cls,
        *,
        center: Union[Tuple[float, float], Float32[Tensor, "2"]],
        radius: float,
        num_samples: int,
        width: int = 800,
        hfov_degree: float = 90.,
        near: float = 1e-3,
        far: float = 1e3,
        device: Optional[torch.device] = None
    ) -> Cameras2D:
        if not isinstance(center, Tensor):
            center = torch.tensor(center).float().to(device)
        if device is None:
            device = center.device
        else:
            assert device == center.device
        yaw = (2 * torch.pi / num_samples) * torch.arange(num_samples, device=device) # [S]
        offsets = torch.stack((yaw.cos(), yaw.sin()), dim=-1) * radius # [S, 2]
        return cls.from_lookat(
            eye=offsets + center,
            target=center.view(1, 2).repeat(num_samples, 1), # [S, 2]
            width=width,
            hfov_degree=hfov_degree,
            near=near,
            far=far,
            device=device
        )

    def generate_rays(self, *, downsample_to: Optional[int] = None) -> Rays2D:
        W = self.width.item()
        num_rays = W if downsample_to is None else downsample_to
        pixel_coord_x = torch.linspace(0.5, W - 0.5, num_rays, device=self.device)
        offset_x = (pixel_coord_x - self.center) / self.focal
        directions = (
            self.c2w[..., None, :2, :2] @
            torch.stack((torch.ones_like(offset_x), -offset_x), dim=-1).unsqueeze(-1)
        ).squeeze(-1)
        return Rays2D(
            origins=self.c2w[..., None, :, 2].expand_as(directions).contiguous(),
            directions=directions,
            near=self.near,
            far=self.far,
        )

@dataclass
class Viser2D:

    scale: float = 1.

    resolution: int = 800

    def __post_init__(self) -> None:
        self._canvas = np.zeros((self.resolution, self.resolution, 4), dtype=np.uint8)

    def show(self, item: Union[Cameras2D, CircleShape2D, Rays2D, RaySamples2D]) -> Viser2D:
        if isinstance(item, Visualizable):
            rgba = (item.visualize(
                width=self.resolution,
                height=self.resolution,
                scale=self.scale,
            ).item().detach().cpu().numpy() * 255).astype(np.uint8)
            self._canvas = np.concatenate((
                rgba[..., :3] + self._canvas[..., :3] * (1 - rgba[..., 3:]),
                rgba[..., 3:] + self._canvas[..., 3:] * (1 - rgba[..., 3:]),
            ), axis=-1)
        elif isinstance(item, Cameras2D):
            cam_color = (208, 118, 2, 255)
            c2w = item.c2w.view(-1, 2, 3)
            pts = c2w[:, None, :2, :2] @ (torch.tensor([
                [0, 0],
                [1, -0.5],
                [1, 0.5],
            ]).view(3, 2, 1).to(c2w) * (0.1 / self.scale)) + c2w[:, None, :2, 2:] # [C, 3, 2, 1]
            pts[..., 1, :] = -pts[..., 1, :]
            pts = (pts.squeeze(-1).detach().cpu() * (self.resolution / 2) + (self.resolution / 2)).int() # [C, 3, 2]
            for i in range(pts.shape[0]):
                cv2.line(self._canvas, pts[i, 0].tolist(), pts[i, 1].tolist(), color=cam_color, lineType=cv2.LINE_AA)
                cv2.line(self._canvas, pts[i, 1].tolist(), pts[i, 2].tolist(), color=cam_color, lineType=cv2.LINE_AA)
                cv2.line(self._canvas, pts[i, 2].tolist(), pts[i, 0].tolist(), color=cam_color, lineType=cv2.LINE_AA)
        elif isinstance(item, Rays2D):
            ray_color = (170, 170, 170, 255)
            item = item.flatten().detach().cpu()
            starts = ((item.origins * 0.5 + 0.5) * self.resolution).int() # [R, 2]
            ends = (((item.origins + item.directions * item.far) * 0.5 + 0.5) * self.resolution).int() # [R, 2]
            starts[:, 1] = self.resolution - starts[:, 1]
            ends[:, 1] = self.resolution - ends[:, 1]
            for i in range(item.shape[0]):
                cv2.line(self._canvas, starts[i].tolist(), ends[i].tolist(), color=ray_color, lineType=cv2.LINE_AA)
        elif isinstance(item, RaySamples2D):
            ray_sample_color = (72, 152, 93, 255)
            radius = self.resolution // 200
            positions = ((item.positions.detach().cpu().view(-1, 2) * 0.5 + 0.5) * self.resolution).int() # [S, 2]
            positions[:, 1] = self.resolution - positions[:, 1]
            for i in range(positions.shape[0]):
                cv2.circle(
                    self._canvas,
                    positions[i].tolist(),
                    radius,
                    color=ray_sample_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
        else:
            raise NotImplementedError
        return self

    def export(self, output: Path) -> None:
        dump_float32_image(output, torch.from_numpy(self._canvas) / 255.)

    def get(self) -> RGBAImages:
        return RGBAImages([torch.from_numpy(self._canvas) / 255.])
