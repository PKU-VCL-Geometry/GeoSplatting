from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Long, TensorDataclass

from ._rays import Rays
from .math import get_rotation_from_relative_vectors, least_squares, slerp


def _get_spherical_positions_from_pitch_yaw(
    *,
    yaw: Float32[Tensor, "S"],
    pitch: Float32[Tensor, "S"],
    up: Float32[Tensor, "3"],
    radius: float,
) -> Float32[Tensor, "S 3"]:
    local_offsets = torch.stack((
        yaw.cos() * pitch.cos(),
        yaw.sin() * pitch.cos(),
        pitch.sin().expand(yaw.shape),
    ), dim=-1) * radius                                                                               # [S, 3]
    z_axis = torch.tensor([0, 0, 1]).to(up)
    rotated = (get_rotation_from_relative_vectors(z_axis, up) @ local_offsets[..., None]).squeeze(-1) # [S, 3]
    return rotated


@dataclass
class Cameras(TensorDataclass):

    c2w: Tensor = Float[..., 3, 4]

    fx: Tensor = Float[...]

    fy: Tensor = Float[...]

    cx: Tensor = Float[...]

    cy: Tensor = Float[...]

    width: Tensor = Long[...]

    height: Tensor = Long[...]

    near: Tensor = Float[...]

    far: Tensor = Float[...]

    @property
    def has_all_same_resolution(self) -> bool:
        return (self.width == self.width.view(-1)[0]).all() and (self.height == self.height.view(-1)[0]).all()

    def resize(self, scale: float) -> Cameras:
        return self.replace(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=(self.width * scale).round().long(),
            height=(self.height * scale).round().long(),
        )

    @classmethod
    def from_lookat(
        cls,
        *,
        eye: Union[Tuple[float, float, float], Float32[Tensor, "*bs 3"]] = (0., 0., 1.),
        target: Union[Tuple[float, float, float], Float32[Tensor, "*bs 3"]] = (0., 0., 0.),
        up: Union[Tuple[float, float, float], Float32[Tensor, "*bs 3"]] = (0., 1., 0.),
        resolution: Union[Tuple[int, int], Int64[Tensor, "*bs 2"]] = (1280, 720),
        hfov_degree: Union[float, Float32[Tensor, "*bs"]] = 90.,
        near: Union[float, Float32[Tensor, "*bs"]] = 1e-3,
        far: Union[float, Float32[Tensor, "*bs"]] = 1e3,
        device: Optional[torch.device] = None
    ) -> Cameras:
        shape = None
        params = (eye, target, up, resolution, hfov_degree, near, far)
        dims = (3, 3, 3, 2, None, None, None)
        types = (torch.float32, torch.float32, torch.float32, torch.long, torch.float32, torch.float32, torch.float32)
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
        eye, target, up, resolution, hfov_degree, near, far = params

        forward = target - eye                                              # [..., 3]
        right = torch.cross(forward, up)                                    # [..., 3]
        up = torch.cross(right, forward)
        R = torch.stack((right, up, -forward), dim=-1)                      # [..., 3, 3]
        T = eye[..., None]                                                  # [..., 3, 1]
        cx = resolution[..., 0] * 0.5                                       # [...]
        cy = resolution[..., 1] * 0.5                                       # [...]
        focal_length = cx / torch.tan(hfov_degree * (0.5 * torch.pi / 180)) # [...]
        return Cameras(
            c2w=torch.cat((R / R.norm(dim=-2, keepdim=True), T), dim=-1),
            fx=focal_length,
            fy=focal_length.clone(),
            cx=cx,
            cy=cy,
            width=resolution[..., 0],
            height=resolution[..., 1],
            near=near,
            far=far
        )

    @classmethod
    def from_orbit(
        cls,
        *,
        center: Union[Tuple[float, float, float], Float32[Tensor, "3"]],
        up: Union[Tuple[float, float, float], Float32[Tensor, "3"]],
        radius: float,
        pitch_degree: float,
        num_samples: int,
        resolution: Tuple[int, int] = (1280, 720),
        hfov_degree: float = 90.,
        near: float = 1e-3,
        far: float = 1e3,
        device: Optional[torch.device] = None
    ) -> Cameras:
        if not isinstance(center, Tensor):
            center = torch.tensor(center).float().to(device)
        if device is None:
            device = center.device
        else:
            assert device == center.device
        if not isinstance(up, Tensor):
            up = torch.tensor(up).float().to(device)
        else:
            assert device == up.device
        pitch = torch.tensor(pitch_degree, device=device) * torch.pi / 180                                # [1]
        yaw = (2 * torch.pi / num_samples) * torch.arange(num_samples, device=device)                     # [S]
        offsets = _get_spherical_positions_from_pitch_yaw(yaw=yaw, pitch=pitch, up=up, radius=radius)     # [S, 3]
        return cls.from_lookat(
            eye=offsets + center,
            target=center.view(1, 3).repeat(num_samples, 1),                                              # [S, 3]
            up=up.view(1, 3).repeat(num_samples, 1),                                                      # [S, 3]
            resolution=resolution,
            hfov_degree=hfov_degree,
            near=near,
            far=far,
            device=device
        )

    @classmethod
    def from_sphere(
        cls,
        *,
        center: Union[Tuple[float, float, float], Float32[Tensor, "3"]],
        up: Union[Tuple[float, float, float], Float32[Tensor, "3"]],
        radius: float,
        num_samples: int,
        resolution: Tuple[int, int] = (1280, 720),
        hfov_degree: float = 90.,
        near: float = 1e-3,
        far: float = 1e3,
        device: Optional[torch.device] = None
    ) -> Cameras:
        if not isinstance(center, Tensor):
            center = torch.tensor(center).float().to(device)
        if device is None:
            device = center.device
        else:
            assert device == center.device
        if not isinstance(up, Tensor):
            up = torch.tensor(up).float().to(device)
        else:
            assert device == up.device
        pitch = torch.arcsin(2 * torch.rand(num_samples, device=device) - 1)                              # [S]
        yaw = (2 * torch.pi) * torch.rand(num_samples, device=device)                                     # [S]
        offsets = _get_spherical_positions_from_pitch_yaw(yaw=yaw, pitch=pitch, up=up, radius=radius)     # [S, 3]
        return cls.from_lookat(
            eye=offsets + center,
            target=center.view(1, 3).repeat(num_samples, 1),                                              # [S, 3]
            up=up.view(1, 3).repeat(num_samples, 1),                                                      # [S, 3]
            resolution=resolution,
            hfov_degree=hfov_degree,
            near=near,
            far=far,
            device=device
        )

    @classmethod
    def from_hemisphere(
        cls,
        *,
        center: Union[Tuple[float, float, float], Float32[Tensor, "3"]],
        up: Union[Tuple[float, float, float], Float32[Tensor, "3"]],
        radius: float,
        num_samples: int,
        resolution: Tuple[int, int] = (1280, 720),
        hfov_degree: float = 90.,
        near: float = 1e-3,
        far: float = 1e3,
        device: Optional[torch.device] = None
    ) -> Cameras:
        if not isinstance(center, Tensor):
            center = torch.tensor(center).float().to(device)
        if device is None:
            device = center.device
        else:
            assert device == center.device
        if not isinstance(up, Tensor):
            up = torch.tensor(up).float().to(device)
        else:
            assert device == up.device
        pitch = torch.arcsin(torch.rand(num_samples, device=device))                                      # [S]
        yaw = (2 * torch.pi) * torch.rand(num_samples, device=device)                                     # [S]
        offsets = _get_spherical_positions_from_pitch_yaw(yaw=yaw, pitch=pitch, up=up, radius=radius)     # [S, 3]
        return cls.from_lookat(
            eye=offsets + center,
            target=center.view(1, 3).repeat(num_samples, 1),                                              # [S, 3]
            up=up.view(1, 3).repeat(num_samples, 1),                                                      # [S, 3]
            resolution=resolution,
            hfov_degree=hfov_degree,
            near=near,
            far=far,
            device=device
        )

    def transform_to_fit_sphere(
        self,
        *,
        radius: float = 1.0,
        auto_orient: Literal['+y', '-y', '+z', 'none'] = 'none',
    ) -> Cameras:

        with torch.no_grad():
            eyes = self.c2w[..., :3, 3:] # [N, 3, 1]
            directions = -self.c2w[..., :3, 2:3] # [N, 3, 1]
            A = torch.eye(3, device=self.device) - directions @ directions.transpose(-1, -2) # [N, 3, 3]
            b = (A @ eyes).sum(0) # [3, 1]
            A = A.sum(0) # [3, 3]
            sphere_center = least_squares(A, b) # [3, 1]

        eyes = self.c2w[..., :3, 3:] - sphere_center # [N, 3, 1]
        eyes: Tensor = (eyes / eyes.norm(dim=-2, keepdim=True)) * radius # [N, 3, 1]

        if auto_orient == 'none':
            c2w = torch.cat((self.c2w[..., :3, :3], eyes), dim=-1)
            return self.replace(c2w=c2w)

        with torch.no_grad():
            polar_directions = self.c2w[..., :3, 1:2] # [N, 3, 1]
            tangents = eyes.cross(polar_directions, dim=-2) # [N, 3, 1]
            # we want to find the pole so that (pole.dot(tangents) == 0).all()
            pole = least_squares(tangents.squeeze(-1)).squeeze(-1) # [3]
            if (polar_directions.squeeze(-1) * pole).sum(-1).mean(0) < 0:
                pole = -pole

            if auto_orient == '+y':
                rotation = get_rotation_from_relative_vectors(pole, torch.tensor([0, 1, 0]).to(pole))
            elif auto_orient == '+z':
                rotation = get_rotation_from_relative_vectors(pole, torch.tensor([0, 0, 1]).to(pole))
            elif auto_orient == '-y':
                rotation = get_rotation_from_relative_vectors(pole, torch.tensor([0, -1, 0]).to(pole))
            else:
                raise ValueError(auto_orient)

        translation = rotation @ eyes # [N, 3, 1]
        rotation = rotation @ self.c2w[..., :3, :3] # [N, 3, 3]
        c2w = torch.cat((rotation, translation), dim=-1)
        return self.replace(c2w=c2w)

    @property
    def intrinsic_matrix(self) -> Float32[Tensor, "*bs 3 3"]:
        K = self.c2w.new_zeros(self.shape + (3, 3))
        K[..., 0, 0] = self.fx
        K[..., 1, 1] = self.fy
        K[..., 0, 2] = self.cx
        K[..., 1, 2] = self.cy
        K[..., 2, 2] = 1
        return K

    @property
    def view_matrix(self) -> Float32[Tensor, "*bs 4 4"]:
        R = self.c2w[..., :3, :3]              # 3 x 3
        T = self.c2w[..., :3, 3:4]             # 3 x 1

        # flip the z and y axes to align with gsplat conventions
        R = R * torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype)

        # analytic matrix inverse to get world2camera matrix
        R_inv = R.transpose(-1, -2)
        T_inv = R_inv @ -T
        viewmat = self.c2w.new_zeros(self.shape + (4, 4))
        viewmat[..., 3, 3] = 1.0               # homogenous
        viewmat[..., :3, :3] = R_inv
        viewmat[..., :3, 3:4] = T_inv
        return viewmat

    @property
    def projection_matrix(self) -> Float32[Tensor, "*bs 4 4"]:
        P = self.c2w.new_zeros(self.shape + (4, 4))
        t = self.cy * (self.near / self.fy)
        b = (self.cy - self.height) * (self.near / self.fy)
        r = self.cx * (self.near / self.fx)
        l = (self.cx - self.width) * (self.near / self.fx)  # noqa: E741
        n = self.near
        f = self.far
        P[..., 0, 0] = 2 * n / (r - l)
        P[..., 0, 2] = (r + l) / (r - l)
        P[..., 1, 1] = 2 * n / (t - b)
        P[..., 1, 2] = (t + b) / (t - b)
        P[..., 2, 2] = (f + n) / (f - n)
        P[..., 2, 3] = -2 * f * n / (f - n)
        P[..., 3, 2] = 1.0
        return P

    def project(self, positions: Float32[Tensor, "*bs 3"]) -> Float32[Tensor, "*bs 3"]:
        assert self.shape == ()
        pose = self.c2w
        xyz_camera_space = (pose[:3, :3].T @ (positions.unsqueeze(-1) - pose[:3, 3:])).squeeze(-1) # [..., 3]
        xyz_base = xyz_camera_space[..., 0:2] / -xyz_camera_space[..., 2:3]
        return torch.cat((
            xyz_base[..., 0:1] * self.fx + self.cx,
            -xyz_base[..., 1:2] * self.fy + self.cy,
            -xyz_camera_space[..., 2:3],
        ), dim=-1)

    @property
    def pixel_coordinates(self) -> Int64[Tensor, "H W 2"]:
        return _CachedPixelCoords.get(
            height=self.height.max().item(),
            width=self.width.max().item(),
            device=self.device,
        )

    def generate_rays(self, num_rays_per_batch: Optional[int] = None) -> Rays:
        cameras = self.view(-1)
        if num_rays_per_batch is None:
            assert self.shape == ()
            pixel_coords = cameras.pixel_coordinates.view(-1, 2)
            camera_indices = torch.zeros_like(pixel_coords[:, 0])
            pixel_coord_y = pixel_coords[:, 0]
            pixel_coord_x = pixel_coords[:, 1]
        else:
            camera_indices = torch.randint(
                low=0,
                high=cameras.shape[0],
                size=(num_rays_per_batch, ),
                device=cameras.device,
            ) # [S]
            pixel_coord_y = (torch.rand(
                num_rays_per_batch,
                device=cameras.device,
            ) * cameras.height[camera_indices]).long()
            pixel_coord_x = (torch.rand(
                num_rays_per_batch,
                device=cameras.device,
            ) * cameras.width[camera_indices]).long()
        cameras = cameras[camera_indices]
        offset_y = (0.5 - cameras.cy + pixel_coord_y) / cameras.fy
        offset_x = (0.5 - cameras.cx + pixel_coord_x) / cameras.fx
        directions = (
            cameras.c2w[:, :3, :3] @
            torch.stack((offset_x, -offset_y, -torch.ones_like(offset_x)), dim=-1).unsqueeze(-1)
        ).squeeze(-1)
        rays = Rays(
            origins=cameras.c2w[:, :, 3],
            directions=directions,
            pixel_indices=torch.stack((
                camera_indices,
                pixel_coord_y,
                pixel_coord_x,
            ), dim=-1),
            near=cameras.near.unsqueeze(-1),
            far=cameras.far.unsqueeze(-1),
        ) # [num_rays_per_batch]

        if num_rays_per_batch is None:
            rays = rays.view(self.height.item(), self.width.item())

        return rays

    def sample_sequentially(self, num_samples: int, *, uniform_by: Literal['index', 'distance']) -> Cameras:
        assert self.ndim == 1 and self.shape[0] > 1 and num_samples > 1
        translations = self.c2w[:, :, 3]                                                   # [N, 3]
        rotations = self.c2w[:, :, :3]                                                     # [N, 3, 3]
        if uniform_by == 'index':
            slices = torch.arange(self.shape[0]) / (self.shape[0] - 1)                     # [N]
        elif uniform_by == 'distance':
            distances = (translations.roll(1, dims=0) - translations).norm(dim=-1)         # [N]
            distances[0] = 0
            slices = distances.cumsum(0) / distances.sum(0)                                # [N]
        else:
            raise NotImplementedError
        samples = torch.linspace(0, 1, num_samples, device=self.device)                    # [S]
        lerp_lefts = torch.searchsorted(slices, samples).clamp(1, slices.shape[0] - 1) - 1 # [S]
        weights = torch.div(
            samples - slices[lerp_lefts],
            slices[lerp_lefts + 1] - slices[lerp_lefts],
        ).clamp(0, 1)                                                                      # [S]
        lerped_translations = torch.add(
            translations[lerp_lefts, :] * (1 - weights[:, None]),
            translations[lerp_lefts + 1, :] * weights[:, None],
        )[:, :, None]                                                                      # [S, 3, 1]
        lerped_rotations = slerp(
            rotations[lerp_lefts, :, :],
            rotations[lerp_lefts + 1, :, :],
            weights=weights,
        )                                                                                  # [S, 3, 3]
        return self[0].expand(num_samples).replace(
            c2w=torch.cat((lerped_rotations, lerped_translations), dim=-1),
        ).contiguous()


class _CachedPixelCoords:

    cache = {}

    @classmethod
    def get(cls, *, height: int, width: int, device: torch.device) -> Int64[Tensor, "H W 2"]:
        cached = cls.cache.get(str(device))
        if cached is None or cached.shape[0] < height or cached.shape[1] < width:
            cached = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            cached = torch.stack(cached, dim=-1)
            cls.cache[str(device)] = cached
        return cached[:height, :width, :] # stored as (y, x) coordinates
