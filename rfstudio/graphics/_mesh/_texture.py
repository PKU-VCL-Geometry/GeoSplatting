from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor
from torch.autograd.function import FunctionCtx

from rfstudio.io import dump_float32_image, load_float32_image
from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from .._cameras import Cameras
from .._images import PBRImages, RGBImages
from .._spherical_gaussians import SphericalGaussians
from ..math import get_rotation_from_axis_angle, safe_normalize
from ._splitsum import (
    diffuse_cubemap as _diffuse_prefilter_cubemap,
    specular_cubemap as _specular_prefilter_cubemap,
)


@dataclass
class Texture2D(TensorDataclass):
    width: int = Size.Dynamic
    height: int = Size.Dynamic
    data: Tensor = Float[height, width, 3]

    @property
    def is_constant(self) -> bool:
        return self.width == 1 and self.height == 1

    @classmethod
    def from_constants(
        cls,
        value: Tuple[float, float, float],
        *,
        device: Optional[torch.device] = None,
    ) -> Texture2D:
        v0, v1, v2 = value
        return Texture2D(data=torch.tensor((v0, v1, v2)).view(1, 1, 3).float().to(device))

    @classmethod
    def from_image_file(
        cls,
        image_file: Path,
        *,
        device: Optional[torch.device] = None,
    ) -> Texture2D:
        kd = load_float32_image(image_file).flip(0).to(device)
        return Texture2D(data=kd)

    def export(self, image_file: Path) -> None:
        dump_float32_image(image_file, self.data.flip(0))

    @classmethod
    def from_mtl_file(
        cls,
        mtl_file: Path,
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Tuple[Texture2D, Optional[Texture2D]]]:
        assert mtl_file.exists() and mtl_file.is_file()

        with mtl_file.open() as f:
            lines = f.readlines()

        materials = {}
        ks = None
        kd = None
        name = None
        for line in lines:
            split_line = re.split(' +|\t+|\n+', line.strip())
            prefix = split_line[0].lower()
            data = split_line[1:]
            if 'newmtl' in prefix:
                if name is not None:
                    assert kd is not None
                    kd = torch.where(
                        kd <= 0.04045,
                        kd / 12.92,
                        torch.pow((kd.clamp_min(0.04045) + 0.055) / 1.055, 2.4),
                    ) # map sRGB to RGB
                    materials[name] = (
                        Texture2D(data=kd),
                        (Texture2D(data=ks) if ks is not None else None),
                    )
                ks = None
                kd = None
                name = data[0]
            elif 'map_kd' in prefix:
                assert (kd is None or (kd == 1).all().item()) and name is not None
                assert len(data) == 1
                kd = load_float32_image(mtl_file.parent / data[0]).flip(0).to(device)
                if kd.ndim == 2:
                    kd = kd.unsqueeze(-1).repeat(1, 1, 3).contiguous()
                assert kd.ndim == 3 and kd.shape[-1] == 3
            elif 'map_ks' in prefix:
                assert ks is None and name is not None
                assert len(data) == 1
                ks = load_float32_image(mtl_file.parent / data[0]).flip(0).to(device)
                if ks.ndim == 2:
                    ks = ks.unsqueeze(-1).repeat(1, 1, 3).contiguous()
                assert ks.ndim == 3 and ks.shape[-1] == 3
            elif 'bump' in prefix:
                pass
            elif 'kd' in prefix:
                assert kd is None and name is not None
                assert len(data) == 3
                kd = torch.tensor([float(d) for d in data], device=device).float().expand(1, 1, 3).contiguous()
            elif 'ks' in prefix:
                assert ks is None and name is not None
                assert len(data) == 3
                ks = torch.tensor([float(d) for d in data], device=device).float().expand(1, 1, 3).contiguous()
            else:
                pass
        assert kd is not None and name is not None
        kd = torch.where(
            kd <= 0.04045,
            kd / 12.92,
            torch.pow((kd.clamp_min(0.04045) + 0.055) / 1.055, 2.4),
        ) # map sRGB to RGB
        materials[name] = (
            Texture2D(data=kd),
            (Texture2D(data=ks) if ks is not None else None),
        )
        return materials

    def resize_to(self, width: int, height: int) -> Texture2D:
        permuted = self.data.permute(2, 0, 1)[None] # [1, H, W, C]
        if self.width >= width:
            if self.height >= height:
                resized = F.interpolate(permuted, (height, width), mode='area')
            else:
                resized = F.interpolate(permuted, (self.height, width), mode='area')
                resized = F.interpolate(
                    resized,
                    (height, width),
                    mode='bilinear',
                    align_corners=True,
                )
        else:  # noqa: PLR5501
            if self.height >= height:
                resized = F.interpolate(permuted, (height, self.width), mode='area')
                resized = F.interpolate(
                    resized,
                    (height, width),
                    mode='bilinear',
                    align_corners=True,
                )
            else:
                resized = F.interpolate(
                    permuted,
                    (height, width),
                    mode='bilinear',
                    align_corners=True,
                )
        return Texture2D(data=resized.squeeze(0).permute(1, 2, 0).contiguous())

    def expand_to(self, width: int, height: int) -> Texture2D:
        return Texture2D(data=self.data.expand(height, width, -1))

    def sample(self, texture_coordinates: Float32[Tensor, "*bs H W 2"]) -> Float32[Tensor, "*bs H W 3"]:
        tex = self.data.unsqueeze(0)
        texc = texture_coordinates.view(-1, *texture_coordinates.shape[-3:]) # [B, H, W, 2]
        texc = torch.stack((
            texc[..., 0] * (1 - 1 / self.width) + (0.5 / self.width),
            texc[..., 1] * (1 - 1 / self.height) + (0.5 / self.height),
        ), dim=-1)
        return dr.texture(tex, texc, filter_mode='linear').view(*texture_coordinates.shape[:-1], 3)


def _cube_to_dir(
    s: Literal[0, 1, 2, 3, 4, 5],
    x: Float32[Tensor, "..."],
    y: Float32[Tensor, "..."],
) -> Float32[Tensor, "... 3"]:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    else:
        raise ValueError
    return torch.stack((rx, ry, rz), dim=-1)

class _CubeMapMip(torch.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, cubemap: Float32[Tensor, "6 H W C"]) -> Float32[Tensor, "6 H//2 W//2 C"]:
        assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2]
        cubemap = cubemap.permute(0, 3, 1, 2) # [6, C, H, W]
        cubemap = F.avg_pool2d(cubemap, (2, 2))
        return cubemap.permute(0, 2, 3, 1).contiguous() # [6, H//2, W//2, C]

    @staticmethod
    def backward(ctx: FunctionCtx, dout: Float32[Tensor, "6 H//2 W//2 C"]) -> Float32[Tensor, "6 H W C"]:
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device=dout.device)
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device=dout.device),
                indexing='ij',
            ) # [H, W]
            v = _cube_to_dir(s, gx, gy)
            v = v / v.norm(dim=-1, keepdim=True)
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25,
                v[None, ...],
                filter_mode='linear',
                boundary_mode='cube',
            )
        return out

def _merge_mipmaps(mipmaps: List[Float32[Tensor, "... R R 3"]]) -> Float32[Tensor, "... 4 R R"]:
    R = mipmaps[0].shape[-2]
    results = torch.stack((
        mipmaps[0][..., 0],
        mipmaps[0][..., 1],
        mipmaps[0][..., 2],
        torch.empty_like(mipmaps[0][..., 0]),
    ), dim=-3)                                 # [..., 4, R, R]
    origin = 0
    for i in range(1, len(mipmaps)):
        assert mipmaps[i].shape[-3:-1] == (R // 2, R // 2)
        results[..., 3, origin:origin+R//2, origin:origin+R//2] = mipmaps[i][..., 0]
        results[..., 3, origin:origin+R//2, origin+R//2:origin+R] = mipmaps[i][..., 1]
        results[..., 3, origin+R//2:origin+R, origin:origin+R//2] = mipmaps[i][..., 2]
        origin += R // 2
        R = R // 2
    return results


def _split_mipmaps(
    mipmaps: Float32[Tensor, "... 4 R R"],
    *,
    num_mipmaps: int
) -> List[Float32[Tensor, "... R R 3"]]:
    results = []
    BS = mipmaps.shape[:-3]
    for _ in range(num_mipmaps):
        R = mipmaps.shape[-1]
        permuted = mipmaps[..., :3, :, :].flatten(-2, -1).transpose(-2, -1) # [..., R*R, 3]
        results.append(permuted.reshape(*BS, R, R, 3).contiguous())
        HR = R // 2
        mipmaps = mipmaps[..., 3, :, :].view(*BS, 2, HR, 2, HR).transpose(-3, -2) # [..., 2, 2, HR, HR]
        mipmaps = mipmaps.reshape(*BS, 4, HR, HR)
    return results

@dataclass
class TextureLatLng(TensorDataclass):
    width: int = Size.Dynamic
    height: int = Size.Dynamic
    data: Tensor = Float[height, width, 3]
    pdf: Optional[Tensor] = Float[height, width, 3]
    transform: Optional[Tensor] = Float[3, 3]

    @classmethod
    def from_image_file(
        cls,
        image_file: Path,
        *,
        scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> TextureLatLng:
        assert image_file.exists() and image_file.suffix in ['.hdr', '.exr']
        hdr = load_float32_image(image_file).to(device)
        assert hdr.dtype == torch.float32
        if scale is not None:
            hdr = hdr * scale
        return TextureLatLng(data=hdr, transform=None)

    @torch.no_grad()
    def compute_pdf_(self) -> TextureLatLng:
        Y, _ = torch.meshgrid(
            (torch.arange(0, self.height, device=self.device) + 0.5) / self.height,
            (torch.arange(0, self.width, device=self.device) + 0.5) / self.width,
            indexing='ij',
        )

        # Scale by sin(theta) for lat-long, https://cs184.eecs.berkeley.edu/sp18/article/25
        pdf = self.data.max(-1).values.clamp_min(1e-3) * (Y * torch.pi).sin()
        pdf = pdf / pdf.sum()

        # Compute cumulative sums over the columns and rows
        cols = torch.cumsum(pdf, dim=1)
        rows = torch.cumsum(cols[:, -1:].repeat(1, cols.shape[1]), dim=0)

        # Normalize
        cols = cols / torch.where(cols[:, -1:] > 0, cols[:, -1:], torch.ones_like(cols))
        rows = rows / torch.where(rows[-1:, :] > 0, rows[-1:, :], torch.ones_like(rows))
        return self.replace_(pdf=torch.stack((pdf, rows, cols), dim=-1))

    def z_up_to_y_up_(self) -> None:
        transform = torch.tensor([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ]).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def y_up_to_z_up_(self) -> None:
        transform = torch.tensor([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ]).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def rotateX_(self, rad: float) -> None:
        transform = get_rotation_from_axis_angle(
            axis=torch.tensor([1., 0., 0.]),
            angle=torch.stack((torch.tensor(rad).cos(), torch.tensor(rad).sin())),
        ).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def rotateZ_(self, rad: float) -> None:
        transform = get_rotation_from_axis_angle(
            axis=torch.tensor([0., 0., 1.]),
            angle=torch.stack((torch.tensor(rad).cos(), torch.tensor(rad).sin())),
        ).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def rotateY_(self, rad: float) -> None:
        transform = get_rotation_from_axis_angle(
            axis=torch.tensor([0., 1., 0.]),
            angle=torch.stack((torch.tensor(rad).cos(), torch.tensor(rad).sin())),
        ).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def as_cubemap(self, *, resolution: int) -> TextureCubeMap:
        cubemap = torch.zeros(6, resolution, resolution, 3, device=self.device)

        for s in range(6):
            linspace = torch.linspace(-1.0 + 1.0 / resolution, 1.0 - 1.0 / resolution, resolution, device=self.device)
            gy, gx = torch.meshgrid(linspace, linspace, indexing='ij') # [H, W]
            v = _cube_to_dir(s, gx, gy)
            v = v / v.norm(dim=-1, keepdim=True)

            tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * torch.pi) + 0.5
            tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / torch.pi
            texcoord = torch.cat((tu, tv), dim=-1)

            cubemap[s, ...] = dr.texture(
                self.data[None, ...].cuda(),
                texcoord[None, ...].cuda(),
                filter_mode='linear',
            )[0].to(cubemap.device)
        return TextureCubeMap(data=cubemap, transform=self.transform)

    def visualize(self) -> RGBImages:
        if self.transform is None:
            return RGBImages([self.data])
        return self.as_cubemap(resolution=max(self.width, self.height)).visualize(width=self.width, height=self.height)


@dataclass
class TextureCubeMap(TensorDataclass):
    resolution: int = Size.Dynamic
    data: Tensor = Float[6, resolution, resolution, 3]
    transform: Optional[Tensor] = Float[3, 3]

    @classmethod
    def from_image_file(
        cls,
        image_file: Path,
        *,
        resolution: int = 512,
        scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> TextureCubeMap:
        texture = TextureLatLng.from_image_file(
            image_file,
            scale=scale,
            device=device,
        )
        return texture.cuda().as_cubemap(resolution=resolution).to(texture.device)

    def downsample(self) -> TextureCubeMap:
        assert self.resolution % 2 == 0
        return TextureCubeMap(data=_CubeMapMip.apply(self.data), transform=None)

    def sample(self, directions: Float32[Tensor, "*bs H W 3"]) -> Float32[Tensor, "*bs H W 3"]:
        H, W, C = directions.shape[-3:]
        assert C == 3
        batched_dirs = (directions / directions.norm(dim=-1, keepdim=True)).view(-1, H, W, 3)
        return dr.texture(
            self.data.unsqueeze(0),
            batched_dirs.contiguous(),
            filter_mode='linear',
            boundary_mode='cube',
        ).view_as(directions)

    def render(self, camera: Cameras) -> RGBImages:
        assert camera.device == self.device
        assert camera.shape == ()
        if not self.is_cuda:
            return self.cuda().render(camera.cuda()).cpu()
        transform = torch.eye(3).to(camera.c2w) if self.transform is None else self.transform
        pixel_coords = camera.pixel_coordinates
        offset_y = (0.5 - camera.cy + pixel_coords[..., 0]) / camera.fy
        offset_x = (0.5 - camera.cx + pixel_coords[..., 1]) / camera.fx
        directions = (transform @ camera.c2w[:3, :3] @ torch.stack((
            offset_x,
            -offset_y,
            -torch.ones_like(offset_x),
        ), dim=-1)[..., None]).squeeze(-1) # [H, W, 3]
        img = self.sample(directions)
        return PBRImages([img]).rgb2srgb()

    def visualize(self, *, width: int = 800, height: int = 400) -> RGBImages:
        if not self.is_cuda:
            return self.cuda().visualize(width=width, height=height).cpu()
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / height, 1.0 - 1.0 / height, height, device=self.device) * torch.pi,
            torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, device=self.device) * torch.pi,
            indexing='ij',
        )
        sin_theta = gy.sin()
        reflvec = torch.stack((sin_theta * gx.sin(), gy.cos(), -sin_theta * gx.cos()), dim=-1)
        if self.transform is not None:
            reflvec = (self.transform @ reflvec.unsqueeze(-1)).squeeze(-1)
        img = self.sample(reflvec)
        return RGBImages([img])

    def as_latlng(self, *, width: int = 1024, height: int = 512, apply_transform: bool = False) -> TextureLatLng:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / height, 1.0 - 1.0 / height, height, device=self.device) * torch.pi,
            torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, device=self.device) * torch.pi,
            indexing='ij',
        )
        sin_theta = gy.sin()
        reflvec = torch.stack((sin_theta * gx.sin(), gy.cos(), -sin_theta * gx.cos()), dim=-1).cuda()
        if apply_transform and self.transform is not None:
            reflvec = (self.transform.cuda() @ reflvec.unsqueeze(-1)).squeeze(-1)
            return TextureLatLng(data=self.cuda().sample(reflvec).to(self.device))
        return TextureLatLng(data=self.cuda().sample(reflvec).to(self.device), transform=self.transform)

    def z_up_to_y_up_(self) -> None:
        transform = torch.tensor([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ]).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def rotateY_(self, rad: float) -> None:
        transform = get_rotation_from_axis_angle(
            axis=torch.tensor([0., 1., 0.]),
            angle=torch.stack((torch.tensor(rad).cos(), torch.tensor(rad).sin())),
        ).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def as_sg(
        self,
        num_gaussians: int,
        *,
        batch_size: int = 16384,
        num_epochs: int = 20,
    ) -> TextureSG:
        linspace = torch.linspace(
            -1.0 + 1.0 / self.resolution,
            1.0 - 1.0 / self.resolution,
            self.resolution,
            device=self.device,
        )
        gy, gx = torch.meshgrid(linspace, linspace, indexing='ij') # [H, W]
        inputs = torch.stack([F.normalize(_cube_to_dir(s, gx, gy), dim=-1) for s in range(6)]).view(-1, 3)
        gt_outputs = self.data.view(-1, 3)
        texture_sg = TextureSG.from_random(num_gaussians, device=inputs.device)
        learnable_parameters = torch.cat((
            texture_sg.axis,
            texture_sg.sharpness,
            texture_sg.amplitude,
        ), dim=-1).requires_grad_()
        texture_sg = TextureSG(
            axis=learnable_parameters[..., 0:3],
            sharpness=learnable_parameters[..., 3:4],
            amplitude=learnable_parameters[..., 4:7]
        )
        with torch.enable_grad():
            optimizer = torch.optim.Adam([learnable_parameters], lr=0.1)
            for _ in range(num_epochs):
                num_batches = inputs.shape[0] // batch_size
                indices = torch.randperm(inputs.shape[0], device=inputs.device)[:num_batches * batch_size]
                for i in range(num_batches):
                    batch_indices = indices[i * batch_size : i * batch_size + batch_size]
                    batch_inputs = inputs[batch_indices]
                    batch_gt_outputs = gt_outputs[batch_indices]
                    optimizer.zero_grad()
                    loss = F.l1_loss(texture_sg.sample(batch_inputs), batch_gt_outputs)
                    loss.backward()
                    optimizer.step()
        return texture_sg.detach()

    def as_splitsum(
        self,
        *,
        cutoff: float = 0.99,
        min_resolution: int = 16,
        min_roughness: float = 0.08,
        max_roughness: float = 0.5,
    ) -> TextureSplitSum:
        mipmaps: List[TextureCubeMap] = [self] # [6, R, R, 3]
        while mipmaps[-1].resolution > min_resolution:
            mipmaps += [mipmaps[-1].downsample()]
        assert len(mipmaps) > 2, "Min resolution is too large."

        base = _diffuse_prefilter_cubemap(mipmaps[-1].data)

        for idx in range(len(mipmaps) - 1):
            roughness = (idx / (len(mipmaps) - 2)) * (max_roughness - min_roughness) + min_roughness
            mipmaps[idx] = _specular_prefilter_cubemap(mipmaps[idx].data, roughness=roughness, cutoff=cutoff)
        mipmaps[-1] = _specular_prefilter_cubemap(mipmaps[-1].data, roughness=1.0, cutoff=cutoff)

        return TextureSplitSum(
            base=base,
            mipmaps=_merge_mipmaps(mipmaps),
            num_mipmaps=torch.tensor([len(mipmaps)], device=base.device, dtype=torch.long),
            min_roughness=base.new_empty(1).fill_(min_roughness),
            max_roughness=base.new_empty(1).fill_(max_roughness),
            transform=self.transform,
        )

@dataclass
class TextureSplitSum(TensorDataclass):

    base_resolution: int = Size.Dynamic
    mip_resolution: int = Size.Dynamic
    base: Tensor = Float[6, base_resolution, base_resolution, 3]
    mipmaps: Tensor = Float[6, 4, mip_resolution, mip_resolution]
    num_mipmaps: Tensor = Long[1]
    min_roughness: Tensor = Float[1]
    max_roughness: Tensor = Float[1]
    transform: Optional[Tensor] = Float[3, 3]

    def sample(
        self,
        normals: Float32[Tensor, "*bs H W 3"],
        directions: Float32[Tensor, "*bs H W 3"],
        *,
        roughness: Float32[Tensor, "*bs H W 1"],
    ) -> Tuple[
        Float32[Tensor, "*bs H W 3"],
        Float32[Tensor, "*bs H W 3"],
    ]:
        if self.transform is not None:
            normals = (self.transform @ normals[..., None]).squeeze(-1)
            directions = (self.transform @ directions[..., None]).squeeze(-1)
        miplevel = torch.where(
            roughness < self.max_roughness,
            torch.div(
                roughness - self.min_roughness,
                self.max_roughness - self.min_roughness,
            ).clamp(0, 1) * (self.num_mipmaps - 2),
            torch.div(
                roughness - self.max_roughness,
                1.0 - self.max_roughness,
            ).clamp(0, 1) + self.num_mipmaps - 2
        ) # [..., H, W, 1] \in [0, self.num_mipmaps - 1)

        l_diffuse = dr.texture(
            self.base.unsqueeze(0),
            normals,
            filter_mode='linear',
            boundary_mode='cube',
        ) # [..., H, W, 3]

        mipmaps = _split_mipmaps(self.mipmaps, num_mipmaps=self.num_mipmaps.item())
        l_specular = dr.texture(
            mipmaps[0].unsqueeze(0),
            directions.contiguous(),
            mip=[m.unsqueeze(0) for m in mipmaps[1:]],
            mip_level_bias=miplevel[..., 0],
            filter_mode='linear-mipmap-linear',
            boundary_mode='cube',
        ) # [..., H, W, 3]

        return l_diffuse, l_specular

    def z_up_to_y_up_(self) -> None:
        transform = torch.tensor([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0],
        ]).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def y_up_to_z_up_(self) -> None:
        transform = torch.tensor([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ]).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

    def rotateY_(self, rad: float) -> None:
        transform = get_rotation_from_axis_angle(
            axis=torch.tensor([0., 1., 0.]),
            angle=torch.stack((torch.tensor(rad).cos(), torch.tensor(rad).sin())),
        ).float().to(self.device)
        if self.transform is None:
            self.replace_(transform=transform)
        else:
            self.replace_(transform=transform @ self.transform)

@dataclass
class TextureSG(TensorDataclass):
    num_gaussians: int = Size.Dynamic
    axis: Tensor = Float[num_gaussians, 3]
    sharpness: Tensor = Float[num_gaussians, 1]
    amplitude: Tensor = Float[num_gaussians, 3]

    @staticmethod
    def from_random(num_gaussians: int, *, device: Optional[torch.device] = None) -> TextureSG:
        return TextureSG(
            axis=torch.randn((num_gaussians, 3), device=device),
            sharpness=3 + torch.randn((num_gaussians, 1), device=device) / 3,
            amplitude=torch.randn((num_gaussians, 3), device=device) / 3 - 2,
        )

    def visualize(self, *, width: int = 800, height: int = 400) -> RGBImages:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / height, 1.0 - 1.0 / height, height, device=self.device) * torch.pi,
            torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, device=self.device) * torch.pi,
            indexing='ij',
        )
        sin_theta = gy.sin()
        reflvec = torch.stack((sin_theta * gx.sin(), gy.cos(), -sin_theta * gx.cos()), dim=-1)
        img = self.sample(reflvec)
        return RGBImages([img])

    def sample(self, directions: Float32[Tensor, "*bs 3"]) -> Float32[Tensor, "*bs 3"]:
        log = self.sharpness.exp() * ((directions[..., None, :] * safe_normalize(self.axis)).sum(-1, keepdim=True) - 1)
        return (self.amplitude.exp() * log.exp()).sum(-2)

    def integral(
        self,
        normals: Float32[Tensor, "*bs 3"],
        wo: Float32[Tensor, "*bs 3"],
        *,
        albedo: Float32[Tensor, "*bs 3"],
        roughness: Float32[Tensor, "*bs 1"],
        metallic: Float32[Tensor, "*bs 1"],
    ) -> Tuple[
        Float32[Tensor, "*bs 3"],
        Float32[Tensor, "*bs 3"],
    ]:
        light_sg = SphericalGaussians(
            axis=safe_normalize(self.axis),
            sharpness=self.sharpness.exp(),
            amplitude=self.amplitude.exp(),
        )
        specular_sg = SphericalGaussians.from_brdf_lobe(
            normals=normals,
            wo=wo,
            roughness=roughness,
        )
        new_half = safe_normalize(specular_sg.axis + wo)
        v_dot_h = (wo * new_half).sum(-1, keepdim=True).clamp(1e-4)
        F0 = 0.04 * (1 - metallic) + metallic * albedo
        F = F0 + (1. - F0) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)
        NoV = (normals * wo).sum(dim=-1, keepdim=True).clamp_min(1e-4)
        NoL = (specular_sg.axis * normals).sum(-1, keepdim=True).clamp(1e-4)
        k = roughness.square() / 2.
        G1 = NoV / (NoV * (1 - k) + k + 1e-6)
        G2 = NoL / (NoL * (1 - k) + k + 1e-6)
        G = G1 * G2
        Moi = F * G / (4 * NoV * NoL + 1e-6)
        light_specular_sg = light_sg @ specular_sg[..., None] # [..., K]
        specular_term = light_specular_sg.cosine_integral(normals[..., None, :])
        diffuse_term = light_sg.cosine_integral(normals[..., None, :])
        return diffuse_term * (albedo / torch.pi), specular_term * Moi
