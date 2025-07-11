from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, Tuple, Type, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
from jaxtyping import Float32
from torch import Tensor

from ._images import DepthImages, IntensityImages, PBRAImages, RGBAImages, VectorImages
from ._mesh import Texture2D, TextureCubeMap, TextureLatLng, TextureSG, TextureSplitSum
from ._mesh._triangle_mesh import BaseShader, ShadingContext
from .math import get_arbitrary_tangents_from_normals, safe_normalize

_ASSETS_DIR: Path = files('rfstudio') / 'assets' / 'geometry' / 'pbr'

@lru_cache(maxsize=64)
def _get_fg_lut(resolution: Literal[256], device: torch.device) -> Float32[Tensor, "1 R R 2"]:
    filename = _ASSETS_DIR / f'bsdf_{resolution}_{resolution}.bin'
    lut = torch.from_numpy(np.fromfile(_ASSETS_DIR / filename, dtype=np.float32))
    return lut.to(device=device, dtype=torch.float32).view(1, resolution, resolution, 2)

@lru_cache(maxsize=8)
def _get_daylight_splitsum(device: torch.Tensor, z_up: bool) -> TextureSplitSum:
    return _get_daylight_cubemap(device=device, z_up=z_up).as_splitsum()

@lru_cache(maxsize=8)
def _get_daylight_cubemap(device: torch.Tensor, z_up: bool) -> TextureCubeMap:
    envmap = TextureCubeMap.from_image_file(
        Path('data') / 'tensoir' / 'city.hdr',
        device=device,
    )
    envmap.replace_(data=envmap.data.mean(dim=-1, keepdim=True).expand_as(envmap.data) * 0.6)
    if z_up:
        envmap.z_up_to_y_up_()
    return envmap

@dataclass
class PBRShader(BaseShader[PBRAImages]):

    normal_type: Literal['flat', 'face', 'vertex'] = 'vertex'
    occlusion_type: Literal['texture', 'ssao', 'ao', 'none'] = 'texture'
    min_visibility: float = 0.0
    bend_back_normal: bool = False
    envmap: Union[TextureSplitSum, TextureSG] = ...

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        frag_uv = context.facial_vertex_attribute_map(context.mesh.uvs)
        frag_n = context.normals(normal_type=self.normal_type, bend_backward=self.bend_back_normal)
        kd = context.mesh.kd.sample(frag_uv) # [1, H, W, 3]
        ks = context.mesh.ks.sample(frag_uv) # [1, H, W, 3]
        roughness = ks[..., 1:2] # [1, H, W, 1]
        metallic  = ks[..., 2:3] # [1, H, W, 1]
        frag_wo = -context.view_directions

        if isinstance(self.envmap, TextureSG):
            l_diff, l_spec = self.envmap.integral(frag_n, frag_wo, albedo=kd, roughness=roughness, metallic=metallic)
            return l_diff + l_spec

        specular  = (1.0 - metallic) * 0.04 + kd * metallic # [1, H, W, 3]
        diffuse  = kd * (1.0 - metallic) # [1, H, W, 3]

        n_dot_v = (frag_n * frag_wo).sum(-1, keepdim=True) # [1, H, W, 1]
        if not self.bend_back_normal:
            n_dot_v = n_dot_v.clamp(min=1e-4)

        fg_uv = torch.cat((n_dot_v, roughness), dim=-1) # [1, H, W, 2]
        fg_lookup = dr.texture(
            _get_fg_lut(resolution=256, device=frag_uv.device),
            fg_uv,
            filter_mode='linear',
            boundary_mode='clamp',
        ) # [1, H, W, 2]

        # Compute aggregate lighting
        frag_inv_wi = 2 * (frag_wo * frag_n).sum(-1, keepdim=True) * frag_n - frag_wo # [1, H, W, 3]
        l_diff, l_spec = self.envmap.sample(
            normals=frag_n,
            directions=frag_inv_wi,
            roughness=roughness,
        ) # [1, H, W, 3]
        reflectance = specular * fg_lookup[..., 0:1] + fg_lookup[..., 1:2] # [1, H, W, 3]

        if self.occlusion_type == 'ao':
            visibility = context.ao.clamp_min(self.min_visibility)
            colors = (l_diff * diffuse * visibility + l_spec * reflectance) # [1, H, W, 3]
        elif self.occlusion_type == 'none':
            colors = l_diff * diffuse + l_spec * reflectance # [1, H, W, 3]
        else:
            assert self.occlusion_type in ['ssao', 'texture']
            visibility = (
                (1.0 - ks[..., 0:1]).clamp_min(self.min_visibility)
                if self.occlusion_type == 'texture'
                else context.ssao(num_samples=32, min_visibility=self.min_visibility, sample_radius=0.02)
            ) # [1, H, W, 1]
            colors = (l_diff * diffuse + l_spec * reflectance) * visibility # [1, H, W, 3]
        return colors

@dataclass
class NormalShader(BaseShader[VectorImages]):

    normal_type: Literal['flat', 'face', 'vertex'] = 'vertex'
    bend_back_normal: bool = False
    antialias: bool = False

    def get_image_class(self) -> Type[VectorImages]:
        return VectorImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        return context.normals(normal_type=self.normal_type, bend_backward=self.bend_back_normal)

@dataclass
class DepthShader(BaseShader[DepthImages]):

    antialias: bool = False

    def get_image_class(self) -> Type[DepthImages]:
        return DepthImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 1"]:
        return torch.mul(
            context.camera.c2w[:3, 3] - context.global_positions,
            context.camera.c2w[:3, 2]
        ).sum(-1, keepdim=True) # [1, H, W, 1]

@dataclass
class SSAOShader(BaseShader[IntensityImages]):

    antialias: bool = False

    num_samples: int = 32

    sample_radius: float = 0.02

    min_visibility: float = 0.0

    def get_image_class(self) -> Type[IntensityImages]:
        return IntensityImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 1"]:
        return context.ssao(
            num_samples=self.num_samples,
            min_visibility=self.min_visibility,
            sample_radius=self.sample_radius,
        )

@dataclass
class PureShader(BaseShader[PBRAImages]):

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        frag_uv = context.facial_vertex_attribute_map(context.mesh.uvs)
        return context.mesh.kd.sample(frag_uv) # [1, H, W, 3]

@dataclass
class FlatShader(BaseShader[RGBAImages]):

    face_colors: Float32[Tensor, "F 3"] = ...

    def get_image_class(self) -> Type[RGBAImages]:
        return RGBAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        assert self.face_colors.shape == (context.mesh.num_faces, 3)
        return context.face_attribute_map(self.face_colors)

@dataclass
class ShadowShader(BaseShader[PBRAImages]):

    envmap: TextureLatLng = ...
    roughness: float = ...

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        context = context.clone()
        mesh = context.mesh
        context._mesh = mesh.replace(
            kd=Texture2D.from_constants((1.0, 1.0, 1.0), device=context.mesh.device),
            ks=Texture2D.from_constants((1.0, self.roughness, 0.0), device=context.mesh.device),
            uvs=context.mesh.vertices.new_zeros(context.mesh.num_faces, 3, 2),
        )
        result = MCShader(envmap=self.envmap, normal_type='vertex', render_type='diffuse').shade(context)
        context._mesh = mesh
        return result.max(-1, keepdim=True).values.clamp(0, 1).expand_as(result)

@dataclass
class PrettyShader(BaseShader[PBRAImages]):

    z_up: bool = False
    occlusion_type: Literal['texture', 'ssao', 'ao', 'none', 'mc'] = 'mc'
    wireframe: bool = False
    normal_type: Literal['flat', 'face', 'vertex'] = 'vertex'

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        context = context.clone()
        mesh = context.mesh
        context._mesh = mesh.replace(
            kd=Texture2D.from_constants((119/255, 150/255, 170/255), device=context.mesh.device),
            ks=Texture2D.from_constants((1.0, 0.25, 0.05), device=context.mesh.device),
            uvs=context.mesh.vertices.new_zeros(context.mesh.num_faces, 3, 2)
        )
        if self.occlusion_type == 'mc':
            envmap = _get_daylight_cubemap(mesh.device, z_up=self.z_up).as_latlng()
            result = MCShader(envmap=envmap, normal_type=self.normal_type).shade(context)
        else:
            envmap = _get_daylight_splitsum(mesh.device, z_up=self.z_up)
            result = PBRShader(
                envmap=envmap,
                occlusion_type=self.occlusion_type,
                normal_type=self.normal_type,
            ).shade(context)
        if self.wireframe:
            wireframe_shader = WireframeShader(line_width=0.02, line_color=(119/512, 150/512, 170/512))
            wireframe = wireframe_shader.shade(context)
            result = torch.where(context._rast[..., -1:] > 0, wireframe, result)
            wireframe_shader.get_image_class()
        context._mesh = mesh
        return result

@dataclass
class WireframeShader(BaseShader[RGBAImages]):

    line_color: Tuple[float, float, float] = (0, 0, 0)
    line_width: float = 0.05

    def __post_init__(self) -> None:
        self._original = None

    @torch.no_grad()
    def get_image_class(self) -> Type[RGBAImages]:
        context, rast = self._original
        context._rast.copy_(rast)
        self._original = None
        return RGBAImages

    @torch.no_grad()
    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        self._original = (context, context._rast.clone())
        rast = context._rast
        valid = (
            (rast[..., 0:1] < self.line_width) |
            (rast[..., 1:2] < self.line_width) |
            ((rast[..., 0:1] + rast[..., 1:2]) > (1 - self.line_width))
        ).float()
        rast[..., -1:].mul_(valid)
        return valid * torch.tensor(self.line_color).to(valid)

@dataclass
class LambertianShader(BaseShader[PBRAImages]):

    normal_type: Literal['flat', 'face', 'vertex'] = 'vertex'
    bend_back_normal: bool = False
    ambient: float = 0.2

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        frag_n = context.normals(normal_type=self.normal_type, bend_backward=self.bend_back_normal)
        frag_uv = context.facial_vertex_attribute_map(context.mesh.uvs)
        frag_wo = -context.view_directions
        n_dot_v = (frag_n * frag_wo).sum(-1, keepdim=True) # [1, H, W, 1]
        if not self.bend_back_normal:
            n_dot_v = n_dot_v.clamp(min=1e-4)
        return context.mesh.kd.sample(frag_uv) * (n_dot_v + self.ambient).clamp_max(1.0) # [1, H, W, 3]

@dataclass
class MCShader(BaseShader[PBRAImages]):

    normal_type: Literal['flat', 'face', 'vertex'] = 'vertex'
    bend_back_normal: bool = False
    envmap: TextureLatLng = ...
    num_samples_per_ray: int = 8
    shadow_scale: float = 1.0
    denoise: bool = True
    render_type: Literal['pbr', 'vis', 'diffuse'] = 'pbr'

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:

        from ._mesh._optix import OptiXContext, bilateral_denoiser, optix_build_bvh, optix_env_shade

        assert self.envmap.is_cuda

        with torch.no_grad():
            optix_ctx = OptiXContext()
            vertices = context.mesh.vertices
            if self.envmap.transform is not None:
                vertices = (self.envmap.transform @ vertices.unsqueeze(-1)).squeeze(-1)
            optix_build_bvh(optix_ctx, vertices.contiguous(), context.mesh.indices.int(), rebuild=1)
        frag_pos = context.global_positions
        frag_n = context.normals(self.normal_type, bend_backward=self.bend_back_normal)
        camera_lookat = -context.camera.c2w[:, 2] # [3]
        camera_pos = context.camera.c2w[:, 3]
        frag_depth = ((frag_pos - camera_pos) * camera_lookat).sum(-1, keepdim=True)
        if self.envmap.transform is not None:
            frag_pos = (self.envmap.transform @ frag_pos.unsqueeze(-1)).squeeze(-1)
            frag_n = (self.envmap.transform @ frag_n.unsqueeze(-1)).squeeze(-1)
            camera_pos = (self.envmap.transform @ camera_pos.unsqueeze(-1)).squeeze(-1)
        frag_uv = context.facial_vertex_attribute_map(context.mesh.uvs)
        kd = context.mesh.kd.sample(frag_uv) # [1, H, W, 3]
        ks = context.mesh.ks.sample(frag_uv) # [1, H, W, 3]
        if self.envmap.pdf is None:
            self.envmap.compute_pdf_()
        diffuse_accum, specular_accum, vis_accum = optix_env_shade(
            optix_ctx,
            context._rast[..., -1],
            frag_pos + frag_n * 1e-3,
            frag_pos,
            frag_n,
            camera_pos.contiguous().view(1, 1, 1, 3),
            kd,
            ks,
            self.envmap.data,
            self.envmap.pdf[..., 0],
            self.envmap.pdf[:, 0, 1],
            self.envmap.pdf[..., 2],
            BSDF='pbr',
            n_samples_x=self.num_samples_per_ray,
            rnd_seed=None,
            shadow_scale=self.shadow_scale,
        )
        if self.render_type == 'pbr':
            if self.denoise:
                sigma = max(self.shadow_scale * 2, 0.0001)
                diffuse_accum  = bilateral_denoiser(diffuse_accum, frag_n, frag_depth, sigma)
                specular_accum = bilateral_denoiser(specular_accum, frag_n, frag_depth, sigma)
            shaded = diffuse_accum * kd * (1.0 - ks[..., 2:3]) + specular_accum
        elif self.render_type == 'diffuse':
            if self.denoise:
                sigma = max(self.shadow_scale * 2, 0.0001)
                diffuse_accum  = bilateral_denoiser(diffuse_accum, frag_n, frag_depth, sigma)
            shaded = diffuse_accum * kd
        elif self.render_type == 'vis':
            if self.denoise:
                sigma = max(self.shadow_scale * 2, 0.0001)
                vis_accum = vis_accum.sum(-1, keepdim=True).expand_as(diffuse_accum).contiguous()
                vis_accum = bilateral_denoiser(vis_accum, frag_n, frag_depth, sigma)
            shaded = 1 - vis_accum
        else:
            raise ValueError(self.render_type)
        return shaded

@dataclass
class PathTraceShader(BaseShader[PBRAImages]):

    normal_type: Literal['flat', 'face', 'vertex'] = 'vertex'
    bend_back_normal: bool = False
    envmap: TextureCubeMap = ...
    num_samples: int = 1024

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def _sample_sphere(self, num_samples, begin_elevation = 0):
        """
        sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(2 * np.pi * n * phi % (2 * np.pi))
            elevations.append(np.arcsin(z))
        return np.array(azimuths), np.array(elevations)

    def __post_init__(self) -> None:
        az, el = self._sample_sphere(self.num_samples // 2)
        az, el = az * 0.5 / torch.pi, 1 - 2 * el / torch.pi # scale to [0,1]
        self.diffuse_direction_samples = torch.from_numpy(np.stack([az, el], -1).astype(np.float32)).cuda() # [dn0,2]
        az, el = self._sample_sphere(self.num_samples - self.num_samples // 2)
        az, el = az * 0.5 / torch.pi, 1 - 2 * el / torch.pi # scale to [0,1]
        self.specular_direction_samples = torch.from_numpy(np.stack([az, el], -1).astype(np.float32)).cuda() # [dn1,2]

    def _sample_specular_directions(self, reflections, roughness):
        z = reflections # [1, H, W, 3]
        x = get_arbitrary_tangents_from_normals(z) # [1, H, W, 3]
        y = torch.cross(z, x, dim=-1)  # [1, H, W, 3]
        a = roughness  # [1, H, W, 1]

        az, el = torch.split(self.specular_direction_samples, 1, dim=1)  # sn,1
        phi = torch.pi * 2 * az # sn,1
        el = el.view(-1, 1, 1, 1) # [sn, 1, 1, 1]
        cos_theta = torch.sqrt((1.0 - el + 1e-6) / (1.0 + (a**2 - 1.0) * el + 1e-6) + 1e-6) # [sn, H, W, 1]
        sin_theta = torch.sqrt(1 - cos_theta**2 + 1e-6) # [sn, H, W, 1]

        phi = (phi.view(-1, 1, 1, 1) + torch.rand(z.shape[0], 1, 1, 1, device=reflections.device) * torch.pi * 2) % (2 * torch.pi) # sn,1,1,1
        coeff_x = torch.cos(phi) * sin_theta # [sn, H, W, 1]
        coeff_y = torch.sin(phi) * sin_theta # [sn, H, W, 1]
        coeff_z = cos_theta # [sn, H, W, 1]

        directions = coeff_x * x + coeff_y * y + coeff_z * z # [sn, H, W, 3]
        return directions

    def _sample_diffuse_directions(self, normals):
        z = normals # [1, H, W, 3]
        x = get_arbitrary_tangents_from_normals(z) # [1, H, W, 3]
        y = torch.cross(z, x, dim=-1)  # [1, H, W, 3]

        # project onto this tangent space
        az, el = torch.split(self.diffuse_direction_samples, 1, dim=1)  # sn,1
        az = az * torch.pi * 2
        el = el.view(-1, 1, 1, 1) # [sn, 1, 1, 1]
        el_sqrt = torch.sqrt(el+1e-7)
        az = (az.view(-1, 1, 1, 1) + torch.rand(z.shape[0], 1, 1, 1, device=normals.device) * torch.pi * 2) % (2 * torch.pi)
        coeff_z = torch.sqrt(1 - el + 1e-7)
        coeff_x = el_sqrt * torch.cos(az)
        coeff_y = el_sqrt * torch.sin(az)

        directions = coeff_x * x + coeff_y * y + coeff_z * z # pn,sn,3
        return directions

    def _saturate_dot(self, a, b):
        return (a * b).sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)

    def _distribution_ggx(self, NoH, roughness):
        a = roughness
        a2 = a**2
        NoH2 = NoH**2
        denom = NoH2 * (a2 - 1.0) + 1.0
        return a2 / (torch.pi * denom**2 + 1e-4)

    def _fresnel_schlick_directions(self, F0, view_dirs, directions):
        H = safe_normalize(view_dirs + directions) # [pn,sn0,3]
        HoV = torch.clamp(torch.sum(H * view_dirs, dim=-1, keepdim=True), min=0.0, max=1.0) # [pn,sn0,1]
        fresnel = self._fresnel_schlick(F0, HoV) # [pn,sn0,1]
        return fresnel, H, HoV

    def _fresnel_schlick(self, F0, HoV):
        return F0 + (1.0 - F0) * torch.clamp(1.0 - HoV, min=0.0, max=1.0)**5.0

    def _geometry_schlick_ggx(self, NoV, roughness):
        a = roughness # a = roughness**2: we assume the predicted roughness is already squared
        k = a / 2
        num = NoV
        denom = NoV * (1 - k) + k
        return num / (denom + 1e-5)

    def _geometry(self, NoV, NoL, roughness):
        ggx2 = self._geometry_schlick_ggx(NoV, roughness)
        ggx1 = self._geometry_schlick_ggx(NoL, roughness)
        return ggx2 * ggx1

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:

        # import raytracing

        # with torch.no_grad():
        #     raytracer = raytracing.RayTracer(context.mesh.vertices, context.mesh.indices)

        frag_n = context.normals(self.normal_type, bend_backward=self.bend_back_normal)
        frag_wo = -context.view_directions
        frag_inv_wi = 2 * (frag_wo * frag_n).sum(-1, keepdim=True) * frag_n - frag_wo # [1, H, W, 3]
        frag_uv = context.facial_vertex_attribute_map(context.mesh.uvs)
        frag_ks = context.mesh.ks.sample(frag_uv) # [1, H, W, 3]
        frag_albedo = context.mesh.kd.sample(frag_uv) # [1, H, W, 3]
        frag_roughness = frag_ks[..., 1:2] ** 2
        frag_metallic = frag_ks[..., 2:3]

        diffuse_num = self.num_samples // 2
        specular_num = self.num_samples - diffuse_num
        diffuse_directions = self._sample_diffuse_directions(frag_n) # [P, H, W, 3]
        specular_directions = self._sample_specular_directions(frag_inv_wi, frag_roughness) # [Q, H, W, 3]

        # diffuse sample prob
        NoL_d = self._saturate_dot(diffuse_directions, frag_n) # [P, H, W, 1]
        diffuse_probability = NoL_d / torch.pi * (diffuse_num / self.num_samples) # [P, H, W, 1]

        # specualr sample prob
        H_s = safe_normalize(frag_wo + specular_directions) # [Q, H, W, 3]
        NoH_s = self._saturate_dot(frag_n, H_s) # [Q, H, W, 1]
        VoH_s = self._saturate_dot(frag_wo, H_s) # [Q, H, W, 1]
        specular_probability = ( # D * NoH / (4 * VoH)
            self._distribution_ggx(NoH_s, frag_roughness) * NoH_s / (4 * VoH_s + 1e-5)
            * (specular_num / self.num_samples)
        ) # [Q, H, W, 1]

        directions = torch.cat((diffuse_directions, specular_directions)) # [P+Q, H, W, 3]
        probability = torch.cat((diffuse_probability, specular_probability)) # [P+Q, H, W, 1]
        lights = self.envmap.sample(directions) # [P+Q, H, W, 3]

        F0 = 0.04 * (1 - frag_metallic) + frag_metallic * frag_albedo # [1, H, W, 3]
        fresnel, H, HoV = self._fresnel_schlick_directions(F0, frag_wo, directions)
        NoV = self._saturate_dot(frag_n, frag_wo) # [1, H, W, 1]
        NoL = self._saturate_dot(frag_n, directions) # [P+Q, H, W, 1]
        geometry = self._geometry(NoV, NoL, frag_roughness) # [P+Q, H, W, 1]
        NoH = self._saturate_dot(frag_n, H)
        specular_weights = fresnel * self._distribution_ggx(NoH, frag_roughness) * geometry / (4 * NoV * probability + 1e-5)
        specular_colors = (lights * specular_weights).mean(0, keepdim=True)
        diffuse_colors = (frag_albedo * (1 - frag_metallic) * lights[:diffuse_num].mean(0, keepdim=True))

        # positions, normals, depths = raytracer.trace(rays.origins, rays.directions, inplace=False)
        # hit = depths.unsqueeze(-1) < 10

        return diffuse_colors + specular_colors


@dataclass
class DiffusePBRShader(BaseShader[PBRAImages]):

    normal_type: Literal['flat', 'face', 'vertex'] = 'vertex'
    bend_back_normal: bool = False
    envmap: TextureCubeMap = ...
    num_samples: int = 1024

    def get_image_class(self) -> Type[PBRAImages]:
        return PBRAImages

    def _sample_diffuse_directions(self, normals):
        """
        Sample diffuse directions using cosine-weighted hemisphere sampling.
        """
        z = normals  # [1, H, W, 3]
        x = get_arbitrary_tangents_from_normals(z)  # [1, H, W, 3]
        y = torch.cross(z, x, dim=-1)  # [1, H, W, 3]

        # Generate random samples on the hemisphere using cosine-weighted sampling
        u1 = torch.rand(self.num_samples, 1, 1, 1, device=normals.device)
        u2 = torch.rand(self.num_samples, 1, 1, 1, device=normals.device)

        # Convert random samples to spherical coordinates
        theta = torch.acos(torch.sqrt(1 - u1))  # Elevation angle (weighted by cosine)
        phi = 2 * torch.pi * u2  # Azimuthal angle

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # Convert spherical coordinates to Cartesian coordinates
        coeff_x = sin_theta * cos_phi
        coeff_y = sin_theta * sin_phi
        coeff_z = cos_theta

        # Combine the tangent space basis vectors
        directions = coeff_x * x + coeff_y * y + coeff_z * z  # [num_samples, H, W, 3]
        return directions

    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W 3"]:
        """
        Perform only-diffuse PBR shading.
        """
        frag_n = context.normals(self.normal_type, bend_backward=self.bend_back_normal)  # [1, H, W, 3]
        frag_uv = context.facial_vertex_attribute_map(context.mesh.uvs)  # [1, H, W, 2]
        frag_albedo = context.mesh.kd.sample(frag_uv)  # [1, H, W, 3]

        # Sample diffuse directions
        diffuse_directions = self._sample_diffuse_directions(frag_n)  # [num_samples, H, W, 3]

        # Sample the environment map
        lights = self.envmap.sample(diffuse_directions)  # [num_samples, H, W, 3]

        # Combine diffuse contributions
        diffuse_colors = (frag_albedo * lights).mean(0, keepdim=True)  # [1, H, W, 3]

        return diffuse_colors
