from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float
from kornia.filters import spatial_gradient
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    DepthImages,
    PBRAImages,
    RGBImages,
    TextureCubeMap,
    TextureLatLng,
    TriangleMesh,
)
from rfstudio.graphics._mesh._optix import OptiXContext, optix_build_bvh
from rfstudio.graphics.math import safe_normalize
from rfstudio.graphics.shaders import ShadowShader
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains

from .components.encoding import HashEncoding
from .geosplat import RenderableAttrs
from .gsplat import GSplatter


@dataclass
class GeoSplatterDefer(Module):

    load: Path = ...

    background_color: Literal["random", "black", "white"] = "random"

    gt_envmap: Optional[Path] = None

    ks_enc: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 2],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        max_res=4096,
        log2_hashmap_size=18,
    )

    def __setup__(self) -> None:
        state_dict = torch.load(self.load, map_location='cpu')
        self.exposure = torch.nn.Parameter(state_dict['exposure'])
        self.means = torch.nn.Parameter(state_dict['means']) # [N, 3]
        self.scales = torch.nn.Parameter(state_dict['scales']) # [N, 3]
        self.quats = torch.nn.Parameter(state_dict['quats']) # [N, 4]
        self.opacities = torch.nn.Parameter(state_dict['opacities']) # [N, 1]
        self.normals = torch.nn.Parameter(state_dict['normals']) # [N, 1]
        if 'vertices' in state_dict.keys():
            self.mesh_pos = torch.nn.Parameter(state_dict['means'], requires_grad=False) # [N, 3]
            self.mesh_v = torch.nn.Parameter(state_dict['vertices'], requires_grad=False)
            self.mesh_i = torch.nn.Parameter(state_dict['indices'], requires_grad=False)
            self.kd_params = torch.nn.Parameter(state_dict['kd'])
        else: # from s2
            self.mesh_pos = torch.nn.Parameter(state_dict['mc_positions'], requires_grad=False) # [N, 3]
            self.mesh_v = torch.nn.Parameter(state_dict['mc_vertices'], requires_grad=False)
            self.mesh_i = torch.nn.Parameter(state_dict['mc_indices'], requires_grad=False)
            self.kd_params = torch.nn.Parameter(state_dict['kd'])
        self.occ_params = torch.nn.Parameter(state_dict['occ'])
        self.ks_enc.load_state_dict(state_dict['ks_enc'])
        self.min_roughness = state_dict['min_roughness']
        self.max_metallic = state_dict.get('max_metallic', 1)
        self.initial_guess: Tensor = torch.nn.Parameter(state_dict['initial_guess'], requires_grad=False)
        self.scale = state_dict['geom_scale']
        if self.gt_envmap is None:
            latlng = state_dict['latlng']
            # latlng =  TextureLatLng.from_image_file(Path('data') / 'tensoir' / 'sunset.hdr').data
            self.latlng_hue = torch.nn.Parameter(latlng / (latlng + 1))
            self.latlng_value = torch.nn.Parameter((latlng + 1.00001).log())
        else:
            self.latlng_hue = torch.nn.Parameter(torch.empty(0))
            self.latlng_value = torch.nn.Parameter(torch.empty(0))

        self.envmap = None
        self.kd_weight = 0.0
        self.ks_weight = 0.0
        self.normal_weight = 0.0
        self.shadow_scale = 1.0
        self.albedo_scaling = None
        self.optix_ctx = None

    def get_background_color(self) -> Float[Tensor, "3 "]:
        if self.background_color == 'black':
            return torch.zeros(3)
        if self.background_color == 'white':
            return torch.ones(3)
        if self.training:
            return torch.rand(3)
        return torch.tensor([0.1490, 0.1647, 0.2157])

    def set_relight_envmap(self, envmap_path: Path, *, albedo_scaling: Float[Tensor, "3"]) -> None:
        self.gt_envmap = envmap_path
        self.envmap = TextureLatLng.from_image_file(envmap_path, device=self.device).compute_pdf_()
        assert albedo_scaling.shape == (3,)
        self.albedo_scaling = albedo_scaling.to(self.device)

    def get_envmap(self) -> TextureLatLng:
        if self.gt_envmap is None:
            return TextureLatLng(data=self.latlng_hue * self.latlng_value.exp()).compute_pdf_()
        if self.envmap is None:
            self.envmap = (
                TextureCubeMap
                    .from_image_file(self.gt_envmap, device=self.device)
                    .as_latlng(width=512, height=256)
                    .compute_pdf_()
            )
        return self.envmap

    def get_gsplat(self) -> GSplatter:
        gsplat = GSplatter(
            num_random=4,
            sh_degree=0,
            background_color=self.background_color,
            rasterize_mode='antialiased',
        ).to(self.device)
        gsplat.__setup__()
        gsplat.gaussians.replace_(
            means=self.means,
            scales=self.scales,
            opacities=self.opacities,
            quats=self.quats,
            shs=self.normals.new_empty(self.normals.shape[0], 0, 3),
            colors=safe_normalize(self.normals),
        )
        return gsplat

    def render_depth(self, inputs: Cameras) -> DepthImages:
        gsplat = self.get_gsplat()
        normals = gsplat.gaussians.colors
        original = gsplat.gaussians
        images = []
        for camera in inputs.view(-1, 1):
            visibilities = ((camera.c2w[:, :3, 3] - gsplat.gaussians.means) * normals).sum(-1) > 0
            gsplat.gaussians = gsplat.gaussians[visibilities]
            images.append(gsplat.render_depth(camera).item())
            gsplat.gaussians = original
        return DepthImages(images)

    def render_report(self, inputs: Cameras, *, indices: Optional[Tensor], gt_images: Optional[Tensor]) -> Tuple[
        PBRAImages,
        RGBImages,
        RGBImages,
        int,
        Tensor,
    ]:
        images = []
        gsplat = self.get_gsplat()
        normals = gsplat.gaussians.colors
        envmap = self.get_envmap()
        kd = self.kd_params
        ks = (self.ks_enc((gsplat.gaussians.means / self.scale).clamp(-1, 1)) + self.initial_guess).sigmoid()
        ks_jitter = (self.ks_enc(
            ((gsplat.gaussians.means + torch.randn_like(gsplat.gaussians.means) * 0.01) / self.scale).clamp(-1, 1)
        ) + self.initial_guess).sigmoid()
        occ = self.occ_params
        if self.albedo_scaling is not None:
            occ = self.occ_params.mean(-1, keepdim=True) * torch.cat((torch.ones_like(kd), kd), dim=-1)

        ks_reg = (ks - ks_jitter).abs().mean() * self.ks_weight

        if self.albedo_scaling is not None:
            kd = kd * self.albedo_scaling

        if self.optix_ctx is None:
            with torch.no_grad():
                self.optix_ctx = OptiXContext()
                assert envmap.transform is None
                optix_build_bvh(self.optix_ctx, self.mesh_v.contiguous(), self.mesh_i, rebuild=1)

        attrs = RenderableAttrs(
            kd=kd,
            ks=ks,
            occ=occ,
            normals=normals,
        )
        regs = []
        if gt_images is None:
            gt_images = [None] * inputs.view(-1, 1).shape[0]

        V = self.mesh_v.shape[0]
        ctx = dr.RasterizeCudaContext(self.mesh_v.device)
        vertices = torch.cat((
            self.mesh_v,
            torch.ones_like(self.mesh_v[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        images = []

        for camera, gt_image in zip(inputs.view(-1, 1), gt_images, strict=True):
            camera_item = camera[0]
            resolution = [camera_item.height.item(), camera_item.width.item()]
            mvp = camera_item.projection_matrix @ camera_item.view_matrix # [4, 4]
            projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]
            with dr.DepthPeeler(ctx, projected, self.mesh_i, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
                frag_pos, _ = dr.interpolate(self.mesh_v, rast, self.mesh_i)
            image, reg = attrs.splat_mc_deferred(
                gsplat,
                camera,
                gt_image=gt_image,
                context=self.optix_ctx,
                envmap=envmap,
                positions=frag_pos,
                mask=(rast[..., -1:] > 0).float(),
                exposure=self.exposure.exp() if self.albedo_scaling is None else 1,
                min_roughness=self.min_roughness,
                max_metallic=self.max_metallic,
                shadow_scale=self.shadow_scale,
                num_samples_per_ray=16,
                # denoise=True,
                tone_type='naive',
                residual=True,
            )
            images.append(image)
            if gt_image is not None:
                gt_image = gt_image[..., :3] * gt_image[..., 3:] + (1 - gt_image[..., 3:])
                gsplat.gaussians.replace_(colors=kd)
                first_order_edge_aware_loss = torch.mul(
                    spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                    (-spatial_gradient(gt_image[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                ).sum(1).mean()
                reg = reg + first_order_edge_aware_loss * self.kd_weight
                gsplat.gaussians.replace_(colors=attrs.normals * 0.5 + 0.5)
                first_order_edge_aware_loss = torch.mul(
                    spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                    (-spatial_gradient(gt_image[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                ).sum(1).mean()
                reg = reg + first_order_edge_aware_loss * self.normal_weight
            regs.append(reg)
        with torch.no_grad():
            camera = inputs.view(-1, 1)[0]
            camera_lookat = -camera.c2w[:, :, 2] # [3]
            bend = ((self.normals.detach() * camera_lookat).sum(-1, keepdim=True) > 0)
            opacities = torch.where(bend, -2, gsplat.gaussians.opacities)
            gsplat.gaussians.replace_(opacities=opacities)

            gsplat.gaussians = gsplat.gaussians.replace(colors=(attrs.occ[..., :3] - 3).sigmoid())
            direct = gsplat.render_rgb(camera).item()
            num_gaussians = gsplat.gaussians.shape[0]
            gsplat.gaussians = gsplat.gaussians.replace(colors=kd)
            kd_map = gsplat.render_rgb(camera).item()
            kd_map = torch.where(
                kd_map <= 0.0031308,
                kd_map * 12.92,
                torch.clamp(kd_map, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
            )
            modulated_ks = torch.cat((torch.zeros_like(attrs.ks[..., :1]), attrs.ks), dim=-1)
            modulated_ks[..., 1] = modulated_ks[..., 1] * (1 - self.min_roughness) + self.min_roughness
            modulated_ks[..., 2] = modulated_ks[..., 2] * self.max_metallic
            gsplat.gaussians.replace_(colors=modulated_ks)
            ks_map = gsplat.render_rgb(camera).item()
            normal_map = (
                gsplat.render_depth(camera)
                    .compute_pseudo_normals(camera[0])
                    .visualize(self.get_background_color())
                    .item()
            )
            if self.gt_envmap is None:
                light = TextureLatLng(
                    data=self.latlng_hue * self.latlng_value.exp(),
                    transform=None,
                ).as_cubemap(resolution=512).visualize(
                    width=kd_map.shape[1],
                    height=kd_map.shape[0],
                ).item()
            else:
                light = self.envmap.as_cubemap(resolution=512).visualize(
                    width=kd_map.shape[1],
                    height=kd_map.shape[0],
                ).item()
            normals = torch.where(bend, -self.normals, self.normals)
            gsplat.gaussians.replace_(colors=normals)
            img = gsplat.render_rgba(camera).item()
            mesh_normal = RGBImages([(safe_normalize(img[..., :3]) * 0.5 + 0.5) + (1 - img[..., 3:])])
        return (
            PBRAImages(images),
            RGBImages([kd_map, ks_map, normal_map, light, direct]),
            mesh_normal,
            num_gaussians,
            ks_reg + sum(regs) / len(regs),
        )

    @torch.no_grad()
    def render_light_transport(self, inputs: Cameras) -> RGBImages:
        gsplat = self.get_gsplat()
        normals = gsplat.gaussians.colors
        envmap = self.get_envmap()
        kd = self.kd_params
        ks = (self.ks_enc((gsplat.gaussians.means / self.scale).clamp(-1, 1)) + self.initial_guess).sigmoid()
        occ = self.occ_params

        if self.optix_ctx is None:
            self.optix_ctx = OptiXContext()
            assert envmap.transform is None
            optix_build_bvh(self.optix_ctx, self.mesh_v.contiguous(), self.mesh_i, rebuild=1)

        attrs = RenderableAttrs(
            kd=kd,
            ks=ks,
            occ=occ,
            normals=normals,
        )

        V = self.mesh_v.shape[0]
        ctx = dr.RasterizeCudaContext(self.mesh_v.device)
        vertices = torch.cat((
            self.mesh_v,
            torch.ones_like(self.mesh_v[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        assert inputs.shape == (1, )
        camera_item = inputs[0]
        resolution = [camera_item.height.item(), camera_item.width.item()]
        mvp = camera_item.projection_matrix @ camera_item.view_matrix # [4, 4]
        projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]
        with dr.DepthPeeler(ctx, projected, self.mesh_i, resolution=resolution) as peeler:
            rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
            frag_pos, _ = dr.interpolate(self.mesh_v, rast, self.mesh_i)

        dir_pbr, _ = attrs.splat_mc_deferred(
            gsplat,
            inputs,
            gt_image=None,
            context=self.optix_ctx,
            envmap=envmap,
            positions=frag_pos,
            mask=(rast[..., -1:] > 0).float(),
            exposure=self.exposure.exp() if self.albedo_scaling is None else 1,
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=1,
            num_samples_per_ray=16,
            tone_type='naive',
            residual=False,
        )
        dir_light, _ = attrs.splat_mc_deferred(
            gsplat,
            inputs,
            gt_image=None,
            context=self.optix_ctx,
            envmap=envmap,
            positions=frag_pos,
            mask=(rast[..., -1:] > 0).float(),
            exposure=self.exposure.exp() if self.albedo_scaling is None else 1,
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=0,
            num_samples_per_ray=16,
            tone_type='naive',
            residual=False,
        )
        full_pbr, _ = attrs.splat_mc_deferred(
            gsplat,
            inputs,
            gt_image=None,
            context=self.optix_ctx,
            envmap=envmap,
            positions=frag_pos,
            mask=(rast[..., -1:] > 0).float(),
            exposure=self.exposure.exp() if self.albedo_scaling is None else 1,
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=1,
            num_samples_per_ray=16,
            tone_type='naive',
            residual=True,
        )
        ind_light = PBRAImages([
            torch.cat(((full_pbr[..., :3] - dir_pbr[..., :3]).clamp(0, 1), full_pbr[..., 3:]), dim=-1)
        ]).rgb2srgb().blend((1, 1, 1)).item()
        attrs.kd.fill_(1.0)
        attrs.ks[..., 0].fill_(0.25)
        gs_shadow, _ = attrs.splat_mc_deferred(
            gsplat,
            inputs,
            gt_image=None,
            context=self.optix_ctx,
            envmap=envmap.replace(data=torch.ones_like(envmap.data)).compute_pdf_(),
            positions=frag_pos,
            mask=(rast[..., -1:] > 0).float(),
            exposure=self.exposure.exp() if self.albedo_scaling is None else 1,
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=1,
            mode='diffuse',
            num_samples_per_ray=16,
            tone_type='naive',
            residual=False,
        )
        gs_shadow = torch.cat(
            (gs_shadow[..., :3].max(-1, keepdim=True).values.repeat(1, 1, 3) * 0.96,
            gs_shadow[..., 3:],
        ), dim=-1)
        gs_shadow = PBRAImages([gs_shadow]).rgb2srgb().blend((1, 1, 1)).item()
        mesh = TriangleMesh(vertices=self.mesh_v, indices=self.mesh_i.long())
        shadow_shader = ShadowShader(roughness=0.25, envmap=envmap.replace(data=torch.ones_like(envmap.data)).compute_pdf_())
        mesh_shadow = mesh.render(camera_item, shader=shadow_shader).item()
        mesh_shadow[..., :3] *= 0.96
        mesh_shadow = PBRAImages([mesh_shadow]).rgb2srgb().blend((1, 1, 1)).item()
        full_pbr = PBRAImages([full_pbr]).rgb2srgb().blend((1, 1, 1)).item()
        dir_light = PBRAImages([dir_light]).rgb2srgb().blend((1, 1, 1)).item()
        return RGBImages([mesh_shadow, gs_shadow, full_pbr, dir_light, ind_light])

    @chains
    def as_module(
        self,
        *,
        field_name: Literal[
            'means',
            'scales',
            'quats',
            'opacities',
            'exposure',
            'normals',
            'light_hue',
            'light_value',
            'kd',
            'occ',
        ],
    ) -> Any:

        def parameters(self) -> Any:
            params = {
                'means': self.means,
                'scales': self.scales,
                'quats': self.quats,
                'opacities': self.opacities,
                'exposure': self.exposure,
                'normals': self.normals,
                'light_hue': self.latlng_hue,
                'light_value': self.latlng_value,
                'kd': self.kd_params,
                'occ': self.occ_params,
            }[field_name]
            return [params]

        return parameters
