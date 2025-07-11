from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
from jaxtyping import Float32
from kornia.filters import spatial_gradient
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    FlexiCubes,
    PBRAImages,
    RGBAImages,
    RGBImages,
    TextureCubeMap,
    TextureLatLng,
    TriangleMesh,
    VectorImages,
)
from rfstudio.graphics._mesh._optix import OptiXContext, optix_build_bvh
from rfstudio.graphics.shaders import NormalShader, PrettyShader, ShadowShader
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains

from .components.encoding import HashEncoding
from .geosplat import GaussianField, RenderableAttrs
from .gsplat import GSplatter


@dataclass
class GeoSplatterMC(Module):

    load: Path = ...

    background_color: Literal["random", "black", "white"] = "random"

    smooth_type: Literal['jitter', 'grad', 'tv'] = 'jitter'

    field: GaussianField = GaussianField(
        occ_enc=HashEncoding(
            mlp=MLP(
                layers=[-1, 32, 32, 6],
                activation='none',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            max_res=4096,
            log2_hashmap_size=18,
        )
    )

    def __setup__(self) -> None:
        state_dict = torch.load(self.load, map_location='cpu')
        self.initial_guess: Tensor = torch.nn.Parameter(state_dict['initial_guess'], requires_grad=False)
        self.scale = state_dict['geom_scale']
        self.shadow_scale = 1.0
        self.resolution = state_dict['resolution']
        self.min_roughness = state_dict['min_roughness']
        self.max_metallic = state_dict['max_metallic']
        self.geometric_repr = FlexiCubes.from_resolution(
            self.resolution,
            scale=self.scale,
        )
        self.deform_params = torch.nn.Parameter(state_dict['deforms'])
        self.sdf_params = torch.nn.Parameter(state_dict['sdfs'])
        self.weight_params = torch.nn.Parameter(state_dict['weights'])
        self.latlng = torch.nn.Parameter(TextureCubeMap(data=state_dict['cubemap']).as_latlng(width=512, height=256).data)
        self.exposure_params = torch.nn.Parameter(state_dict['exposure'])
        self.field.ks_enc.load_state_dict(state_dict['ks_enc'])
        self.envmap = None
        self.sdf_weight = 0.0
        self.occ_weight = 0.0
        self.light_weight = 0.0
        self.kd_grad_weight = 0.0
        self.kd_regualr_perturb_std = 0.0
        self.ks_grad_weight = 0.0
        self.ks_regualr_perturb_std = 0.0
        self.normal_grad_weight = 0.0
        self.optix_ctx = None

    def get_geometry(self) -> Tuple[TriangleMesh, Tensor]:
        # TODO: fix the device
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))
        vertices = self.geometric_repr.vertices + self.deform_params.tanh() * (0.5 * self.scale / self.resolution)
        flexicubes = self.geometric_repr.replace(
            vertices=vertices,
            sdf_values=self.sdf_params,
            alpha=self.weight_params[:, :8],
            beta=self.weight_params[:, 8:20],
            gamma=self.weight_params[:, 20:],
        )
        mesh, L_dev = flexicubes.dual_marching_cubes()
        reg_loss = torch.add(
            L_dev.mean() * 0.5 + self.weight_params[:, :20].abs().mean() * 0.1,
            flexicubes.compute_entropy() * self.sdf_weight
        )
        return mesh, reg_loss

    def get_background_color(self) -> Float32[Tensor, "3 "]:
        if self.background_color == 'black':
            return torch.zeros(3)
        if self.background_color == 'white':
            return torch.ones(3)
        if self.training:
            return torch.rand(3)
        return torch.tensor([0.1490, 0.1647, 0.2157])

    def get_envmap(self) -> TextureLatLng:
        return TextureLatLng(data=self.latlng).compute_pdf_()

    def get_gsplat(self) -> Tuple[TriangleMesh, GSplatter, RenderableAttrs, Tensor, Tensor]:
        mesh, reg = self.get_geometry()
        gsplat = GSplatter(
            num_random=4,
            sh_degree=0,
            background_color=self.background_color,
            rasterize_mode='antialiased',
        ).to(self.device)
        gsplat.__setup__()
        kd_regualr_perturb_std = self.kd_regualr_perturb_std if self.smooth_type == 'jitter' else 0
        ks_regualr_perturb_std = self.ks_regualr_perturb_std if self.smooth_type == 'jitter' else 0
        splats, attrs, offsets = self.field.get_gaussians_from_face(
            mesh=mesh,
            scale=self.scale,
            kd_perturb_std=kd_regualr_perturb_std,
            ks_perturb_std=ks_regualr_perturb_std,
            use_checkpoint=False,
            initial_guess=self.initial_guess,
        )
        gsplat.gaussians = splats
        if kd_regualr_perturb_std > 0 and self.kd_grad_weight > 0:
            reg = reg + self.kd_grad_weight * (attrs.kd_jitter - attrs.kd).abs().mean()
        if ks_regualr_perturb_std > 0 and self.ks_grad_weight > 0:
            reg = reg + self.ks_grad_weight * (attrs.ks_jitter - attrs.ks).abs().mean()
        if self.occ_weight > 0:
            reg = reg + self.occ_weight * attrs.occ.abs().mean()
        return mesh, gsplat, attrs.replace(kd_jitter=None, ks_jitter=None), reg, offsets

    @torch.no_grad()
    def export_model(self, path: Path) -> None:
        mesh, gsplat, attrs, _, offsets = self.get_gsplat()
        attributes = {
            'geom_scale': self.scale,
            'resolution': self.resolution,
            'min_roughness': self.min_roughness, # [1]
            'max_metallic': self.max_metallic, # [1]
            'exposure': self.exposure_params, # [1]
            'latlng': self.latlng,
            'means': gsplat.gaussians.means,
            'scales': gsplat.gaussians.scales,
            'quats': gsplat.gaussians.quats,
            'opacities': gsplat.gaussians.opacities,
            'normals': attrs.normals,
            'kd': attrs.kd,
            'ks': attrs.ks,
            'occ': attrs.occ,
            'ks_enc': self.field.ks_enc.state_dict(),
            'occ_enc': self.field.occ_enc.state_dict(),
            'mc_positions': gsplat.gaussians.means + offsets,
            'mc_vertices': mesh.vertices,
            'mc_indices': mesh.indices.int(),
            'initial_guess': self.initial_guess,
        }
        torch.save(attributes, path)

    def render_report(self, inputs: Cameras, *, indices: Optional[Tensor], gt_outputs: RGBAImages) -> Tuple[
        PBRAImages,
        RGBImages,
        VectorImages,
        int,
        Tensor,
    ]:
        images = []
        mesh, gsplat, attrs, regularization, offsets = self.get_gsplat()

        with torch.no_grad():
            if self.optix_ctx is None:
                self.optix_ctx = OptiXContext()
            optix_build_bvh(self.optix_ctx, mesh.vertices.contiguous(), mesh.indices.int(), rebuild=1)
            mc_positions = gsplat.gaussians.means + offsets
            gt_pbra = gt_outputs.srgb2rgb()
        envmap = self.get_envmap()
        batch_size = len(gt_outputs)
        for i in range(batch_size):
            img, reg = attrs.splat_mc(
                gsplat,
                inputs[i:i+1],
                envmap=envmap,
                gt_image=gt_pbra[i].item(),
                context=self.optix_ctx,
                positions=mc_positions,
                exposure=self.exposure_params.exp(),
                num_samples_per_ray=8,
                denoise=True,
                residual=True,
                min_roughness=self.min_roughness,
                max_metallic=self.max_metallic,
                shadow_scale=self.shadow_scale,
            )
            images.append(img)
            regularization = regularization + reg / batch_size

        if self.smooth_type == 'grad':
            for camera, gt_rgb in zip(inputs.view(-1, 1), gt_outputs.blend(self.get_background_color())):
                if self.kd_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=attrs.kd)
                    first_order_edge_aware_loss = torch.mul(
                        spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                        (-spatial_gradient(gt_rgb[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                    ).sum(1).mean()
                    regularization = regularization + first_order_edge_aware_loss * self.kd_grad_weight / batch_size
                if self.ks_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=torch.cat((torch.zeros_like(attrs.ks[..., :1]), attrs.ks), dim=-1))
                    first_order_edge_aware_loss = torch.mul(
                        spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                        (-spatial_gradient(gt_rgb[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                    ).sum(1).mean()
                    regularization = regularization + first_order_edge_aware_loss * self.ks_grad_weight / batch_size
        if self.normal_grad_weight > 0:
            for camera, gt_rgb in zip(inputs.view(-1, 1), gt_outputs.blend(self.get_background_color())):
                gsplat.gaussians.replace_(colors=attrs.normals * 0.5 + 0.5)
                first_order_edge_aware_loss = torch.mul(
                    spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                    (-spatial_gradient(gt_rgb[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                ).sum(1).mean()
                regularization = regularization + first_order_edge_aware_loss * self.normal_grad_weight / batch_size
        if self.smooth_type == 'tv':
            for camera in inputs.view(-1, 1):
                if self.kd_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=attrs.kd)
                    rendered = gsplat.render_rgb(camera).item()
                    tv_loss = torch.add(
                        (rendered[1:, :] - rendered[:-1, :]).square().mean(),
                        (rendered[:, 1:] - rendered[:, :-1]).square().mean(),
                    )
                    regularization = regularization + tv_loss * self.kd_grad_weight / batch_size
                if self.ks_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=torch.cat((torch.zeros_like(attrs.ks[..., :1]), attrs.ks), dim=-1))
                    rendered = gsplat.render_rgb(camera).item()
                    tv_loss = torch.add(
                        (rendered[1:, :] - rendered[:-1, :]).square().mean(),
                        (rendered[:, 1:] - rendered[:, :-1]).square().mean(),
                    )
                    regularization = regularization + tv_loss * self.ks_grad_weight / batch_size
        with torch.no_grad():
            camera = inputs.view(-1, 1)[0]
            gsplat.gaussians.replace_(colors=attrs.kd)
            kd = gsplat.render_rgb(camera).item()
            kd = torch.where(
                kd <= 0.0031308,
                kd * 12.92,
                torch.clamp(kd, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
            )
            gsplat.gaussians.replace_(colors=torch.cat((
                torch.zeros_like(attrs.ks[..., :1]),
                attrs.ks[..., :1] * (1 - self.min_roughness) + self.min_roughness,
                attrs.ks[..., 1:] * self.max_metallic,
            ), dim=-1))
            ks = gsplat.render_rgb(camera).item()
            gsplat.gaussians.replace_(colors=attrs.normals * 0.5 + 0.5)
            normal_map = gsplat.render_rgb(camera).item()
            gsplat.gaussians.replace_(colors=(attrs.occ[..., :3] - 3).sigmoid())
            occ_diff = gsplat.render_rgb(camera).item()
            gsplat.gaussians.replace_(colors=(attrs.occ[..., 3:] - 3).sigmoid())
            occ_spec = gsplat.render_rgb(camera).item()
            cubemap = TextureLatLng(data=self.latlng).as_cubemap(resolution=512)
            light = cubemap.visualize(
                width=kd.shape[1] * 2,
                height=kd.shape[0],
            ).item()
            mesh_normal = mesh.render(camera, shader=NormalShader())
            pretty_mesh = mesh.render(
                camera,
                shader=PrettyShader(occlusion_type='none'),
            ).rgb2srgb().blend(self.get_background_color()).item()
        return (
            PBRAImages(images),
            RGBImages([kd, ks, occ_diff, occ_spec, normal_map, light, pretty_mesh]),
            mesh_normal,
            gsplat.gaussians.shape[0],
            regularization
        )

    @torch.no_grad()
    def render_light_transport(self, inputs: Cameras) -> RGBImages:
        assert inputs.shape == (1, )
        mesh, gsplat, attrs, _, offsets = self.get_gsplat()
        envmap = self.get_envmap()
        if self.optix_ctx is None:
            self.optix_ctx = OptiXContext()
            optix_build_bvh(self.optix_ctx, mesh.vertices.contiguous(), mesh.indices.int(), rebuild=1)
        mc_positions = gsplat.gaussians.means + offsets

        dir_pbr, _ = attrs.splat_mc(
            gsplat,
            inputs,
            envmap=envmap,
            gt_image=None,
            context=self.optix_ctx,
            positions=mc_positions,
            exposure=self.exposure_params.exp(),
            num_samples_per_ray=16,
            denoise=True,
            residual=False,
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=self.shadow_scale,
        )
        dir_light, _ = attrs.splat_mc(
            gsplat,
            inputs,
            envmap=envmap,
            gt_image=None,
            context=self.optix_ctx,
            positions=mc_positions,
            exposure=self.exposure_params.exp(),
            num_samples_per_ray=16,
            denoise=True,
            residual=False,
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=0,
        )
        full_pbr, _ = attrs.splat_mc(
            gsplat,
            inputs,
            envmap=envmap,
            gt_image=None,
            context=self.optix_ctx,
            positions=mc_positions,
            exposure=self.exposure_params.exp(),
            num_samples_per_ray=16,
            denoise=True,
            residual=True,
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=self.shadow_scale,
        )
        ind_light = PBRAImages([
            torch.cat(((full_pbr[..., :3] - dir_pbr[..., :3]).clamp(0, 1), full_pbr[..., 3:]), dim=-1)
        ]).rgb2srgb().blend((1, 1, 1)).item()
        attrs.kd.fill_(1.0)
        attrs.ks[..., 0].fill_(0.25)
        gs_shadow, _ = attrs.splat_mc(
            gsplat,
            inputs,
            envmap=envmap.replace(data=torch.ones_like(envmap.data)),
            gt_image=None,
            context=self.optix_ctx,
            positions=mc_positions,
            exposure=torch.ones_like(self.exposure_params),
            num_samples_per_ray=16,
            denoise=True,
            residual=False,
            mode='diffuse',
            min_roughness=self.min_roughness,
            max_metallic=self.max_metallic,
            shadow_scale=self.shadow_scale,
        )
        gs_shadow = torch.cat(
            (gs_shadow[..., :3].max(-1, keepdim=True).values.repeat(1, 1, 3) * 0.92,
            gs_shadow[..., 3:],
        ), dim=-1)
        gs_shadow = PBRAImages([gs_shadow]).rgb2srgb().blend((1, 1, 1)).item()
        shadow_shader = ShadowShader(roughness=0.25, envmap=envmap.replace(data=torch.ones_like(envmap.data)))
        mesh_shadow = mesh.render(inputs[0], shader=shadow_shader).item()
        mesh_shadow[..., :3] *= 0.92
        mesh_shadow = PBRAImages([mesh_shadow]).rgb2srgb().blend((1, 1, 1)).item()
        full_pbr = PBRAImages([full_pbr]).rgb2srgb().blend((1, 1, 1)).item()
        dir_light = PBRAImages([dir_light]).rgb2srgb().blend((1, 1, 1)).item()
        return RGBImages([mesh_shadow, gs_shadow, full_pbr, dir_light, ind_light])

    @chains
    def as_module(self, *, field_name: Literal['deforms', 'sdfs', 'weights', 'light', 'exposure']) -> Any:

        def parameters(self) -> Any:
            params = {
                'deforms': self.deform_params,
                'sdfs': self.sdf_params,
                'weights': self.weight_params,
                'light': self.latlng,
                'exposure': self.exposure_params,
            }[field_name]
            return [params]

        return parameters
