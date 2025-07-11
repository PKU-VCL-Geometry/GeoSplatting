from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, Optional, Tuple, Type, TypeVar

import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from .._cameras import Cameras
from .._images import BaseImages, DepthImages
from .._points import Points
from ..math import get_uniform_normal_from_hemisphere, safe_normalize
from ._dpsr import diff_poisson_surface_recon
from ._obj import _load_obj, _merge_materials
from ._texture import Texture2D

IMG = TypeVar('IMG', bound=BaseImages)
T = TypeVar('T', bound='BaseShader')


class ShadingContext:

    def __init__(
        self,
        mesh: TriangleMesh,
    ) -> None:
        self._mesh = mesh
        self._flat_normals = None
        self._face_normals = None
        self._vertex_normals = None
        self._ao = None
        self._vertex_indices = mesh.indices.int()
        self._face_indices = torch.arange(mesh.num_faces * 3, device=mesh.device, dtype=torch.int32).view(-1, 3)
        self._clear()

    def _clear(self) -> None:
        self._rast = None
        self._visibilities = None
        self._camera = None
        self._vertex_normal_map = None
        self._face_normal_map = None
        self._flat_normal_map = None
        self._gpos_map = None
        self._view_dir_map = None
        self._facial_indices = None
        self._ao_map = None

    def clone(self) -> ShadingContext:

        dummy = ShadingContext(self._mesh)

        dummy._flat_normals = self._flat_normals
        dummy._face_normals = self._face_normals
        dummy._vertex_normals = self._vertex_normals
        dummy._ao = self._ao
        dummy._vertex_indices = self._vertex_indices
        dummy._face_indices = self._face_indices

        dummy._rast = self._rast
        dummy._visibilities = self._visibilities
        dummy._camera = self._camera
        dummy._vertex_normal_map = self._vertex_normal_map
        dummy._face_normal_map = self._face_normal_map
        dummy._flat_normal_map = self._flat_normal_map
        dummy._gpos_map = self._gpos_map
        dummy._view_dir_map = self._view_dir_map
        dummy._facial_indices = self._facial_indices
        dummy._ao_map = self._ao_map

        return dummy

    def vertex_attribute_map(self, attributes: Float32[Tensor, "V C"]) -> Float32[Tensor, "1 H W C"]:
        indices = self._vertex_indices
        if self._visibilities is not None:
            indices = indices[self._visibilities, :]
        frags, _ = dr.interpolate(
            attributes[None].contiguous(),
            self._rast,
            indices,
        ) # [1, H, W, C]
        return frags

    def face_attribute_map(self, attributes: Float32[Tensor, "F C"]) -> Float32[Tensor, "1 H W C"]:
        if self._visibilities is not None:
            attributes = attributes[self._visibilities, :]
        frags = attributes[(self._rast[..., -1].flatten().long() - 1).clamp_min(0), :].contiguous()
        return frags.view(*self._rast.shape[:-1], attributes.shape[-1])

    def facial_vertex_attribute_map(self, attributes: Float32[Tensor, "F 3 C"]) -> Float32[Tensor, "1 H W C"]:
        indices = self._face_indices
        if self._visibilities is not None:
            indices = indices[self._visibilities, :]
        frags, _ = dr.interpolate(
            attributes.view(1, -1, attributes.shape[-1]),
            self._rast,
            indices,
        ) # [1, H, W, C]
        return frags

    @property
    def ao(self) -> Float32[Tensor, "1 H W 1"]:
        if self._ao_map is None:
            if self._ao is None:
                self._ao = self.mesh.compute_ambient_occlusion().unsqueeze(-1)
            self._ao_map = self.face_attribute_map(self._ao)
        return self._ao_map

    def ssao(
        self,
        *,
        num_samples: int,
        min_visibility: float,
        sample_radius: float,
    ) -> Float32[Tensor, "1 H W 1"]:
        depths = torch.mul(
            self.camera.c2w[:3, 3] - self.global_positions,
            self.camera.c2w[:3, 2]
        ).sum(-1, keepdim=True) # [1, H, W, 1]
        normals = self.normals(normal_type='vertex', bend_backward=False) # [1, H, W, 3]
        sample_dirs = get_uniform_normal_from_hemisphere(
            num_samples,
            device=normals.device,
            direction=normals[..., None, :], # [1, H, W, 1, 3]
        )
        samples = self.global_positions[..., None, :] + sample_dirs * sample_radius # [1, H, W, S, 3]
        sample_image_space = self.camera.project(samples) # [1, H, W, S, 3]
        sample_depths = sample_image_space[..., 2:3] # [1, H, W, S, 1]
        H, W = sample_depths.shape[1:3]
        depths = depths[..., None, :].expand_as(sample_depths).flatten(1, 2) # [1, H * W, S, 1]
        xs = sample_image_space[..., 0:1].long() # [1, H, W, S, 1]
        ys = sample_image_space[..., 1:2].long() # [1, H, W, S, 1]
        depths = depths.gather(
            dim=1,
            index=torch.add(
                xs.clamp(min=0, max=W - 1), # [1, H, W, S, 1]
                ys.clamp(min=0, max=H - 1) * W, # [1, H, W, S, 1]
            ).flatten(1, 2),
        ).view(1, H, W, -1, 1) # [1, H, W, S, 1]
        valid = (self.alphas[..., None, :].expand_as(depths).flatten(1, 2) > 0).gather(
            dim=1,
            index=torch.add(
                xs.clamp(min=0, max=W - 1), # [1, H, W, S, 1]
                ys.clamp(min=0, max=H - 1) * W, # [1, H, W, S, 1]
            ).flatten(1, 2),
        ).view(1, H, W, -1, 1) # [1, H, W, S, 1]
        valid = valid & (0 <= xs) & (xs < W) & (0 <= ys) & (ys < H) # [1, H, W, S, 1]
        n_dot_d = (sample_dirs * normals[..., None, :]).sum(-1, keepdim=True) * valid.float() # [1, H, W, S, 1]
        return torch.div(
            ((sample_depths < depths).float() * n_dot_d).sum(-2),
            n_dot_d.sum(-2).clamp_min(1e-10),
        ).clamp(min=min_visibility) # [1, H, W, 1]

    @property
    def mesh(self) -> TriangleMesh:
        return self._mesh

    @property
    def camera(self) -> Cameras:
        return self._camera

    @property
    def _culling_info(self) -> Tuple[Float32[Tensor, "F 3"], Float32[Tensor, "F 3"]]:
        if self._flat_normals is not None:
            normals = self._flat_normals
        else:
            normals = self._mesh.replace(face_normals=None).compute_face_normals(fix=True).face_normals[:, 0, :]
            self._flat_normals = normals
        face_centers = (
            self._mesh.vertices[self._mesh.indices[:, 0], :] +
            self._mesh.vertices[self._mesh.indices[:, 1], :] +
            self._mesh.vertices[self._mesh.indices[:, 2], :]
        ) / 3 # [F, 3]
        return normals, face_centers

    def normals(
        self,
        normal_type: Literal['flat', 'face', 'vertex'],
        *,
        bend_backward: bool = False,
    ) -> Float32[Tensor, "1 H W 3"]:
        if normal_type == 'vertex':
            if self._vertex_normal_map is None:
                if self._vertex_normals is not None:
                    normals = self._vertex_normals
                else:
                    normals = (
                        self._mesh.compute_vertex_normals(fix=True).normals
                        if self._mesh.normals is None
                        else self._mesh.normals
                    ) # [V, 3]
                    self._vertex_normals = normals
                self._vertex_normal_map = safe_normalize(self.vertex_attribute_map(normals))
            normal_map = self._vertex_normal_map
        elif normal_type == 'flat':
            if self._flat_normal_map is None:
                if self._flat_normals is not None:
                    normals = self._flat_normals
                else:
                    normals = self._mesh.replace(face_normals=None).compute_face_normals(fix=True).face_normals[:, 0, :]
                    self._flat_normals = normals
                self._flat_normal_map = self.face_attribute_map(normals)
            normal_map = self._flat_normal_map
        elif normal_type == 'face':
            if self._face_normal_map is None:
                if self._face_normals is not None:
                    normals = self._face_normals
                else:
                    normals = (
                        self._mesh.compute_face_normals(fix=True).face_normals
                        if self._mesh.face_normals is None
                        else self._mesh.face_normals
                    ).contiguous() # [F, 3, 3]
                    self._face_normals = normals
                self._face_normal_map = safe_normalize(self.facial_vertex_attribute_map(normals))
            normal_map = self._face_normal_map
        else:
            raise ValueError(normal_type)
        if bend_backward:
            n_dot_v = (normal_map * -self.view_directions).sum(-1, keepdim=True) # [1, H, W, 1]
            return torch.where(n_dot_v > 0, normal_map, -normal_map)
        return normal_map

    @property
    def global_positions(self) -> Float32[Tensor, "1 H W 3"]:
        if self._gpos_map is None:
            self._gpos_map = self.vertex_attribute_map(self._mesh.vertices)
        return self._gpos_map

    @property
    def view_directions(self) -> Float32[Tensor, "1 H W 3"]:
        if self._view_dir_map is None:
            self._view_dir_map = safe_normalize(self.global_positions - self._camera.c2w[:, 3])
        return self._view_dir_map

    @property
    def alphas(self) -> Float32[Tensor, "1 H W 1"]:
        return (self._rast[..., -1:] > 0).float()

    @property
    def face_indices(self) -> Int64[Tensor, "1 H W 1"]:
        indices = torch.arange(self._mesh.num_faces, device=self._mesh.device)
        if self._visibilities is not None:
            indices = indices[self._visibilities]
        indices = torch.cat((indices, -indices.new_ones(1)))
        if self._facial_indices is None:
            self._facial_indices = indices[self._rast[..., -1:].long() - 1].contiguous()
        return self._facial_indices


@dataclass
class BaseShader(Generic[IMG], ABC):

    antialias: bool = True
    culling: bool = True

    @abstractmethod
    def get_image_class(self) -> Type[IMG]:
        ...

    @abstractmethod
    def shade(self, context: ShadingContext) -> Float32[Tensor, "1 H W C"]:
        ...


@dataclass
class TriangleMesh(TensorDataclass):

    num_vertices: int = Size.Dynamic
    num_faces: int = Size.Dynamic

    vertices: Tensor = Float[num_vertices, 3]
    indices: Tensor = Long[num_faces, 3]

    normals: Optional[Tensor] = Float[num_vertices, 3]
    face_normals: Optional[Tensor] = Float[num_faces, 3, 3]
    uvs: Optional[Tensor] = Float[num_faces, 3, 2]
    kd: Optional[Texture2D] = Texture2D[...]
    ks: Optional[Texture2D] = Texture2D[...]

    @staticmethod
    def from_poisson_reconstruction(oriented_points: Points, *, depth: int = 8) -> TriangleMesh:
        assert oriented_points.normals is not None
        positions = oriented_points.positions.detach().cpu().numpy()
        normals = oriented_points.normals.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)[0]
        vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float().to(oriented_points.device)
        indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long().to(vertices.device)
        return TriangleMesh(vertices=vertices, indices=indices)

    @staticmethod
    def from_diff_poisson_reconstruction(
        oriented_points: Points,
        *,
        resolution: int = 512,
        smoothness: float = 0.02,
    ) -> TriangleMesh:
        assert oriented_points.normals is not None
        V, F, N = diff_poisson_surface_recon(
            oriented_points.positions,
            oriented_points.normals,
            resolution=resolution,
            smoothness=smoothness,
        )
        return TriangleMesh(
            vertices=V,
            indices=F,
            normals=N,
        )

    @staticmethod
    def from_depth_fusion(
        depths: DepthImages,
        *,
        cameras: Cameras,
        voxel_size: float = 0.01,
        sdf_trunc: float = 0.05,
        depth_scale: float = 1.0,
        depth_trunc: float = 4.0,
        alpha_trunc: float = 0.5,
        progress_handle: Optional[Callable] = None,
    ) -> TriangleMesh:
        """
        Generate a TriangleMesh using TSDF fusion from a set of cameras and corresponding depth images.

        Args:
            cameras (Cameras): A Cameras object containing camera parameters (intrinsics and extrinsics).
            depths (DepthImages): A DepthImages object containing depth maps captured by the cameras.

        Returns:
            TriangleMesh: A triangle mesh generated from TSDF fusion.
        """
        assert len(cameras) == len(depths), "Number of cameras must match the number of depth images."

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

        cameras = cameras.detach().cpu()
        intrinsics = cameras.intrinsic_matrix.numpy()
        extrinsics = cameras.view_matrix.numpy()
        Ws = cameras.width.numpy()
        Hs = cameras.height.numpy()
        if progress_handle is not None:
            Ws = progress_handle(Ws)
        for W, H, intrin, extrin, depth_img in zip(Ws, Hs, intrinsics, extrinsics, depths.detach().cpu(), strict=True):
            depth_image = o3d.geometry.Image((depth_img[..., :1] * (depth_img[..., 1:] > alpha_trunc).float()).numpy())

            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
            o3d_intrinsics.set_intrinsics(
                width=W,
                height=H,
                fx=intrin[0, 0],
                fy=intrin[1, 1],
                cx=intrin[0, 2],
                cy=intrin[1, 2],
            )

            volume.integrate(
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.zeros_like(depth_image)),  # No color, only depth
                    depth_image,
                    depth_scale=depth_scale,
                    depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False,
                ),
                o3d_intrinsics,
                extrin,
            )

        mesh = volume.extract_triangle_mesh()

        vertices = torch.from_numpy(np.asarray(mesh.vertices)).float().to(depths.device)
        indices = torch.from_numpy(np.asarray(mesh.triangles)).long().to(depths.device)

        return TriangleMesh(vertices=vertices, indices=indices)

    @staticmethod
    def merge(*meshes: TriangleMesh, only_geometry: bool = False) -> TriangleMesh:
        vertices = []
        indices = []
        tfaces = []
        mfaces = []
        vcnt = 0
        for idx, mesh in enumerate(meshes):
            vertices.append(mesh.vertices)
            indices.append(mesh.indices + vcnt)
            if mesh.uvs is not None:
                tfaces.append(mesh.uvs)
            assert len(tfaces) in [idx + 1, 0]
            if only_geometry:
                continue
            if mesh.kd is not None:
                mfaces.append(mesh.indices.new_empty(mesh.num_faces).fill_(idx))
            assert len(mfaces) in [idx + 1, 0]
            vcnt += mesh.vertices.shape[0]
        vertices = torch.cat(vertices, dim=0)
        indices = torch.cat(indices, dim=0)
        tfaces = torch.cat(tfaces, dim=0) if tfaces else None
        mfaces = torch.cat(mfaces, dim=0) if mfaces else None
        assert vcnt == vertices.shape[0] and indices.max().item() + 1 == vcnt
        if only_geometry or mfaces is None:
            return TriangleMesh(vertices=vertices, indices=indices, uvs=tfaces)
        kd, ks, tfaces = _merge_materials([(m.kd, m.ks) for m in meshes], tfaces=tfaces, mfaces=mfaces)
        return TriangleMesh(vertices=vertices, indices=indices, uvs=tfaces, kd=kd, ks=ks)

    @staticmethod
    def create_sphere(*, radius: float = 1.0, resolution: int = 64) -> TriangleMesh:
        triangle_mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius,
            resolution=resolution,
        )
        vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float()
        indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long()
        return TriangleMesh(
            vertices=vertices,
            indices=indices,
        )

    @staticmethod
    def create_cube(*, size: float = 1.0) -> TriangleMesh:
        triangle_mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float() - size / 2
        indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long()
        return TriangleMesh(
            vertices=vertices,
            indices=indices,
        )

    @staticmethod
    def from_file(
        file: Path,
        *,
        kd_texture: Optional[Texture2D] = None,
        ks_texture: Optional[Texture2D] = None,
    ) -> TriangleMesh:
        assert file.exists() and file.is_file()
        kd, ks = None, None
        mtl_override = (kd_texture is not None or ks_texture is not None)
        if file.suffix == '.ply':
            triangle_mesh = o3d.io.read_triangle_mesh(str(file))
            vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float()
            indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long()
            assert len(indices) > 0
            if triangle_mesh.has_vertex_normals and len(triangle_mesh.vertex_normals) > 0:
                normals = torch.from_numpy(np.asarray(triangle_mesh.vertex_normals)).float()
            else:
                normals = None
            if triangle_mesh.has_triangle_normals and len(triangle_mesh.triangle_normals) > 0:
                face_normals = torch.from_numpy(np.asarray(triangle_mesh.triangle_normals)).float()
            else:
                face_normals = None
            uvs = None
        elif file.suffix == '.obj':
            normals = None
            vertices, indices, face_normals, uvs, kd, ks = _load_obj(file, mtl_override=mtl_override)
        else:
            raise ValueError(file.suffix)
        if mtl_override:
            kd, ks = kd_texture, ks_texture
        return TriangleMesh(
            vertices=vertices[..., :3],
            indices=indices,
            normals=normals,
            face_normals=face_normals,
            uvs=uvs,
            kd=kd,
            ks=ks,
        )

    @torch.no_grad()
    def export(self, path: Path, *, only_geometry: bool = False) -> None:
        assert path.suffix in ['.obj', '.ply']
        if not only_geometry:
            raise NotImplementedError
        path.parent.mkdir(exist_ok=True, parents=True)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.indices.detach().cpu().numpy())
        if self.normals is not None:
            mesh.triangle_normals = o3d.utility.Vector3dVector(self.normals.detach().cpu().numpy())
        o3d.io.write_triangle_mesh(str(path), mesh)

    def visible_faces(self, camera: Cameras) -> Int64[Tensor, "N"]:
        assert camera.is_cuda and camera.shape == ()

        V = self.num_vertices
        ctx = dr.RasterizeCudaContext(camera.device)
        vertices = torch.cat((
            self.vertices,
            torch.ones_like(self.vertices[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        resolution = [camera.height.item(), camera.width.item()]
        mvp = camera.projection_matrix @ camera.view_matrix # [4, 4]
        projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]
        with dr.DepthPeeler(ctx, projected, self.indices.int(), resolution=resolution) as peeler:
            rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
        visible: Tensor = (rast[..., -1:][rast[..., -1:] > 0].long() - 1)
        return visible.unique()

    def render(self, cameras: Cameras, *, shader: BaseShader[IMG]) -> IMG:
        assert cameras.is_cuda

        V = self.num_vertices
        ctx = dr.RasterizeCudaContext(cameras.device)
        vertices = torch.cat((
            self.vertices,
            torch.ones_like(self.vertices[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        context = ShadingContext(mesh=self)

        if shader.culling:
            culling_normals, face_centers = context._culling_info

        images = []
        for camera in cameras.view(-1):
            camera_pos = camera.c2w[:, 3]
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix # [4, 4]
            projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]

            context._clear()
            context._camera = camera
            if shader.culling:
                context._visibilities = ((camera_pos - face_centers) * culling_normals).sum(-1) > 0 # [F]
                indices = self.indices[context._visibilities, :].int()
            else:
                context._visibilities = None
                indices = self.indices.int()

            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
                context._rast = rast

            colors = shader.shade(context)
            alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
            rgba = torch.cat((colors, alphas), dim=-1)
            if shader.antialias:
                rgba = dr.antialias(rgba, rast, projected, indices)
            images.append(rgba.squeeze(0))
        return shader.get_image_class()(images)

    def compute_face_normals(self, *, fix: bool = False) -> TriangleMesh:
        assert self.face_normals is None
        return self.replace().compute_face_normals_(fix=fix)

    def compute_vertex_normals(self, *, fix: bool = False) -> TriangleMesh:
        assert self.normals is None
        return self.replace().compute_vertex_normals_(fix=fix)

    def compute_face_normals_(self, *, fix: bool = False) -> TriangleMesh:
        assert self.face_normals is None
        F = self.num_faces
        expanded_inds = self.indices.view(F, 3, 1).expand(F, 3, 3) # [F, 3, 3]
        vertices = self.vertices.gather(
            dim=-2,
            index=expanded_inds.view(-1, 3)                        # [3F, 3]
        ).view(F, 3, 3)                                            # [F, 3, 3]

        normals = torch.cross(
            vertices[:, 1, :] - vertices[:, 0, :],
            vertices[:, 2, :] - vertices[:, 0, :],
            dim=-1,
        ) # [F, 3]
        normal_lengths = normals.norm(dim=-1, keepdim=True)        # [V, 1]
        if not fix:
            assert (normal_lengths > 1e-10).all()
            normals = normals / normal_lengths
        else:
            fixing = torch.tensor([0, 0, 1]).to(normals)
            normals = torch.where(normal_lengths > 1e-10, normals / normal_lengths.clamp_min(1e-10), fixing)
        return self.annotate_(face_normals=normals.view(F, 1, 3).expand(F, 3, 3))

    def compute_vertex_normals_(self, *, fix: bool = False) -> TriangleMesh:
        assert self.normals is None
        F = self.num_faces
        expanded_inds = self.indices.view(F, 3, 1).expand(F, 3, 3) # [F, 3, 3]
        vertices = self.vertices.gather(
            dim=-2,
            index=expanded_inds.view(-1, 3)                        # [3F, 3]
        ).view(F, 3, 3)                                            # [F, 3, 3]
        weighted_face_normals = torch.cross(
            vertices[:, 1, :] - vertices[:, 0, :],
            vertices[:, 2, :] - vertices[:, 0, :],
            dim=-1,
        ).unsqueeze(-2).expand(expanded_inds.shape)                # [F, 3, 3]
        normals = torch.zeros_like(self.vertices)                  # [V, 3]
        normals.scatter_add_(
            dim=-2,
            index=expanded_inds.flatten(-3, -2),                   # [3F, 3]
            src=weighted_face_normals.flatten(-3, -2),             # [3F, 3]
        )
        normal_lengths = normals.norm(dim=-1, keepdim=True)        # [V, 1]
        if not fix:
            assert (normal_lengths > 1e-10).all()
            normals = normals / normal_lengths
        else:
            fixing = torch.tensor([0, 0, 1]).to(normals)
            normals = torch.where(normal_lengths > 1e-10, normals / normal_lengths.clamp_min(1e-10), fixing)
        return self.annotate_(normals=normals)

    @torch.no_grad()
    def compute_ambient_occlusion(self, cameras: Optional[Cameras] = None) -> Float32[Tensor, "F"]:
        assert self.is_cuda

        if cameras is None:
            cameras = Cameras.from_sphere(
                center=(0, 0, 0),
                up=(0, 0, 1),
                radius=3.,
                num_samples=128,
                resolution=(1024, 1024),
                device=self.device
            )

        V = self.num_vertices
        F = self.num_faces
        ctx = dr.RasterizeCudaContext(cameras.device)
        vertices = torch.cat((
            self.vertices,
            torch.ones_like(self.vertices[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        context = ShadingContext(mesh=self)

        culling_normals, face_centers = context._culling_info

        face_visibilities = torch.zeros(F, device=cameras.device) # [F]
        face_counts = torch.zeros(F, device=cameras.device) # [F]
        for camera in cameras.view(-1):
            camera_pos = camera.c2w[:, 3]
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix # [4, 4]
            projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]
            view_directions = safe_normalize(camera_pos - face_centers) # [F, 3]

            context._clear()
            context._camera = camera
            context._visibilities = (view_directions * culling_normals).sum(-1) > 0 # [F]
            indices = self.indices[context._visibilities, :].int()

            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
                context._rast = rast

            depths = torch.mul(
                context.camera.c2w[:3, 3] - context.global_positions,
                context.camera.c2w[:3, 2]
            ).sum(-1).flatten() # [H * W]

            sample_image_space = camera.project(face_centers) # [F, 3]
            sample_depths = sample_image_space[..., 2] # [F]
            H, W = rast.shape[1:3]
            xs = (sample_image_space[..., 0] + 0.5).long() # [F]
            ys = (sample_image_space[..., 1] + 0.5).long() # [F]
            depths = depths[xs.clamp(min=0, max=W - 1) + ys.clamp(min=0, max=H - 1) * W] # [F]
            valid = (context.alphas > 0).flatten()[xs.clamp(min=0, max=W - 1) + ys.clamp(min=0, max=H - 1) * W] # [F]
            valid = valid & (0 <= xs) & (xs < W) & (0 <= ys) & (ys < H) # [F]
            non_occluded = ((sample_depths < depths + 0.02) & valid).float() # [F]

            weights = (culling_normals * view_directions).sum(-1).clamp_min(1e-4) # [F]
            face_counts += valid.float() * weights
            face_visibilities += non_occluded * weights
        return face_visibilities / face_counts.clamp_min(1e-6)

    def subdivide(self) -> TriangleMesh:
        F = self.num_faces
        V = self.num_vertices

        # with self.create_geometric_processing_context() as GP:
        #     sum_neighbors = GP.Vertex.scatter(GP.HalfEdge.VertexFrom.value(self.vertices)) # [V, 3]
        #     cnt_neighbors = GP.Vertex.scatter(GP.HalfEdge.VertexFrom.constant(1).long()) # [V, 1]
        #     inserted_vertices = GP.Vertex.gather(
        #         GP.UniqueEdge.VertexWinged.value(self.vertices) * 0.125,
        #         GP.UniqueEdge.VertexEnded.value(self.vertices) * 0.375,
        #     ) # [E, 3]
        #     new_indices = GP.Face.collect(GP.Vertex.index(), GP.UniqueEdge.index()) # [F, 6]

        with torch.no_grad():
            edges = self.indices[:, [1, 2, 2, 0, 0, 1]].view(F * 3, 2) # [F * 3, 2]
            sum_neighbors = self.vertices.scatter_reduce(
                dim=0,
                index=edges[:, 0:1].expand(F * 3, 3),                  # [F * 3, 3]
                src=self.vertices[edges[:, 1], :],                     # [F * 3, 3]
                reduce='sum',
                include_self=False,
            )                                                          # [V, 3]
            cnt_neighbors = edges.new_empty(V).scatter_reduce(
                dim=0,
                index=self.indices.flatten(),                          # [F * 3]
                src=torch.ones_like(edges[:, 0]),                      # [F * 3]
                reduce='sum',
                include_self=False,
            ).view(V, 1)                                               # [V, 1]
            assert cnt_neighbors.min().item() >= 3
            weights = torch.where(cnt_neighbors == 3, 7 / 16, 5 / 8)   # [V, 1]
            updated_vertices = torch.add(
                weights * self.vertices,
                (1 - weights) * (sum_neighbors / cnt_neighbors),
            )                                                          # [V, 3]

            edge_code = torch.stack((
                edges.min(1).values,
                edges.max(1).values,
            ), dim=-1)                                                 # [F * 3]
            unique_edges, inverse_indices = edge_code.unique(
                dim=0,
                return_inverse=True,
            )                                                          # [E, 3], [F * 3]\in[0, E)

            assert unique_edges.shape[0] * 2 == F * 3, "Not a manifold."

            inserted_vertices = self.vertices.new_zeros(unique_edges.shape[0], 3) # [E, 3]
            vertices = self.vertices[self.indices.view(F * 3), :].view(F, 3, 3)   # [F, 3, 3]
            wing_sum = 3 * vertices.sum(-2, keepdim=True) - vertices              # [F, 3, 3]
            inserted_vertices.scatter_add_(
                dim=0,
                index=inverse_indices.view(-1, 1).expand(F * 3, 3),               # [F * 3, 3]
                src=wing_sum.view(F * 3, 3) / 16,                                 # [F * 3, 3]
            )
            expanded_indices = torch.cat((
                self.indices,
                inverse_indices.view(F, 3) + V,
            ), dim=-1)                                                            # [F, 6]

        return TriangleMesh(
            vertices=torch.cat((updated_vertices, inserted_vertices), dim=0),
            indices=expanded_indices[:, [0, 5, 4, 4, 3, 2, 3, 4, 5, 5, 1, 3]].view(F * 4, 3)
        )

    def normalize(self, *, scale: float = 1.0) -> TriangleMesh:
        max_bound = self.vertices.view(-1, 3).max(0).values
        min_bound = self.vertices.view(-1, 3).min(0).values
        assert (max_bound > min_bound).all()
        center = (max_bound + min_bound) / 2
        scale = 2 * scale / (max_bound - min_bound).max()
        return self.replace(vertices=(self.vertices - center) * scale)

    def translate(self, x: float, y: float, z: float, /) -> TriangleMesh:
        return self.replace(vertices=self.vertices + torch.tensor([x, y, z]).to(self.vertices))

    def poisson_disk_sample(self, num_samples: int) -> Points:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.indices.detach().cpu().int().numpy())
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(num_samples)
        return Points(
            positions=torch.from_numpy(np.asarray(pcd.points)).to(self.vertices),
            colors=None,
            normals=torch.from_numpy(np.asarray(pcd.normals)).to(self.vertices)
        )

    def uniformly_sample(
        self,
        num_samples: int,
        *,
        samples_per_face: Literal['random', 'uniform', 'uniform+nonzero'] = 'random',
        samples_in_face: Literal['random', 'r2'] = 'random',
        probs_per_face: Literal['uniform', 'area'] = 'area',
    ) -> Points:
        F = self.num_faces
        S = num_samples
        vertices = self.vertices.gather(
            dim=-2,
            index=self.indices.view(*self.shape, F * 3, 1).expand(*self.shape, F * 3, 3) # [..., 3F, 3]
        ).view(*self.shape, F, 3, 3)                                                     # [..., F, 3, 3]
        cross = torch.cross(
            vertices.detach()[..., 1, :] - vertices.detach()[..., 0, :],
            vertices.detach()[..., 2, :] - vertices.detach()[..., 0, :],
            dim=-1
        )                                                                                # [..., F, 3]
        areas_2x: Tensor = cross.norm(dim=-1)                                            # [..., F]
        if probs_per_face == 'area':
            probabilities = areas_2x / areas_2x.sum(-1, keepdim=True)                        # [..., F]
        else:
            assert probs_per_face == 'uniform' and samples_per_face in ['uniform', 'uniform+nonzero']
            probabilities = torch.empty_like(areas_2x).fill_(1.0 / F + 1e-12)
        if self.face_normals is None:
            face_normals = (cross / areas_2x[..., None]).unsqueeze(-2).repeat_interleave(3, dim=-2).contiguous()
        else:
            face_normals = self.face_normals

        if samples_per_face == 'random':
            dist = torch.distributions.Categorical(probabilities)
            sampled_indices = dist.sample([S]).view(S, -1).t().reshape(*self.shape, S)                # [..., S]
        elif samples_per_face in ['uniform', 'uniform+nonzero']:
            ideal_counts = probabilities * num_samples
            counts = ideal_counts.floor().long()                                                      # [..., F]
            if samples_per_face == 'uniform+nonzero' and (counts == 0).any():
                at_least = (1 / probabilities.min()).ceil().long().item()
                raise RuntimeError(
                    'To make sure each face has at least one sample, '
                    f'you should set `num_samples` to at least {at_least} (currently {num_samples}).'
                )
            fracs = ideal_counts - counts
            R = S - counts.sum().item()
            rng = torch.arange(F, device=self.device)
            if R > 0:
                dist = torch.distributions.Categorical(fracs)
                random_parts = dist.sample([R]).view(R, -1).t().reshape(*self.shape, R)               # [..., R]
                uniform_parts = rng.repeat_interleave(counts.view(-1)).view(*self.shape, S - R)       # [..., S-R]
                sampled_indices = torch.cat((uniform_parts, random_parts), dim=-1)                    # [..., S]
            else:
                sampled_indices = rng.repeat_interleave(counts.view(-1)).view(*self.shape, S)         # [..., S]
        else:
            raise ValueError('Argument `samples_per_face` must be any of random, uniform and uniform+nonzero')

        if samples_in_face == 'random':
            u = torch.rand(sampled_indices.shape, device=sampled_indices.device)
            v = torch.rand(sampled_indices.shape, device=sampled_indices.device).sqrt()
            # Barycentric coordinates for sampling within the triangles
            weights = torch.stack((v * (1 - u), u * v, 1 - v), dim=-1) # [..., S, 3]

        elif samples_in_face == 'r2':
            # See https://extremelearning.com.au/evenly-distributing-points-in-a-triangle/
            G = 1.3247179572447460259609088
            rng = torch.arange(num_samples, device=sampled_indices.device)
            u = (rng / G).frac()
            v = (rng / (G * G)).frac()
            flip = u + v > 1
            # Barycentric coordinates for sampling within the triangles
            u = torch.where(flip, 1 - u, u)
            v = torch.where(flip, 1 - v, v)
            weights = torch.stack((u, v, 1 - u - v), dim=-1)           # [..., S, 3]
        else:
            raise ValueError('Argument `samples_in_face` must be any of random and r2')

        sampled_vertices = vertices.gather(
            dim=-3,
            index=sampled_indices[..., None, None].expand(*self.shape, S, 3, 3),  # [..., S, 1, 1]
        )                                                                         # [..., S, 3, 3]
        positions = (sampled_vertices * weights[..., None]).sum(-2)               # [..., S, 3]
        colors = None
        sampled_normals = face_normals.gather(
            dim=-3,
            index=sampled_indices[..., None, None].expand(*self.shape, S, 3, 3),
        )                                                                         # [..., S, 3]
        normals = (sampled_normals * weights[..., None]).sum(-2)

        return Points(
            positions=positions,
            colors=colors,
            normals=normals / normals.norm(dim=-1, keepdim=True),
        )
