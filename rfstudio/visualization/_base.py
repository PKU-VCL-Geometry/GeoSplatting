from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Iterator,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import rfviser
import rfviser.transforms as tf
import torch
import trimesh

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, RelightDataset, SfMDataset
from rfstudio.graphics import Cameras, Points, Splats, TriangleMesh
from rfstudio.graphics.math import get_arbitrary_tangents_from_normals
from rfstudio.model import GSplatter
from rfstudio.utils.colormap import RainbowColorMap
from rfstudio.utils.webserver import open_webserver


def vis_3dgs(
    model: GSplatter,
    *,
    port: int = 6789,
    host: str = '0.0.0.0',
    backend: Literal['viser', 'custom'] = 'custom',
) -> NoReturn:
    if backend == 'custom':
        with open_webserver('gsplat', port=port, host=host) as basedir:
            model.export_point_cloud(basedir / 'point_cloud.ply')
    elif backend == 'viser':
        server = rfviser.ViserServer(host=host, port=port)
        splats = model.gaussians.detach()
        M = splats.get_cov3d_half()
        server.scene._add_gaussian_splats(
            "/splats",
            centers=splats.means.cpu().numpy(),
            rgbs=splats.colors.cpu().numpy(),
            opacities=splats.opacities.sigmoid().cpu().numpy(),
            covariances=(M @ M.transpose(-1, -2)).cpu().numpy(),
        )
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
    else:
        raise ValueError(backend)


class _ColorManager:

    level: int = 0
    num: int = 0

    @classmethod
    def get_random_color(cls) -> Tuple[float, float, float]:
        index = (cls.num + 0.5) / (2 ** cls.level)
        cls.num += 1
        if cls.num == (2 ** cls.level):
            cls.level += 1
            cls.num = 0
        return tuple(RainbowColorMap().from_scaled(torch.tensor(index)))

    @classmethod
    def reset(cls) -> None:
        cls.level = 0
        cls.num = 0

def _get_aabb_mesh(min_xyz=-np.ones(3), max_xyz=np.ones(3)) -> trimesh.primitives.Box:
    return trimesh.primitives.Box(bounds=np.stack((min_xyz, max_xyz)))


Visualizable: TypeAlias = Union[
    TriangleMesh,
    Points,
    Splats,
    Cameras,
    MeshViewSynthesisDataset,
    MultiViewDataset,
    SfMDataset,
]

T = TypeVar('T', bound=Visualizable)


class _Viser(Generic[T]):

    def __init__(self, name: str, target: T) -> None:
        self._name = name
        self._target = target
        self._config = {}

    def _apply(self, server: rfviser.ViserServer) -> None:
        raise NotImplementedError


class _MeshViser(_Viser[TriangleMesh]):

    def _apply(self, server: rfviser.ViserServer) -> None:
        mesh = self._target
        rand_color = _ColorManager.get_random_color()
        normal_size = self._config.get('normal_size', None)
        server.scene.add_mesh_simple(
            name=(self._name if normal_size is None else f'{self._name}/mesh'),
            vertices=mesh.vertices.detach().cpu().numpy(),
            faces=mesh.indices.detach().cpu().numpy(),
            flat_shading=self._config.get('flat_shading', True),
            wireframe=self._config.get('wireframe', False),
            side=self._config.get('side', 'front'),
            color=rand_color
        )
        if normal_size is not None:
            scale = float(normal_size)
            shape_coeff = scale * 0.1
            if mesh.normals is None:
                mesh = mesh.compute_vertex_normals(fix=True)
            tangent1 = get_arbitrary_tangents_from_normals(mesh.normals)       # [V, 3]
            tangent2 = tangent1.cross(mesh.normals, dim=-1)                    # [V, 3]
            centers = mesh.vertices + mesh.normals * shape_coeff               # [V, 3]
            p0 = mesh.vertices + mesh.normals * scale                          # [V, 3]
            p1 = centers + tangent1 * shape_coeff                              # [V, 3]
            p2 = centers + tangent2 * shape_coeff                              # [V, 3]
            p3 = centers - tangent1 * shape_coeff                              # [V, 3]
            p4 = centers - tangent2 * shape_coeff                              # [V, 3]
            new_vertices = torch.stack((p0, p1, p2, p3, p4), dim=-2)           # [V, 5, 3]
            new_indices = torch.add(
                torch.tensor([
                    [0, 2, 1],
                    [0, 1, 4],
                    [0, 4, 3],
                    [0, 3, 2],
                ], device=p0.device),
                torch.arange(p0.shape[0], device=p0.device).view(-1, 1, 1) * 5 # [V, F, 3]
            )
            server.scene.add_mesh_simple(
                name=f'{self._name}/normals',
                vertices=new_vertices.view(-1, 3).detach().cpu().numpy(),
                faces=new_indices.view(-1, 3).detach().cpu().numpy(),
                flat_shading=True,
                color=(196, 122, 43)
            )

    def configurate(
        self,
        *,
        shade: Literal['flat', 'gouraud', 'wireframe'] = 'flat',
        culling: bool = True,
        normal_size: Optional[float] = None
    ) -> None:
        if shade == 'flat':
            self._config['wireframe'] = False
            self._config['flat_shading'] = True
        elif shade == 'gouraud':
            self._config['wireframe'] = False
            self._config['flat_shading'] = False
        elif shade == 'wireframe':
            self._config['wireframe'] = True
            self._config['flat_shading'] = False
        else:
            raise ValueError(shade)
        self._config['side'] = 'front' if culling else 'double'
        self._config['normal_size'] = normal_size


class _PointsViser(_Viser[Points]):

    def _apply(self, server: rfviser.ViserServer) -> None:
        points = self._target
        if points.colors is None:
            colors = _ColorManager.get_random_color()
        else:
            colors = points.colors.detach().cpu().view(-1, 3).numpy()
        server.scene.add_point_cloud(
            name=self._name,
            points=points.positions.detach().cpu().view(-1, 3).numpy(),
            colors=colors,
            point_size=self._config.get('point_size', 5e-3),
            point_shape=self._config.get('point_shape', 'circle'),
        )

    def configurate(
        self,
        *,
        point_size: float = 0.005,
        point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = 'circle'
    ) -> None:
        self._config['point_size'] = point_size
        self._config['point_shape'] = point_shape
        if point_shape not in ["square", "diamond", "circle", "rounded", "sparkle"]:
            raise ValueError(point_shape)


class _CamerasViser(_Viser[Cameras]):

    def _apply(self, server: rfviser.ViserServer) -> None:
        cameras = self._target.view(-1)
        if cameras.shape[0] == 1:
            names = [self._name]
        else:
            assert cameras.shape[0] < 10000
            names = [f'{self._name}/{i:04d}' for i in range(cameras.shape[0])]
        rand_color = _ColorManager.get_random_color()
        fovs = torch.atan2(cameras.width / 2, cameras.fx) * 2
        aspects = cameras.width / cameras.height
        positions = cameras.c2w[:, :, 3].detach().cpu().numpy()
        rotations = cameras.c2w[:, :3, :3].detach().cpu().numpy()
        for i in range(cameras.shape[0]):
            pose = tf.SE3.from_rotation_and_translation(
                tf.SO3.from_matrix(rotations[i]).multiply(tf.SO3.from_x_radians(np.pi)),
                positions[i],
            )
            server.scene.add_camera_frustum(
                name=names[i],
                fov=fovs[i].item(),
                aspect=aspects[i].item(),
                color=rand_color,
                scale=self._config.get('camera_scale', 0.05),
                wxyz=pose.rotation().wxyz,
                position=pose.translation()
            )

    def configurate(self, *, camera_scale: float = 0.05) -> None:
        self._config['camera_scale'] = camera_scale


class _MultiViewDatasetViser(_Viser[MultiViewDataset]):

    def _apply(self, server: rfviser.ViserServer) -> None:
        for split in self._config.get('splits', ['train', 'val', 'test']):
            if split == 'none':
                cameras = Cameras.stack((
                    self._target.get_inputs(split='train'),
                    self._target.get_inputs(split='val'),
                    self._target.get_inputs(split='test')
                ), dim=0)
            elif split in ['train', 'val', 'test']:
                cameras = self._target.get_inputs(split=split)
            else:
                raise ValueError(split)
            camera_viser = _CamerasViser(name=f'{self._name}/{split}', target=cameras)
            camera_viser.configurate(camera_scale=self._config.get('camera_scale', 0.05))
            camera_viser._apply(server)
        aabb = _get_aabb_mesh()
        server.scene.add_mesh_simple(
            name=f'{self._name}/aabb',
            vertices=aabb.vertices,
            faces=aabb.faces,
            color=_ColorManager.get_random_color(),
            wireframe=True
        )

    def configurate(
        self,
        *,
        camera_scale: float = 0.05,
        splits: Literal[
            'train',
            'val',
            'test',
            'train+test',
            'train+val',
            'val+test',
            'all',
            'none',
        ] = 'all'
    ) -> None:
        self._config['camera_scale'] = camera_scale
        if splits == 'all':
            splits = 'train+val+test'

        if splits == 'none':
            self._config['splits'] = {'none'}
        else:
            split_lst = splits.split('+')
            self._config['splits'] = set(split_lst)


class _SfMDatasetViser(_Viser[SfMDataset]):

    def _apply(self, server: rfviser.ViserServer) -> None:
        _MultiViewDatasetViser._apply(self, server)
        point_viser = _PointsViser(
            name=f'{self._name}/mesh',
            target=self._target.get_meta(split='train').view_as(Points),
        )
        point_viser._config = self._config
        point_viser._apply(server)

    def configurate(
        self,
        *,
        camera_scale: float = 0.05,
        point_size: float = 0.005,
        point_shape: Literal["square", "diamond", "circle", "rounded", "sparkle"] = 'circle',
        splits: Literal[
            'train',
            'val',
            'test',
            'train+test',
            'train+val',
            'val+test',
            'all',
            'none',
        ] = 'all'
    ) -> None:
        _MultiViewDatasetViser.configurate(
            self,
            camera_scale=camera_scale,
            splits=splits,
        )
        _PointsViser.configurate(
            self,
            point_size=point_size,
            point_shape=point_shape,
        )


class _MeshViewSynthesisDatasetViser(_Viser[MeshViewSynthesisDataset]):

    def _apply(self, server: rfviser.ViserServer) -> None:
        _MultiViewDatasetViser._apply(self, server)
        mesh = self._target.get_meta(split='train')
        if mesh is not None:
            mesh_viser = _MeshViser(name=f'{self._name}/mesh', target=mesh)
            mesh_viser._config = self._config
            mesh_viser._apply(server)

    def configurate(
        self,
        *,
        camera_scale: float = 0.05,
        shade: Literal['flat', 'gouraud', 'wireframe'] = 'flat',
        culling: bool = True,
        splits: Literal[
            'train',
            'val',
            'test',
            'train+test',
            'train+val',
            'val+test',
            'all',
            'none',
        ] = 'all'
    ) -> None:
        _MultiViewDatasetViser.configurate(
            self,
            camera_scale=camera_scale,
            splits=splits,
        )
        _MeshViser.configurate(
            self,
            shade=shade,
            culling=culling,
        )


class _SplatsViser(_Viser[Splats]):

    def _apply(self, server: rfviser.ViserServer) -> None:
        splats = self._target.view(-1)
        assert splats.shape == 1
        raise NotImplementedError


class _Slot:

    def __init__(self, paths: Tuple[str] = (), *, storage: List) -> None:
        self.paths = paths
        self.storage = storage

    def __getitem__(self, name: str) -> _Slot:
        return _Slot(self.paths + (name, ), storage=self.storage)

    @overload
    def show(self, target: TriangleMesh) -> _MeshViser:
        ...

    @overload
    def show(self, target: Points) -> _PointsViser:
        ...

    @overload
    def show(self, target: Splats) -> _SplatsViser:
        ...

    @overload
    def show(self, target: Cameras) -> _CamerasViser:
        ...

    @overload
    def show(self, target: Union[MultiViewDataset, RelightDataset]) -> _MultiViewDatasetViser:
        ...

    @overload
    def show(self, target: SfMDataset) -> _SfMDatasetViser:
        ...

    @overload
    def show(self, target: MeshViewSynthesisDataset) -> _MeshViewSynthesisDatasetViser:
        ...

    def show(self, target: Visualizable) -> Any:
        name = '/' + '/'.join([p.replace('/', '_') for p in self.paths])
        if isinstance(target, TriangleMesh):
            viser = _MeshViser(name=name, target=target)
        elif isinstance(target, Points):
            viser = _PointsViser(name=name, target=target)
        elif isinstance(target, Splats):
            viser = _SplatsViser(name=name, target=target)
        elif isinstance(target, Cameras):
            viser = _CamerasViser(name=name, target=target)
        elif isinstance(target, (MultiViewDataset, RelightDataset)):
            viser = _MultiViewDatasetViser(name=name, target=target)
        elif isinstance(target, SfMDataset):
            viser = _SfMDatasetViser(name=name, target=target)
        elif isinstance(target, MeshViewSynthesisDataset):
            viser = _MeshViewSynthesisDatasetViser(name=name, target=target)
        else:
            raise TypeError(target)
        self.storage.append(viser)
        return viser


class _Handle:

    def __init__(self) -> None:
        self.storage: List[_Viser] = []

    def _setup(self, server: rfviser.ViserServer) -> None:
        for viser in self.storage:
            viser._apply(server)

    def __getitem__(self, name: str) -> _Slot:
        return _Slot((name, ), storage=self.storage)


@dataclass
class Visualizer:
    host: str = '0.0.0.0'
    port: int = 6789
    verbose: bool = True

    @contextmanager
    def customize(self) -> Iterator[_Handle]:
        _ColorManager.reset()
        try:
            handle = _Handle()
            yield handle
            server = rfviser.ViserServer(
                host=self.host,
                port=self.port,
                verbose=self.verbose,
            )
            with torch.no_grad():
                handle._setup(server)
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                pass
            finally:
                server.stop()
        finally:
            pass

    def show(self, **kwargs: Visualizable) -> NoReturn:
        _ColorManager.reset()
        with self.customize() as handle:
            for name, value in kwargs.items():
                handle[name].show(value)
