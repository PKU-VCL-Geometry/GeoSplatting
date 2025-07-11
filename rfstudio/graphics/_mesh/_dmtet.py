from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float32, Int64
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from ._triangle_mesh import TriangleMesh

_ASSETS_DIR: Path = files('rfstudio') / 'assets' / 'geometry' / 'dmtet'


@lru_cache(maxsize=64)
def _get_base_tet_edges(device: torch.device) -> Int64[Tensor, "2 6"]:
    return torch.tensor([
        [0, 0, 0, 1, 1, 2],
        [1, 2, 3, 2, 3, 3],
    ], dtype=torch.long).to(device)


@lru_cache(maxsize=64)
def _get_triangle_table(device: torch.device) -> Int64[Tensor, "16 6"]:
    _ = -1
    return torch.tensor([
        [_, _, _, _, _, _],
        [1, 0, 2, _, _, _],
        [4, 0, 3, _, _, _],
        [1, 4, 2, 1, 3, 4],
        [3, 1, 5, _, _, _],
        [2, 3, 0, 2, 5, 3],
        [1, 4, 0, 1, 5, 4],
        [4, 2, 5, _, _, _],
        [4, 5, 2, _, _, _],
        [4, 1, 0, 4, 5, 1],
        [3, 2, 0, 3, 5, 2],
        [1, 3, 5, _, _, _],
        [4, 1, 2, 4, 3, 1],
        [3, 0, 4, _, _, _],
        [2, 0, 1, _, _, _],
        [_, _, _, _, _, _],
    ], dtype=torch.long).to(device)


@lru_cache(maxsize=64)
def _get_num_triangles_table(device: torch.device) -> Int64[Tensor, "16"]:
    return torch.tensor([
        [0, 1, 1, 2],
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        [2, 1, 1, 0],
    ], dtype=torch.long).flatten().to(device)


@lru_cache(maxsize=4)
def _get_uvs(num_tets: int, device: torch.device) -> Float32[Tensor, "T*4 2"]:
    N = int(np.ceil(np.sqrt(num_tets)))
    padding = 0.9 / N
    tex_y, tex_x = torch.meshgrid(
        torch.linspace(0, 1 - (1 / N), N, device=device),
        torch.linspace(0, 1 - (1 / N), N, device=device),
        indexing='ij'
    ) # [N, N], [N, N]
    uvs = torch.stack([
        tex_x,
        tex_y,
        tex_x + padding,
        tex_y,
        tex_x + padding,
        tex_y + padding,
        tex_x,
        tex_y + padding,
    ], dim=-1).view(-1, 2)      # [N*N*4, 2]
    return uvs


@dataclass
class DMTet(TensorDataclass):

    num_vertices: int = Size.Dynamic
    num_tets: int = Size.Dynamic

    vertices: Tensor = Float[num_vertices, 3]
    sdf_values: Tensor = Float[num_vertices, 1]
    indices: Tensor = Long[num_tets, 4]

    @classmethod
    def from_predefined(
        self,
        *,
        resolution: Literal[32, 64, 128],
        scale: float = 1.0,
        random_sdf: bool = True,
        device: Optional[torch.device] = None
    ) -> DMTet:
        predefined_file = _ASSETS_DIR / f'{resolution}_tets.npz'
        assert predefined_file.exists()
        predefined = np.load(predefined_file)
        vertices = torch.tensor(predefined['vertices'], dtype=torch.float32, device=device) * (2 * scale)
        indices = torch.tensor(predefined['indices'], dtype=torch.long, device=device)
        sdfs = (
            (torch.rand_like(vertices[..., 0:1]) - 0.1)
            if random_sdf
            else torch.zeros_like(vertices[..., 0:1])
        )
        return DMTet(
            vertices=vertices,
            indices=indices,
            sdf_values=sdfs,
        )

    @torch.no_grad()
    def _get_interp_edges(self) -> Tuple[
        Int64[Tensor, "num_tets"],
        Int64[Tensor, "num_tets"],
        Int64[Tensor, "num_edges 2"],
        Int64[Tensor, "num_tets 6"],
    ]:
        base_tet_edges = _get_base_tet_edges(self.device)                             # [2, 6]
        T = self.num_tets
        occupancy = (self.sdf_values > 0).squeeze(-1)                                 # [V]
        vertex_occupancy = occupancy.unsqueeze(-1).gather(
            dim=-2,
            index=self.indices.view(T * 4, 1)                                         # [4T, 1]
        ).view(T, 4)                                                                  # [T, 4]
        valid_tets = (vertex_occupancy.any(-1)) & ~(vertex_occupancy.all(-1))         # [T]
        valid_indices = self.indices[valid_tets.view(-1, 1).expand(T, 4)].view(-1, 4) # [T', 4]
        tet_codes = torch.mul(
            vertex_occupancy[valid_tets, :].long(),
            torch.pow(2, torch.arange(4, device=valid_tets.device)),
        ).sum(-1)                                                                     # [T']
        tet_global_indices = torch.arange(
            self.num_tets,
            dtype=torch.long,
            device=self.device,
        )[valid_tets]                                                                 # [T]

        # find all vertices
        endpoint_a = valid_indices[..., base_tet_edges[0]]                      # [T', 6]
        endpoint_b = valid_indices[..., base_tet_edges[1]]                      # [T', 6]
        idx_map = -torch.ones_like(endpoint_a, dtype=torch.long)                # [T', 6]
        edge_mask = occupancy[endpoint_a] != occupancy[endpoint_b]              # [T', 6]
        valid_a = endpoint_a[edge_mask]                                         # [E]
        valid_b = endpoint_b[edge_mask]                                         # [E]
        valid_edges = torch.stack((
            torch.minimum(valid_a, valid_b),
            torch.maximum(valid_a, valid_b),
        ), dim=-1)                                                              # [E, 2]
        unique_edges, inv_inds = valid_edges.unique(dim=0, return_inverse=True) # [E', 2], Map[E -> E']
        idx_map[edge_mask] = torch.arange(
            valid_a.shape[0],
            device=valid_a.device,
        )[inv_inds]                                                             # [E]
        return tet_global_indices, tet_codes, unique_edges, idx_map

    def _get_interp_vertices(
        self,
        edges: Float32[Tensor, "E 2"],
        *,
        sdf_eps: Optional[float],
    ) -> Float32[Tensor, "E 3"]:
        v_a = self.vertices[edges[:, 0], :]                # [E, 3]
        v_b = self.vertices[edges[:, 1], :]                # [E, 3]
        sdf_a = self.sdf_values[edges[:, 0], :]            # [E, 1]
        sdf_b = self.sdf_values[edges[:, 1], :]            # [E, 1]
        w_b = sdf_a / (sdf_a - sdf_b)                      # [E, 1]
        if sdf_eps is not None:
            w_b = (1 - sdf_eps) * w_b + (sdf_eps / 2)      # [E, 1]
        return v_b * w_b + v_a * (1 - w_b)                 # [E, 3]

    def marching_tets(
        self,
        *,
        map_uv: bool = False,
        sdf_eps: Optional[float] = None,
    ) -> TriangleMesh:
        triangle_table = _get_triangle_table(self.device)              # [16, 6]
        num_triangles_table = _get_num_triangles_table(self.device)    # [16]
        [
            tet_indices,                                               # [T']
            tet_codes,                                                 # [T']
            edges,                                                     # [E, 2]
            idx_map                                                    # Map[[T', 6] -> E]
        ] = self._get_interp_edges()
        vertices = self._get_interp_vertices(edges, sdf_eps=sdf_eps)           # [E, 3]

        num_triangles = num_triangles_table[tet_codes]     # [T']
        one_tri_mask = num_triangles == 1                  # [T']
        two_tri_mask = num_triangles == 2                  # [T']

        # Generate triangle indices
        indices = torch.cat((
            torch.gather(
                input=idx_map[one_tri_mask, :],                    # Map[[T'', 6] -> E]
                dim=1,
                index=triangle_table[tet_codes[one_tri_mask], :3]  # [T'', 3]
            ).reshape(-1, 3),
            torch.gather(
                input=idx_map[two_tri_mask, :],                    # Map[[T'', 6] -> E]
                dim=1,
                index=triangle_table[tet_codes[two_tri_mask], :6]  # [T'', 6]
            ).reshape(-1, 3),
        ), dim=0)                                                  # [F, 3]

        assert indices.min().item() == 0 and indices.max().item() + 1 == vertices.shape[0]

        if map_uv:
            # Generate triangle uvs
            face_global_indices = torch.cat((
                tet_indices[one_tri_mask] * 2,
                torch.stack((
                    tet_indices[two_tri_mask] * 2,
                    tet_indices[two_tri_mask] * 2 + 1
                ), dim=-1).view(-1)
            ), dim=0)                                      # [F] \in [0, 2T-1]

            tet_idx = (face_global_indices // 2) * 4       # [F] \in [0, 4T-4]
            tri_idx = face_global_indices % 2              # [F] \in [0, 1]

            uv_idx = torch.stack((
                tet_idx,
                tet_idx + tri_idx + 1,
                tet_idx + tri_idx + 2
            ), dim=-1)                                     # [F, 3] \in [0, 4T-1]

            uvs = _get_uvs(self.num_tets, self.device)[uv_idx.view(-1), :].view(-1, 3, 2)
        else:
            uvs = None

        if torch.is_anomaly_enabled():
            assert vertices.isfinite().all()
        return TriangleMesh(vertices=vertices, indices=indices, uvs=uvs)

    def get_tetrahedra(self) -> TriangleMesh:
        vertices = self.vertices
        tet_face_indices = torch.tensor([
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ], dtype=torch.long, device=self.device).flatten()
        indices = self.indices[..., tet_face_indices].view(*self.shape, -1, 3) # [..., 4T, 3]
        return TriangleMesh(vertices=vertices, indices=indices)

    def compute_entropy(self) -> Float32[Tensor, "1"]:
        edges = self._get_interp_edges()[2]                                 # [E, 2]
        sdf_a = self.sdf_values[edges[:, 0]]                                # [E]
        sdf_b = self.sdf_values[edges[:, 1]]                                # [E]
        return torch.add(
            F.binary_cross_entropy_with_logits(sdf_a, (sdf_b > 0).float()),
            F.binary_cross_entropy_with_logits(sdf_b, (sdf_a > 0).float())
        )
