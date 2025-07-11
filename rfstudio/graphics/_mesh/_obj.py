from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from ._texture import Texture2D


@torch.no_grad()
def _merge_materials(
    textures: List[Tuple[Texture2D, Optional[Texture2D]]],
    tfaces: Float32[Tensor, "F 3 2"],
    mfaces: Int64[Tensor, "F"],
) -> Tuple[
    Texture2D,
    Texture2D,
    Float32[Tensor, "F 3 2"],
]:
    assert len(textures) > 0
    preprocessed_textures = []
    preprocessed_size = []
    for kd, ks in textures:
        if ks is None:
            ks = Texture2D.from_constants((0, 0, 0), device=kd.device)
        assert kd.data.shape == ks.data.shape or ks.is_constant or kd.is_constant
        if ks.is_constant:
            ks = ks.expand_to(kd.width, kd.height)
        if kd.is_constant:
            kd = kd.expand_to(ks.width, ks.height)
        preprocessed_textures.append((kd, ks))
        preprocessed_size.append((kd.width, kd.height))

    max_height = max(h for w, h in preprocessed_size)
    merged_width = sum(w for w, h in preprocessed_size)
    merged_kd = tfaces.new_zeros(max_height, merged_width, 3)
    merged_ks = tfaces.new_zeros(max_height, merged_width, 3)
    new_tfaces = torch.empty_like(tfaces)

    base_width = 0
    for idx, (kd, ks) in enumerate(textures):
        # 0 --- bw --- bw+kw --- mw
        # map u to pixel: 0.5 + u * (W - 1)
        # remap: 0.5 + u * (W - 1) + bw = 0.5 + new_u * (MW - 1)
        #   -->  new_u = (u * (W - 1) + bw) / (MW - 1)
        mask = mfaces == idx
        curr_tfaces = tfaces[mask, :, :].clone() # [K, 3, 2]
        curr_tfaces[..., 0] = (curr_tfaces[..., 0] * (kd.width - 1) + base_width) / (merged_width - 1)
        new_tfaces[mask, :, :] = curr_tfaces
        merged_kd[:kd.height, base_width:base_width+kd.width, :] = kd.data
        merged_ks[:kd.height, base_width:base_width+kd.width, :] = ks.data
        base_width += kd.width

    return Texture2D(data=merged_kd), Texture2D(data=merged_ks), new_tfaces


def _load_obj(
    filename: Path,
    *,
    mtl_override: bool,
    device: Optional[torch.device] = None,
) -> Tuple[
    Float32[Tensor, "V 3"],
    Int64[Tensor, "F 3"],
    Optional[Float32[Tensor, "F 3 3"]],
    Optional[Float32[Tensor, "F 3 2"]],
    Optional[Texture2D],
    Optional[Texture2D],
]:
    '''
    Adapted from https://github.com/NVlabs/nvdiffrec/blob/main/render/obj.py
    '''

    # Read entire file
    with filename.open('r') as f:
        lines = [line.split() for line in f.readlines()]

    # Load materials
    all_materials = {}
    if not mtl_override:
        for line in lines:
            if len(line) == 0:
                continue
            if line[0] == 'mtllib':
                assert len(line) == 2
                all_materials.update(Texture2D.from_mtl_file(filename.parent / line[1]))

    # load vertices
    vertices, texcoords, face_normals = [], [], []
    for line in lines:
        if len(line) == 0:
            continue
        prefix = line[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line[1:]]
            texcoords.append([val[0], val[1]])
        elif prefix == 'vn':
            val = [float(v) for v in line[1:]]
            face_normals.append([val[0], val[1], val[2]])
        elif prefix == 'f':
            for idx, v in enumerate(line[1:]):
                if '/' not in v and int(v) < 0:
                    line[idx + 1] = str(len(vertices) + 1 + int(v))
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device) # [V, 3]
    if len(texcoords) > 0:
        texcoords = torch.tensor(texcoords, dtype=torch.float32)          # [VN, 2]
    else:
        texcoords = None
    if len(face_normals) > 0:
        face_normals = torch.tensor(face_normals, dtype=torch.float32)    # [VN, 3]
    else:
        face_normals = None

    # load faces
    activeMatIdx = None
    has_texture = mtl_override
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line) == 0:
            continue
        prefix = line[0].lower()
        if prefix == 'usemtl' and not mtl_override:                    # Track used materials
            assert line[1] in all_materials
            if line[1] not in used_materials:
                used_materials.append(line[1])
            activeMatIdx = used_materials.index(line[1])
            has_texture = True
        elif prefix == 'f':                                            # Parse face
            vs = line[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if (len(vv) > 1 and vv[1] != "") else -1
            n0 = int(vv[2]) - 1 if (len(vv) > 2 and vv[2] != "") else -1
            for i in range(nv - 2):                                    # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if (len(vv) > 1 and vv[1] != "") else -1
                n1 = int(vv[2]) - 1 if (len(vv) > 2 and vv[2] != "") else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if (len(vv) > 1 and vv[1] != "") else -1
                n2 = int(vv[2]) - 1 if (len(vv) > 2 and vv[2] != "") else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                if n0 != -1 and n1 != -1 and n2 != -1:
                    nfaces.append(face_normals[[n0, n1, n2], :])
                elif nfaces:
                    normal = (vertices[v1] - vertices[v0]).cross(vertices[v2] - vertices[v0], dim=-1) # [3]
                    normal = normal / normal.norm(dim=-1, keepdim=True)
                    nfaces.append(normal.expand(3, 3))
                if has_texture:
                    if mtl_override or all([
                        activeMatIdx is not None,
                        not all_materials[used_materials[activeMatIdx]][0].is_constant,
                    ]):
                        assert texcoords is not None and t0 != -1 and t1 != -1 and t2 != -1
                        tfaces.append(texcoords[[t0, t1, t2], :])          # [3, 2]
                    else:
                        # assert t0 == -1 and t1 == -1 and t2 == -1
                        tfaces.append(torch.zeros((3, 2), device=device))

    if has_texture:
        assert len(tfaces) == len(faces)
    assert len(nfaces) in (len(faces), 0)

    faces = torch.tensor(faces, dtype=torch.int64, device=device)           # [F, 3]
    tfaces = torch.stack(tfaces, dim=0).to(device) if has_texture else None # [F, 3, 2]
    nfaces = torch.stack(nfaces, dim=0).to(device) if nfaces else None # [F, 3, 3]

    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        mfaces = torch.tensor(mfaces, dtype=torch.int64, device=device) # [F]
        kd, ks, tfaces = _merge_materials(
            [all_materials[name] for name in used_materials],
            tfaces,
            mfaces,
        )
    elif len(used_materials) == 1:
        kd, ks = all_materials[used_materials[0]]
    else:
        kd, ks = None, None

    return (vertices, faces, nfaces, tfaces, kd, ks)
