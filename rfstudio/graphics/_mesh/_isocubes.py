from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float32, Int64
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from ._triangle_mesh import TriangleMesh


@lru_cache(maxsize=64)
def _get_cube_edges(device: torch.device) -> Int64[Tensor, "12*2"]:
    return torch.tensor([
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [4, 5],
        [5, 7],
        [7, 6],
        [6, 4],
        [0, 4],
        [1, 5],
        [3, 7],
        [2, 6],
    ], dtype=torch.long).flatten().to(device)


@lru_cache(maxsize=64)
def _get_triangle_table(device: torch.device) -> Int64[Tensor, "256 12"]:
    _ = -1
    A = 10
    B = 11
    return torch.tensor([
        [_, _, _, _, _, _, _, _, _, _, _, _],
        [0, 3, 8, _, _, _, _, _, _, _, _, _],
        [0, 9, 1, _, _, _, _, _, _, _, _, _],
        [3, 8, 1, 1, 8, 9, _, _, _, _, _, _],
        [2, B, 3, _, _, _, _, _, _, _, _, _],
        [8, 0, B, B, 0, 2, _, _, _, _, _, _],
        [3, 2, B, 1, 0, 9, _, _, _, _, _, _],
        [B, 1, 2, B, 9, 1, B, 8, 9, _, _, _],
        [1, A, 2, _, _, _, _, _, _, _, _, _],
        [0, 3, 8, 2, 1, A, _, _, _, _, _, _],
        [A, 2, 9, 9, 2, 0, _, _, _, _, _, _],
        [8, 2, 3, 8, A, 2, 8, 9, A, _, _, _],
        [B, 3, A, A, 3, 1, _, _, _, _, _, _],
        [A, 0, 1, A, 8, 0, A, B, 8, _, _, _],
        [9, 3, 0, 9, B, 3, 9, A, B, _, _, _],
        [8, 9, B, B, 9, A, _, _, _, _, _, _],
        [4, 8, 7, _, _, _, _, _, _, _, _, _],
        [7, 4, 3, 3, 4, 0, _, _, _, _, _, _],
        [4, 8, 7, 0, 9, 1, _, _, _, _, _, _],
        [1, 4, 9, 1, 7, 4, 1, 3, 7, _, _, _],
        [8, 7, 4, B, 3, 2, _, _, _, _, _, _],
        [4, B, 7, 4, 2, B, 4, 0, 2, _, _, _],
        [0, 9, 1, 8, 7, 4, B, 3, 2, _, _, _],
        [7, 4, B, B, 4, 2, 2, 4, 9, 2, 9, 1],
        [4, 8, 7, 2, 1, A, _, _, _, _, _, _],
        [7, 4, 3, 3, 4, 0, A, 2, 1, _, _, _],
        [A, 2, 9, 9, 2, 0, 7, 4, 8, _, _, _],
        [A, 2, 3, A, 3, 4, 3, 7, 4, 9, A, 4],
        [1, A, 3, 3, A, B, 4, 8, 7, _, _, _],
        [A, B, 1, B, 7, 4, 1, B, 4, 1, 4, 0],
        [7, 4, 8, 9, 3, 0, 9, B, 3, 9, A, B],
        [7, 4, B, 4, 9, B, 9, A, B, _, _, _],
        [9, 4, 5, _, _, _, _, _, _, _, _, _],
        [9, 4, 5, 8, 0, 3, _, _, _, _, _, _],
        [4, 5, 0, 0, 5, 1, _, _, _, _, _, _],
        [5, 8, 4, 5, 3, 8, 5, 1, 3, _, _, _],
        [9, 4, 5, B, 3, 2, _, _, _, _, _, _],
        [2, B, 0, 0, B, 8, 5, 9, 4, _, _, _],
        [4, 5, 0, 0, 5, 1, B, 3, 2, _, _, _],
        [5, 1, 4, 1, 2, B, 4, 1, B, 4, B, 8],
        [1, A, 2, 5, 9, 4, _, _, _, _, _, _],
        [9, 4, 5, 0, 3, 8, 2, 1, A, _, _, _],
        [2, 5, A, 2, 4, 5, 2, 0, 4, _, _, _],
        [A, 2, 5, 5, 2, 4, 4, 2, 3, 4, 3, 8],
        [B, 3, A, A, 3, 1, 4, 5, 9, _, _, _],
        [4, 5, 9, A, 0, 1, A, 8, 0, A, B, 8],
        [B, 3, 0, B, 0, 5, 0, 4, 5, A, B, 5],
        [4, 5, 8, 5, A, 8, A, B, 8, _, _, _],
        [8, 7, 9, 9, 7, 5, _, _, _, _, _, _],
        [3, 9, 0, 3, 5, 9, 3, 7, 5, _, _, _],
        [7, 0, 8, 7, 1, 0, 7, 5, 1, _, _, _],
        [7, 5, 3, 3, 5, 1, _, _, _, _, _, _],
        [5, 9, 7, 7, 9, 8, 2, B, 3, _, _, _],
        [2, B, 7, 2, 7, 9, 7, 5, 9, 0, 2, 9],
        [2, B, 3, 7, 0, 8, 7, 1, 0, 7, 5, 1],
        [2, B, 1, B, 7, 1, 7, 5, 1, _, _, _],
        [8, 7, 9, 9, 7, 5, 2, 1, A, _, _, _],
        [A, 2, 1, 3, 9, 0, 3, 5, 9, 3, 7, 5],
        [7, 5, 8, 5, A, 2, 8, 5, 2, 8, 2, 0],
        [A, 2, 5, 2, 3, 5, 3, 7, 5, _, _, _],
        [8, 7, 5, 8, 5, 9, B, 3, A, 3, 1, A],
        [5, B, 7, A, B, 5, 1, 9, 0, _, _, _],
        [B, 5, A, 7, 5, B, 8, 3, 0, _, _, _],
        [5, B, 7, A, B, 5, _, _, _, _, _, _],
        [6, 7, B, _, _, _, _, _, _, _, _, _],
        [7, B, 6, 3, 8, 0, _, _, _, _, _, _],
        [6, 7, B, 0, 9, 1, _, _, _, _, _, _],
        [9, 1, 8, 8, 1, 3, 6, 7, B, _, _, _],
        [3, 2, 7, 7, 2, 6, _, _, _, _, _, _],
        [0, 7, 8, 0, 6, 7, 0, 2, 6, _, _, _],
        [6, 7, 2, 2, 7, 3, 9, 1, 0, _, _, _],
        [6, 7, 8, 6, 8, 1, 8, 9, 1, 2, 6, 1],
        [B, 6, 7, A, 2, 1, _, _, _, _, _, _],
        [3, 8, 0, B, 6, 7, A, 2, 1, _, _, _],
        [0, 9, 2, 2, 9, A, 7, B, 6, _, _, _],
        [6, 7, B, 8, 2, 3, 8, A, 2, 8, 9, A],
        [7, A, 6, 7, 1, A, 7, 3, 1, _, _, _],
        [8, 0, 7, 7, 0, 6, 6, 0, 1, 6, 1, A],
        [7, 3, 6, 3, 0, 9, 6, 3, 9, 6, 9, A],
        [6, 7, A, 7, 8, A, 8, 9, A, _, _, _],
        [B, 6, 8, 8, 6, 4, _, _, _, _, _, _],
        [6, 3, B, 6, 0, 3, 6, 4, 0, _, _, _],
        [B, 6, 8, 8, 6, 4, 1, 0, 9, _, _, _],
        [1, 3, 9, 3, B, 6, 9, 3, 6, 9, 6, 4],
        [2, 8, 3, 2, 4, 8, 2, 6, 4, _, _, _],
        [4, 0, 6, 6, 0, 2, _, _, _, _, _, _],
        [9, 1, 0, 2, 8, 3, 2, 4, 8, 2, 6, 4],
        [9, 1, 4, 1, 2, 4, 2, 6, 4, _, _, _],
        [4, 8, 6, 6, 8, B, 1, A, 2, _, _, _],
        [1, A, 2, 6, 3, B, 6, 0, 3, 6, 4, 0],
        [B, 6, 4, B, 4, 8, A, 2, 9, 2, 0, 9],
        [A, 4, 9, 6, 4, A, B, 2, 3, _, _, _],
        [4, 8, 3, 4, 3, A, 3, 1, A, 6, 4, A],
        [1, A, 0, A, 6, 0, 6, 4, 0, _, _, _],
        [4, A, 6, 9, A, 4, 0, 8, 3, _, _, _],
        [4, A, 6, 9, A, 4, _, _, _, _, _, _],
        [6, 7, B, 4, 5, 9, _, _, _, _, _, _],
        [4, 5, 9, 7, B, 6, 3, 8, 0, _, _, _],
        [1, 0, 5, 5, 0, 4, B, 6, 7, _, _, _],
        [B, 6, 7, 5, 8, 4, 5, 3, 8, 5, 1, 3],
        [3, 2, 7, 7, 2, 6, 9, 4, 5, _, _, _],
        [5, 9, 4, 0, 7, 8, 0, 6, 7, 0, 2, 6],
        [3, 2, 6, 3, 6, 7, 1, 0, 5, 0, 4, 5],
        [6, 1, 2, 5, 1, 6, 4, 7, 8, _, _, _],
        [A, 2, 1, 6, 7, B, 4, 5, 9, _, _, _],
        [0, 3, 8, 4, 5, 9, B, 6, 7, A, 2, 1],
        [7, B, 6, 2, 5, A, 2, 4, 5, 2, 0, 4],
        [8, 4, 7, 5, A, 6, 3, B, 2, _, _, _],
        [9, 4, 5, 7, A, 6, 7, 1, A, 7, 3, 1],
        [A, 6, 5, 7, 8, 4, 1, 9, 0, _, _, _],
        [4, 3, 0, 7, 3, 4, 6, 5, A, _, _, _],
        [A, 6, 5, 8, 4, 7, _, _, _, _, _, _],
        [9, 6, 5, 9, B, 6, 9, 8, B, _, _, _],
        [B, 6, 3, 3, 6, 0, 0, 6, 5, 0, 5, 9],
        [B, 6, 5, B, 5, 0, 5, 1, 0, 8, B, 0],
        [B, 6, 3, 6, 5, 3, 5, 1, 3, _, _, _],
        [9, 8, 5, 8, 3, 2, 5, 8, 2, 5, 2, 6],
        [5, 9, 6, 9, 0, 6, 0, 2, 6, _, _, _],
        [1, 6, 5, 2, 6, 1, 3, 0, 8, _, _, _],
        [1, 6, 5, 2, 6, 1, _, _, _, _, _, _],
        [2, 1, A, 9, 6, 5, 9, B, 6, 9, 8, B],
        [9, 0, 1, 3, B, 2, 5, A, 6, _, _, _],
        [B, 0, 8, 2, 0, B, A, 6, 5, _, _, _],
        [3, B, 2, 5, A, 6, _, _, _, _, _, _],
        [1, 8, 3, 9, 8, 1, 5, A, 6, _, _, _],
        [6, 5, A, 0, 1, 9, _, _, _, _, _, _],
        [8, 3, 0, 5, A, 6, _, _, _, _, _, _],
        [6, 5, A, _, _, _, _, _, _, _, _, _],
        [A, 5, 6, _, _, _, _, _, _, _, _, _],
        [0, 3, 8, 6, A, 5, _, _, _, _, _, _],
        [A, 5, 6, 9, 1, 0, _, _, _, _, _, _],
        [3, 8, 1, 1, 8, 9, 6, A, 5, _, _, _],
        [2, B, 3, 6, A, 5, _, _, _, _, _, _],
        [8, 0, B, B, 0, 2, 5, 6, A, _, _, _],
        [1, 0, 9, 2, B, 3, 6, A, 5, _, _, _],
        [5, 6, A, B, 1, 2, B, 9, 1, B, 8, 9],
        [5, 6, 1, 1, 6, 2, _, _, _, _, _, _],
        [5, 6, 1, 1, 6, 2, 8, 0, 3, _, _, _],
        [6, 9, 5, 6, 0, 9, 6, 2, 0, _, _, _],
        [6, 2, 5, 2, 3, 8, 5, 2, 8, 5, 8, 9],
        [3, 6, B, 3, 5, 6, 3, 1, 5, _, _, _],
        [8, 0, 1, 8, 1, 6, 1, 5, 6, B, 8, 6],
        [B, 3, 6, 6, 3, 5, 5, 3, 0, 5, 0, 9],
        [5, 6, 9, 6, B, 9, B, 8, 9, _, _, _],
        [5, 6, A, 7, 4, 8, _, _, _, _, _, _],
        [0, 3, 4, 4, 3, 7, A, 5, 6, _, _, _],
        [5, 6, A, 4, 8, 7, 0, 9, 1, _, _, _],
        [6, A, 5, 1, 4, 9, 1, 7, 4, 1, 3, 7],
        [7, 4, 8, 6, A, 5, 2, B, 3, _, _, _],
        [A, 5, 6, 4, B, 7, 4, 2, B, 4, 0, 2],
        [4, 8, 7, 6, A, 5, 3, 2, B, 1, 0, 9],
        [1, 2, A, B, 7, 6, 9, 5, 4, _, _, _],
        [2, 1, 6, 6, 1, 5, 8, 7, 4, _, _, _],
        [0, 3, 7, 0, 7, 4, 2, 1, 6, 1, 5, 6],
        [8, 7, 4, 6, 9, 5, 6, 0, 9, 6, 2, 0],
        [7, 2, 3, 6, 2, 7, 5, 4, 9, _, _, _],
        [4, 8, 7, 3, 6, B, 3, 5, 6, 3, 1, 5],
        [5, 0, 1, 4, 0, 5, 7, 6, B, _, _, _],
        [9, 5, 4, 6, B, 7, 0, 8, 3, _, _, _],
        [B, 7, 6, 9, 5, 4, _, _, _, _, _, _],
        [6, A, 4, 4, A, 9, _, _, _, _, _, _],
        [6, A, 4, 4, A, 9, 3, 8, 0, _, _, _],
        [0, A, 1, 0, 6, A, 0, 4, 6, _, _, _],
        [6, A, 1, 6, 1, 8, 1, 3, 8, 4, 6, 8],
        [9, 4, A, A, 4, 6, 3, 2, B, _, _, _],
        [2, B, 8, 2, 8, 0, 6, A, 4, A, 9, 4],
        [B, 3, 2, 0, A, 1, 0, 6, A, 0, 4, 6],
        [6, 8, 4, B, 8, 6, 2, A, 1, _, _, _],
        [4, 1, 9, 4, 2, 1, 4, 6, 2, _, _, _],
        [3, 8, 0, 4, 1, 9, 4, 2, 1, 4, 6, 2],
        [6, 2, 4, 4, 2, 0, _, _, _, _, _, _],
        [3, 8, 2, 8, 4, 2, 4, 6, 2, _, _, _],
        [4, 6, 9, 6, B, 3, 9, 6, 3, 9, 3, 1],
        [8, 6, B, 4, 6, 8, 9, 0, 1, _, _, _],
        [B, 3, 6, 3, 0, 6, 0, 4, 6, _, _, _],
        [8, 6, B, 4, 6, 8, _, _, _, _, _, _],
        [A, 7, 6, A, 8, 7, A, 9, 8, _, _, _],
        [3, 7, 0, 7, 6, A, 0, 7, A, 0, A, 9],
        [6, A, 7, 7, A, 8, 8, A, 1, 8, 1, 0],
        [6, A, 7, A, 1, 7, 1, 3, 7, _, _, _],
        [3, 2, B, A, 7, 6, A, 8, 7, A, 9, 8],
        [2, 9, 0, A, 9, 2, 6, B, 7, _, _, _],
        [0, 8, 3, 7, 6, B, 1, 2, A, _, _, _],
        [7, 6, B, 1, 2, A, _, _, _, _, _, _],
        [2, 1, 9, 2, 9, 7, 9, 8, 7, 6, 2, 7],
        [2, 7, 6, 3, 7, 2, 0, 1, 9, _, _, _],
        [8, 7, 0, 7, 6, 0, 6, 2, 0, _, _, _],
        [7, 2, 3, 6, 2, 7, _, _, _, _, _, _],
        [8, 1, 9, 3, 1, 8, B, 7, 6, _, _, _],
        [B, 7, 6, 1, 9, 0, _, _, _, _, _, _],
        [6, B, 7, 0, 8, 3, _, _, _, _, _, _],
        [B, 7, 6, _, _, _, _, _, _, _, _, _],
        [7, B, 5, 5, B, A, _, _, _, _, _, _],
        [A, 5, B, B, 5, 7, 0, 3, 8, _, _, _],
        [7, B, 5, 5, B, A, 0, 9, 1, _, _, _],
        [7, B, A, 7, A, 5, 3, 8, 1, 8, 9, 1],
        [5, 2, A, 5, 3, 2, 5, 7, 3, _, _, _],
        [5, 7, A, 7, 8, 0, A, 7, 0, A, 0, 2],
        [0, 9, 1, 5, 2, A, 5, 3, 2, 5, 7, 3],
        [9, 7, 8, 5, 7, 9, A, 1, 2, _, _, _],
        [1, B, 2, 1, 7, B, 1, 5, 7, _, _, _],
        [8, 0, 3, 1, B, 2, 1, 7, B, 1, 5, 7],
        [7, B, 2, 7, 2, 9, 2, 0, 9, 5, 7, 9],
        [7, 9, 5, 8, 9, 7, 3, B, 2, _, _, _],
        [3, 1, 7, 7, 1, 5, _, _, _, _, _, _],
        [8, 0, 7, 0, 1, 7, 1, 5, 7, _, _, _],
        [0, 9, 3, 9, 5, 3, 5, 7, 3, _, _, _],
        [9, 7, 8, 5, 7, 9, _, _, _, _, _, _],
        [8, 5, 4, 8, A, 5, 8, B, A, _, _, _],
        [0, 3, B, 0, B, 5, B, A, 5, 4, 0, 5],
        [1, 0, 9, 8, 5, 4, 8, A, 5, 8, B, A],
        [A, 3, B, 1, 3, A, 9, 5, 4, _, _, _],
        [3, 2, 8, 8, 2, 4, 4, 2, A, 4, A, 5],
        [A, 5, 2, 5, 4, 2, 4, 0, 2, _, _, _],
        [5, 4, 9, 8, 3, 0, A, 1, 2, _, _, _],
        [2, A, 1, 4, 9, 5, _, _, _, _, _, _],
        [8, B, 4, B, 2, 1, 4, B, 1, 4, 1, 5],
        [0, 5, 4, 1, 5, 0, 2, 3, B, _, _, _],
        [0, B, 2, 8, B, 0, 4, 9, 5, _, _, _],
        [5, 4, 9, 2, 3, B, _, _, _, _, _, _],
        [4, 8, 5, 8, 3, 5, 3, 1, 5, _, _, _],
        [0, 5, 4, 1, 5, 0, _, _, _, _, _, _],
        [5, 4, 9, 3, 0, 8, _, _, _, _, _, _],
        [5, 4, 9, _, _, _, _, _, _, _, _, _],
        [B, 4, 7, B, 9, 4, B, A, 9, _, _, _],
        [0, 3, 8, B, 4, 7, B, 9, 4, B, A, 9],
        [B, A, 7, A, 1, 0, 7, A, 0, 7, 0, 4],
        [3, A, 1, B, A, 3, 7, 8, 4, _, _, _],
        [3, 2, A, 3, A, 4, A, 9, 4, 7, 3, 4],
        [9, 2, A, 0, 2, 9, 8, 4, 7, _, _, _],
        [3, 4, 7, 0, 4, 3, 1, 2, A, _, _, _],
        [7, 8, 4, A, 1, 2, _, _, _, _, _, _],
        [7, B, 4, 4, B, 9, 9, B, 2, 9, 2, 1],
        [1, 9, 0, 4, 7, 8, 2, 3, B, _, _, _],
        [7, B, 4, B, 2, 4, 2, 0, 4, _, _, _],
        [4, 7, 8, 2, 3, B, _, _, _, _, _, _],
        [9, 4, 1, 4, 7, 1, 7, 3, 1, _, _, _],
        [7, 8, 4, 1, 9, 0, _, _, _, _, _, _],
        [3, 4, 7, 0, 4, 3, _, _, _, _, _, _],
        [7, 8, 4, _, _, _, _, _, _, _, _, _],
        [B, A, 8, 8, A, 9, _, _, _, _, _, _],
        [0, 3, 9, 3, B, 9, B, A, 9, _, _, _],
        [1, 0, A, 0, 8, A, 8, B, A, _, _, _],
        [A, 3, B, 1, 3, A, _, _, _, _, _, _],
        [3, 2, 8, 2, A, 8, A, 9, 8, _, _, _],
        [9, 2, A, 0, 2, 9, _, _, _, _, _, _],
        [8, 3, 0, A, 1, 2, _, _, _, _, _, _],
        [2, A, 1, _, _, _, _, _, _, _, _, _],
        [2, 1, B, 1, 9, B, 9, 8, B, _, _, _],
        [B, 2, 3, 9, 0, 1, _, _, _, _, _, _],
        [B, 0, 8, 2, 0, B, _, _, _, _, _, _],
        [3, B, 2, _, _, _, _, _, _, _, _, _],
        [1, 8, 3, 9, 8, 1, _, _, _, _, _, _],
        [1, 9, 0, _, _, _, _, _, _, _, _, _],
        [8, 3, 0, _, _, _, _, _, _, _, _, _],
        [_, _, _, _, _, _, _, _, _, _, _, _],
    ], dtype=torch.long).to(device)


@lru_cache(maxsize=64)
def _get_num_triangles_table(device: torch.device) -> Int64[Tensor, "256"]:
    return torch.tensor([
        [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2],
        [1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3],
        [1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3],
        [2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2],
        [1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3],
        [2, 3, 3, 4, 3, 2, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2],
        [2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2],
        [3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1],
        [1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 3],
        [2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 4, 3, 3, 2],
        [2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 2, 3, 4, 3, 3, 2],
        [3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1],
        [2, 3, 3, 4, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2],
        [3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1],
        [3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1],
        [2, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0],
    ], dtype=torch.long).flatten().to(device)


@dataclass
class IsoCubes(TensorDataclass):

    num_vertices: int = Size.Dynamic
    num_cubes: int = Size.Dynamic

    vertices: Tensor = Float[num_vertices, 3]
    sdf_values: Tensor = Float[num_vertices, 1]
    indices: Tensor = Long[num_cubes, 8]
    resolution: Tensor = Long[3]

    @classmethod
    def from_resolution(
        cls,
        *resolution: int,
        device: Optional[torch.device] = None,
        random_sdf: bool = True,
        scale: float = 1.0,
    ) -> IsoCubes:
        assert len(resolution) in [1, 3]
        if len(resolution) == 1:
            resolution = (resolution[0], resolution[0], resolution[0])
        voxel_grid_template = torch.ones(
            resolution[0] + 1,
            resolution[1] + 1,
            resolution[2] + 1,
            device=device,
        ) # [R + 1, R + 1, R + 1]

        cube_corners = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=torch.long, device=device) # [8, 3]

        res = torch.tensor(resolution, dtype=torch.long, device=device) # [3]
        coords = torch.nonzero(voxel_grid_template).float() # [R+ * R+ * R+, 3]
        verts = coords.reshape(-1, 3) / res # [R+ * R+ * R+, 3]
        cubes = torch.arange(resolution[0] * resolution[1] * resolution[2], device=device) # [R*R*R]
        cubes = torch.stack((
            cubes % resolution[0],
            (cubes // resolution[0]) % resolution[1],
            cubes // (resolution[1] * resolution[0]),
        ), dim=-1)[:, None, :] + cube_corners # [R*R*R, 8, 3]
        cubes = (cubes[..., 2] * (1 + resolution[1]) + cubes[..., 1]) * (1 + resolution[0]) + cubes[..., 0] # [R * R * R, 8]

        sdfs = (
            (torch.rand_like(verts[..., 0:1]) - 0.1)
            if random_sdf
            else torch.zeros_like(verts[..., 0:1])
        )

        return IsoCubes(
            vertices=(2 * verts - 1) * scale,
            indices=cubes,
            sdf_values=sdfs,
            resolution=res,
        )

    @torch.no_grad()
    def _get_interp_edges(self) -> Tuple[
        Int64[Tensor, "C'"],
        Int64[Tensor, "C'"],
        Int64[Tensor, "E 2"],
        Int64[Tensor, "C' 6"],
    ]:
        base_cube_edges = _get_cube_edges(self.device).view(12, 2)                    # [12, 2]
        C = self.num_cubes
        occupancy = (self.sdf_values > 0).squeeze(-1)                                 # [V]
        vertex_occupancy = occupancy.unsqueeze(-1).gather(
            dim=-2,
            index=self.indices.view(C * 8, 1)                                         # [8C, 1]
        ).view(C, 8)                                                                  # [C, 8]
        valid_cubes = (vertex_occupancy.any(-1)) & ~(vertex_occupancy.all(-1))         # [C]
        valid_indices = self.indices[valid_cubes.view(-1, 1).expand(C, 8)].view(-1, 8) # [C', 8]
        cube_codes = torch.mul(
            vertex_occupancy[valid_cubes, :].long(),
            torch.pow(2, torch.arange(8, device=valid_cubes.device)),
        ).sum(-1)                                                                     # [C']
        cube_global_indices = torch.arange(
            self.num_cubes,
            dtype=torch.long,
            device=self.device,
        )[valid_cubes]                                                                 # [C']

        # find all vertices
        endpoint_a = valid_indices[..., base_cube_edges[:, 0]]                      # [C', 12]
        endpoint_b = valid_indices[..., base_cube_edges[:, 1]]                      # [C', 12]
        idx_map = -torch.ones_like(endpoint_a, dtype=torch.long)                # [C', 12]
        edge_mask = occupancy[endpoint_a] != occupancy[endpoint_b]              # [C', 12]
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
        return cube_global_indices, cube_codes, unique_edges, idx_map

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

    def marching_cubes(
        self,
        *,
        sdf_eps: Optional[float] = None,
    ) -> TriangleMesh:
        triangle_table = _get_triangle_table(self.device)              # [256, 12]
        num_triangles_table = _get_num_triangles_table(self.device)    # [256]
        [
            cube_indices,                                              # [C']
            cube_codes,                                                # [C']
            edges,                                                     # [E, 2]
            idx_map,                                                   # Map[[C', 12] -> E]
        ] = self._get_interp_edges()
        vertices = self._get_interp_vertices(edges, sdf_eps=sdf_eps)           # [E, 3]

        num_triangles = num_triangles_table[cube_codes]     # [C']
        indices = []
        for i in range(1, 5):
            tri_mask = num_triangles == i                  # [C']
            indices.append(
                idx_map[tri_mask, :].gather(
                    dim=1,
                    index=triangle_table[cube_codes[tri_mask], :(i*3)]  # [C'', i*3]
                ).reshape(-1, 3)
            )
        indices = torch.cat(indices)
        assert indices.min().item() == 0 and indices.max().item() + 1 == vertices.shape[0]
        return TriangleMesh(vertices=vertices, indices=indices)

    def compute_entropy(self) -> Float32[Tensor, "1"]:
        edges = self._get_interp_edges()[2]                                 # [E, 2]
        sdf_a = self.sdf_values[edges[:, 0]]                                # [E]
        sdf_b = self.sdf_values[edges[:, 1]]                                # [E]
        return torch.add(
            F.binary_cross_entropy_with_logits(sdf_a, (sdf_b > 0).float()),
            F.binary_cross_entropy_with_logits(sdf_b, (sdf_a > 0).float())
        )
