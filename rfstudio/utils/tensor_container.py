from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import torch

from rfstudio.utils.scalar import FloatScalarType, make_scalar_float_tensor


class TensorList:

    def __init__(self, *, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        self._lst = None
        self._dtype = dtype
        self._device = device
        self._len = 0

    def _check_not_empty(self) -> None:
        if self._len == 0:
            raise ValueError("The list is empty!")

    @property
    def dtype(self) -> torch.dtype:
        return self._lst.dtype

    @property
    def device(self) -> torch.device:
        return self._lst.device

    @property
    def capacity(self) -> int:
        return self._lst.shape[0]

    def __len__(self) -> int:
        return self._len

    def __bool__(self) -> bool:
        return self._len > 0

    def __getitem__(self, key: Any) -> torch.Tensor:
        self._check_not_empty()
        return self._lst[:self._len][key]

    def __iter__(self) -> Iterable:
        npy = self.as_numpy()
        for i in range(self._len):
            yield npy[i]

    @torch.no_grad()
    def append(self, value: FloatScalarType) -> None:
        value = make_scalar_float_tensor(value)
        assert value.shape == ()
        if self._lst is None:
            self._lst = torch.empty(8, dtype=value.dtype, device=value.device)
            dtype = self._dtype or value.dtype
            device = self._device or value.device
        else:
            dtype = self._lst.dtype
            device = self._lst.device
        assert dtype == value.dtype, (
            f"Inconsistent dtype: get {value.dtype}, "
            f"but expect {dtype}"
        )
        assert device == value.device, (
            f"Inconsistent device: get {value.device}, "
            f"but expect {device}"
        )
        if self._lst.shape[0] == self._len:
            new_lst = torch.empty(self._lst.shape[0] * 2, dtype=dtype, device=device)
            new_lst[:self._lst.shape[0]].copy_(self._lst)
            del self._lst
            self._lst = new_lst
        self._lst[self._len:self._len + 1] = value.view(1)
        self._len += 1

    def as_tensor(self) -> torch.Tensor:
        return self._lst[:self._len]

    def as_numpy(self) -> Any:
        if self._len == 0:
            return np.empty(0)
        return self._lst[:self._len].cpu().numpy()

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> TensorList:
        assert t.ndim == 1
        tl = TensorList(dtype=t.dtype, device=t.device)
        tl._lst = t
        tl._len = t.shape[0]
