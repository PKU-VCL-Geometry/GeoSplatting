import pathlib
from typing import Any, TypeAlias, Union

import numpy as np
import torch
from numpy.typing import NDArray

ScalarType: TypeAlias = Union[
    torch.Tensor,
    NDArray[Any],
    int,
    np.integer,
    float,
    np.floating,
    bool,
    np.bool_,
    str,
    pathlib.Path,
]


def is_scalar(value: Any) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if value.view(-1).shape != (1, ):
            return False
        value = value.item()
    if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating, str, pathlib.Path)):
        return True
    return False


def make_scalar_pure(value: ScalarType) -> Union[int, float, bool, str, pathlib.Path]:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return value.item() if isinstance(value, np.bool_) else value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    assert isinstance(value, (str, pathlib.Path)), f"A {torch.typename(value)} instance is not a valid scalar."
    return value


FloatScalarType: TypeAlias = Union[
    torch.Tensor,
    NDArray[Any],
    np.integer,
    np.floating,
]


def is_float_scalar(value: Any) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if value.view(-1).shape != (1, ):
            return False
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return True
    return False


def make_scalar_float(value: FloatScalarType) -> float:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        value = value.item()
    assert not isinstance(value, (bool, np.bool_)), "A bool instance is not a float scalar."
    if isinstance(value, (float, np.floating)):
        return float(value)
    assert isinstance(value, (int, np.integer)), f"A {torch.typename(value)} instance is not a float scalar."
    return float(int(value))


def make_scalar_float_tensor(value: FloatScalarType) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        assert value.dtype in (np.float16, np.float32, np.float64), f"NDArray({value.dtype}) is not a float scalar."
        return torch.from_numpy(value)
    assert not isinstance(value, (bool, np.bool_)), "A bool instance is not a float scalar."
    if isinstance(value, (float, np.floating)):
        return torch.tensor(float(value), dtype=torch.float32)
    assert isinstance(value, (int, np.integer)), f"A {torch.typename(value)} instance is not an float scalar."
    return torch.tensor(float(int(value)), dtype=torch.float32)


IntScalarType: TypeAlias = Union[
    torch.Tensor,
    NDArray[Any],
    np.integer,
]


def is_int_scalar(value: Any) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if value.view(-1).shape != (1, ):
            return False
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return False


def make_scalar_int(value: IntScalarType) -> int:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        value = value.item()
    assert not isinstance(value, (bool, np.bool_)), "A bool instance is not an int scalar."
    assert isinstance(value, (int, np.integer)), f"A {torch.typename(value)} instance is not an int scalar."
    return int(value)


def make_scalar_int_tensor(value: IntScalarType) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        assert value.dtype in (np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64), (
            f"NDArray({value.dtype}) is not a int scalar."
        )
        return torch.from_numpy(value)
    assert not isinstance(value, (bool, np.bool_)), "A bool instance is not a float scalar."
    assert isinstance(value, (int, np.integer)), f"A {torch.typename(value)} instance is not an float scalar."
    return torch.tensor(int(value), dtype=torch.int32)
