from typing import Any, Generic, Iterable, Optional, SupportsIndex, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

from rfstudio.utils.typing import (
    FloatLike,
    Indexable,
    IntLike,
    SelfIndexable,
    is_optional,
    remove_optional,
)

T = TypeVar('T')

class StaticChecker(Generic[T]):
    def __init__(self, target: T) -> None:
        pass

def test_runtime_check():
    assert is_optional(Optional[int], int)
    assert is_optional(Union[None, int], int)
    assert is_optional(Union[int, None], int)
    assert is_optional(int | None, int)
    assert not is_optional(int, int)
    assert not is_optional(None, int)
    assert not is_optional(Union[int, str], int)
    assert not is_optional(Any, int)

    assert int is remove_optional(Optional[int])
    assert int is remove_optional(Union[None, int])
    assert int is remove_optional(Union[int, None])
    assert int is remove_optional(int | None)
    assert int is remove_optional(int)
    assert None is remove_optional(None)
    assert Union[int, str] is remove_optional(Union[int, str])
    assert Any is remove_optional(Any)

def test_static_numeric_check():
    npi32 = np.int32(1)
    npf32 = np.float32(1.1)
    npu8 = np.uint8(1.1)
    npb = np.bool_(True)

    StaticChecker[  IntLike    ](1)
    StaticChecker[  IntLike    ](npi32)
    StaticChecker[  IntLike    ](npu8)
    StaticChecker[  IntLike    ](npf32) # type: ignore[arg-type]
    StaticChecker[  IntLike    ](npb)   # type: ignore[arg-type]
    StaticChecker[  IntLike    ](True)
    StaticChecker[  IntLike    ](None)  # type: ignore[arg-type]
    StaticChecker[  IntLike    ]('1')   # type: ignore[arg-type]
    StaticChecker[  IntLike    ](1.1)   # type: ignore[arg-type]
    StaticChecker[  IntLike    ]([])    # type: ignore[arg-type]
    StaticChecker[  IntLike    ](())    # type: ignore[arg-type]

    StaticChecker[  FloatLike  ](1)
    StaticChecker[  FloatLike  ](npi32) # type: ignore[arg-type]
    StaticChecker[  FloatLike  ](npu8)  # type: ignore[arg-type]
    StaticChecker[  FloatLike  ](npf32)
    StaticChecker[  FloatLike  ](npb)   # type: ignore[arg-type]
    StaticChecker[  FloatLike  ](True)
    StaticChecker[  FloatLike  ](None)  # type: ignore[arg-type]
    StaticChecker[  FloatLike  ]('1')   # type: ignore[arg-type]
    StaticChecker[  FloatLike  ](1.1)
    StaticChecker[  FloatLike  ]([])    # type: ignore[arg-type]
    StaticChecker[  FloatLike  ](())    # type: ignore[arg-type]

    StaticChecker[  SupportsIndex             ](1)
    StaticChecker[  SupportsIndex             ](npi32)
    StaticChecker[  Indexable[FloatLike]      ]([1.1])
    StaticChecker[  SelfIndexable[FloatLike]  ]([1.1])
    StaticChecker[  Iterable[FloatLike]       ]([1.1])
    StaticChecker[  Indexable[FloatLike]      ]((1.1,))
    StaticChecker[  SelfIndexable[FloatLike]  ]((1.1,))
    StaticChecker[  Iterable[FloatLike]       ]((1.1,))
    StaticChecker[  Indexable[FloatLike]      ](np.zeros(0))
    StaticChecker[  SelfIndexable[FloatLike]  ](np.zeros(0))
    StaticChecker[  Iterable[FloatLike]       ](np.zeros(0))
    StaticChecker[  Indexable[Tensor]         ](torch.zeros(0))
    StaticChecker[  SelfIndexable[Tensor]     ](torch.zeros(0))
    StaticChecker[  Iterable[Tensor]          ](torch.zeros(0))

class Test:
    def foo1(self): pass
    def foo2(self: 'Test') -> None: pass
    def foo3() -> None: pass # type: ignore[misc]
    def foo4(self, a: int) -> int: return a
    @staticmethod
    def foo5(self: 'Test', a: int) -> int: return a


if __name__ == '__main__':
    test_runtime_check()
    test_static_numeric_check()
