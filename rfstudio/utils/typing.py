from __future__ import annotations

from types import EllipsisType, NoneType, UnionType
from typing import (
    Any,
    Generic,
    Iterator,
    Optional,
    Protocol,
    Sequence,
    Sized,
    SupportsIndex,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
    runtime_checkable,
)

from numpy import dtype, floating, integer
from numpy._typing import _ArrayLikeInt_co, _SupportsArray


@runtime_checkable
class _OptionalProtocol(Protocol):
    __args__: Union[Tuple[None, Any], Tuple[Any, None]]


def is_optional(source: Any, target: Any) -> bool:
    if isinstance(source, UnionType):
        return source == Optional[target]
    return (source is Optional[target]) or (source is Union[None, target]) or (source is Union[target, None])


def remove_optional(t: Any) -> Any:
    if isinstance(t, _OptionalProtocol):
        if t.__args__[0] is NoneType:
            arg = t.__args__[1]
        elif t.__args__[1] is NoneType:
            arg = t.__args__[0]
        else:
            return t
        if is_optional(t, arg):
            return arg
    return t


def issubclass_optional(source: Any, target: Type[Any]) -> bool:
    return issubclass(remove_optional(source), target)


def get_generic_args(cls: Type[Any], generic_base: Type[Any]) -> Tuple[Type[Any], ...]:
    for base in cls.__orig_bases__:
        origin = get_origin(base)
        if origin is None or not issubclass(origin, generic_base):
            continue
        return get_args(base)
    return ()


T = TypeVar('T', covariant=True)


class Indexable(Protocol, Sized, Generic[T]):

    @overload
    def __getitem__(self, index: Union[IntLike, EllipsisType]) -> T:
        ...

    @overload
    def __getitem__(self, indices: IntArrayLike) -> Indexable[T]:
        ...

    def __iter__(self) -> Iterator[T]:
        ...


class SelfIndexable(Protocol, Sized, Generic[T]):

    def __getitem__(self: T, index: Any) -> T:
        ...


Indexer: TypeAlias = Union[
    None,
    slice,
    EllipsisType,
    SupportsIndex,
    _ArrayLikeInt_co,
    Tuple[Union[None, slice, EllipsisType, _ArrayLikeInt_co, SupportsIndex], ...],
]

IntLike: TypeAlias = Union[int, integer[Any]]

FloatLike: TypeAlias = Union[float, floating[Any]]

IntArrayLike: TypeAlias = Union[
    _SupportsArray[dtype[integer[Any]]],
    Sequence[IntLike],
]

FloatArrayLike: TypeAlias = Union[
    _SupportsArray[dtype[floating[Any]]],
    Sequence[FloatLike],
]
