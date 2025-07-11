from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import NoneType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
from torch import Tensor

from rfstudio.utils.typing import Indexer, remove_optional


def _convert_args_to_tuple_shape(args: Tuple[Union[Tuple[int, ...], torch.Size, int], ...]) -> Tuple[int, ...]:
    assert len(args) > 0, "Must specify shape"
    if isinstance(args[0], (torch.Size, tuple)):
        assert len(args) == 1
        return tuple(args[0])
    return args


def _process_unspecified_dimension(shape: Tuple[int, ...], num_elements: int) -> Tuple[int, ...]:
    size_of_shape = int(np.prod(shape).item())
    if size_of_shape == num_elements:
        return shape
    assert len([s for s in shape if s == -1]) == 1, f"Invalid shape: {shape}"
    assert size_of_shape < 0 and num_elements % -size_of_shape == 0
    return tuple((num_elements // -size_of_shape if s == -1 else s) for s in shape)


class _DynamicSize:

    def set_name(self, value: str) -> None:
        self._name = value

    def get_name(self) -> str:
        assert self._name is not None
        return self._name

    @staticmethod
    def deduce(
        real_shape: Tuple[int, ...],
        annotated_shape: Tuple[Union[_DynamicSize, int], ...],
        dynamic_dict: Dict[str, int]
    ) -> Tuple[int, ...]:
        assert len(real_shape) == len(annotated_shape)
        return tuple(
            dynamic_dict.setdefault(annotated.get_name(), real) if isinstance(annotated, _DynamicSize) else annotated
            for real, annotated in zip(real_shape, annotated_shape)
        )

    @staticmethod
    def extract_and_assert(
        annotated_shape: Tuple[Union[_DynamicSize, int], ...],
        dynamic_dict: Dict[str, int],
    ) -> Tuple[int, ...]:
        shape = tuple(
            dynamic_dict.get(annotated.get_name(), annotated)
            if isinstance(annotated, _DynamicSize) else annotated
            for annotated in annotated_shape
        )
        not_deduced = _DynamicSize.names_not_deduced(shape)
        assert not_deduced == [], f"All dynamic sizes should be manually specified, but missing {not_deduced[0]}"
        return shape

    @staticmethod
    def names_not_deduced(shape: Tuple[Union[_DynamicSize, int], ...]) -> List[str]:
        if len(shape) == 0:
            return []
        return [s.get_name() for s in shape if isinstance(s, _DynamicSize)]


class _SizeType:

    @property
    def Dynamic(self) -> int:
        return cast(int, _DynamicSize())

Size = _SizeType()


class _TensorDataclassField(NamedTuple):
    name: str
    dtype: Union[torch.dtype, Type[TensorDataclass]]
    shape: Tuple[Union[int, _DynamicSize], ...]
    fixed: bool
    optional: bool
    trainable: bool


T = TypeVar('T', bound='TensorDataclass')


class _TrainableProxy(Generic[T]):
    def __init__(self, cls: Type[T]) -> None:
        self._cls = cls

    def __getitem__(self, shape: Any) -> T:
        return cast(T, (self._cls, shape, True))


@dataclasses.dataclass
class TensorDataclass:

    if TYPE_CHECKING:
        class Trainable:
            @classmethod
            def __class_getitem__(self, shape: Any) -> TensorDataclass:
                pass

    def __init_subclass__(cls, **kwargs: Any) -> None:
        setattr(cls, 'Trainable', _TrainableProxy(cls))

    @classmethod
    def get_td_fields(cls: Type[T]) -> List[_TensorDataclassField]:
        if cls is TensorDataclass:
            return []
        if hasattr(cls, '_td_fields'):
            cls_for_check, result = getattr(cls, '_td_fields')
            if cls is cls_for_check:
                return result
        fields: List[_TensorDataclassField] = list(cls.__mro__[1].get_td_fields())
        for field_name, field_annotation in inspect.get_annotations(cls, eval_str=True).items():
            assert (
                remove_optional(field_annotation) is torch.Tensor or
                issubclass(remove_optional(field_annotation), TensorDataclass) or
                (field_annotation is int and isinstance(getattr(cls, field_name, None), _DynamicSize))
            ), f"Unknown field: [{field_name}:{repr(field_annotation)}]"

            is_fixed = False
            is_optional = False
            if field_annotation is int:
                getattr(cls, field_name).set_name(field_name)
            else:
                real_type = remove_optional(field_annotation)
                if field_annotation is not real_type:
                    is_optional = True
                assert real_type is torch.Tensor or issubclass(real_type, TensorDataclass)
                field_annotation, field_shape, trainable = getattr(cls, field_name)
                if not isinstance(field_shape, tuple):
                    field_shape = (field_shape, )
                if field_shape[0] is Ellipsis:
                    field_shape = field_shape[1:]
                else:
                    is_fixed = True
                fields.append(_TensorDataclassField(
                    name=field_name,
                    dtype=field_annotation,
                    shape=field_shape,
                    fixed=is_fixed,
                    optional=is_optional,
                    trainable=trainable,
                ))

        assert any([not f.optional for f in fields]), "At least one field should be non-optional"
        setattr(cls, '_td_fields', (cls, fields))
        return fields

    def __post_init__(self: T) -> None:
        dynamic_dict: Dict[str, int]
        data_annotations: List[_TensorDataclassField]
        lazy_shape_checks: List[int]

        cls = self.__class__
        fields = cls.get_td_fields()
        dynamic_dict = {}
        data_annotations = []
        lazy_shape_checks = []
        shape = None
        device = None
        kwargs = {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
        }
        for idx, field in enumerate(fields):
            tensor = kwargs[field.name]
            if field.optional and (tensor is None or tensor is getattr(cls, field.name)):
                setattr(self, field.name, None)
                lazy_shape_checks.append(idx)
                data_annotations.append(field)
                continue
            if tensor is getattr(cls, field.name):
                raise ValueError(f"Missing field: {field.name}")
            if not isinstance(tensor, (torch.Tensor, TensorDataclass)):
                raise ValueError(f"Not a tensor: get {type(tensor)} instead")
            if device is None:
                device = tensor.device
            else:
                assert device == tensor.device, f"Inconsistent device @ {field.name}: {device} and {tensor.device}"
            if field.fixed:
                field_shape = _DynamicSize.deduce(tensor.shape, field.shape, dynamic_dict)
                assert (tensor.shape == field_shape), (
                    f"Incorrect field shape @ {field.name}: "
                    f"{tensor.shape} but expected {field_shape}"
                )
            else:
                field_shape = field.shape
                if len(field_shape) > 0:
                    field_shape = _DynamicSize.deduce(tensor.shape[-len(field_shape):], field_shape, dynamic_dict)
                    assert (tensor.shape[-len(field_shape):] == field_shape), (
                        f"Incorrect field shape @ {field.name}: "
                        f"{tensor.shape[-len(field_shape):]} but expected {field_shape}"
                    )
                else:
                    field_shape = ()
                deduced_shape = tensor.shape[:-len(field_shape)] if len(field_shape) > 0 else tensor.shape
                if shape is None:
                    shape = deduced_shape
                else:
                    assert shape == deduced_shape, f"Inconsistent shape @ {field.name}: {shape} and {deduced_shape}"

            data_annotations.append(
                _TensorDataclassField(
                    name=field.name,
                    dtype=field.dtype,
                    shape=field_shape,
                    fixed=field.fixed,
                    optional=field.optional,
                    trainable=field.trainable,
                )
            )
        for idx in lazy_shape_checks:
            field = data_annotations[idx]
            field_shape = _DynamicSize.extract_and_assert(field.shape, dynamic_dict)
            data_annotations[idx] = _TensorDataclassField(
                name=field.name,
                dtype=field.dtype,
                shape=field_shape,
                fixed=field.fixed,
                optional=field.optional,
                trainable=field.trainable,
            )
        for dynamic_name, value in dynamic_dict.items():
            setattr(self, dynamic_name, value)
        assert device is not None
        self._shape = shape
        self._device = device
        self._data_annotations = data_annotations

    @classmethod
    def _external_init(
        cls: Type[T],
        size: Union[int, Tuple[int, ...]],
        device: Optional[torch.device] = None,
        init_fn: Callable[..., torch.Tensor] = torch.zeros,
        **kwargs: int
    ) -> T:
        kw_fields: Dict[str, Optional[TensorLike]]

        if not isinstance(size, (torch.Size, tuple)):
            size = (size, )
        kw_fields = {}
        for field in cls.get_td_fields():
            if field.optional:
                kw_fields[field.name] = None
                continue
            field_shape = _DynamicSize.extract_and_assert(field.shape, kwargs)
            if isinstance(field.dtype, torch.dtype):
                new_shape = field_shape if field.fixed else (*size, *field_shape)
                kw_fields[field.name] = init_fn(*new_shape, dtype=field.dtype, device=device)
            else:
                assert issubclass(field.dtype, TensorDataclass), f"Unsupported nested type: {field.dtype}"
                new_shape = field_shape if field.fixed else (*size, *field_shape)
                kw_fields[field.name] = field.dtype._external_init(
                    size=new_shape,
                    device=device,
                    init_fn=init_fn,
                    **kwargs,
                )
        return cls(**kw_fields, **{ k: v for k, v in kwargs.items() if hasattr(cls, k) })

    def _recursively_map_shaped(
        self: T,
        fn: Callable[[TensorLike, Tuple[int, ...]], TensorLike],
    ) -> T:
        kwargs: Dict[str, Any] = {}
        for field in self._data_annotations:
            field_value = getattr(self, field.name)
            if field_value is not None and not field.fixed:
                kwargs[field.name] = fn(field_value, field.shape)
            else:
                kwargs[field.name] = field_value
        return self.__class__(**kwargs)

    def _recursively_map_nonempty(
        self: T,
        fn: Callable[[TensorLike], TensorLike],
    ) -> T:
        kwargs: Dict[str, Any] = {}
        for field in self._data_annotations:
            field_value = getattr(self, field.name)
            if field_value is not None:
                kwargs[field.name] = fn(field_value)
            else:
                kwargs[field.name] = field_value
        return self.__class__(**kwargs)

    def _recursively_map_empty(
        self: T,
        fn: Callable[[str], Optional[TensorLike]],
    ) -> T:
        kwargs: Dict[str, Any] = {}
        for field in self._data_annotations:
            field_value = getattr(self, field.name)
            if field_value is None:
                assert field.optional
                kwargs[field.name] = fn(field.name)
            else:
                kwargs[field.name] = field_value
        return self.__class__(**kwargs)

    @classmethod
    def __class_getitem__(cls: Type[T], shape: Any) -> T:
        return cast(T, (cls, shape, False))

    def __getitem__(self: T, indices: Indexer) -> T:
        assert self._shape is not None, (
            f"{self.__class__.__name__} cannot be indexed "
            "because all the fields have fixed shape."
        )
        if not isinstance(indices, tuple):
            if hasattr(indices, '__array__') or isinstance(indices, Sequence):
                return self._recursively_map_shaped(lambda t, s: t[indices])
            indices = (indices, )
        return self._recursively_map_shaped(lambda t, s: t[indices + (slice(None), ) * len(s)])

    def __iter__(self: T) -> Iterator[T]:
        assert self._shape is not None, (
            f"{self.__class__.__name__} cannot be iterated "
            "because all the fields have fixed shape."
        )
        if self.shape == () or self.shape[0] == 0:
            raise ValueError(self.shape)
        for i in range(self.shape[0]):
            yield self[i]

    def reshape(self: T, *shape: int) -> T:
        assert self._shape is not None, (
            f"{self.__class__.__name__} cannot be reshaped "
            "because all the fields have fixed shape."
        )
        shape = _convert_args_to_tuple_shape(shape)
        shape = _process_unspecified_dimension(shape, int(np.prod(self._shape).item()))
        return self._recursively_map_shaped(lambda t, s: t.reshape(*shape, *s))

    def view(self: T, *shape: int) -> T:
        assert self._shape is not None, (
            f"{self.__class__.__name__} cannot be viewed "
            "because all the fields have fixed shape."
        )
        shape = _convert_args_to_tuple_shape(shape)
        shape = _process_unspecified_dimension(shape, int(np.prod(self._shape).item()))
        return self._recursively_map_shaped(lambda t, s: t.view(*shape, *s))

    def to(self: T, device: torch.device) -> T:
        return self._recursively_map_nonempty(lambda t: t.to(device))

    def flatten(self: T) -> T:
        assert self._shape is not None, (
            f"{self.__class__.__name__} cannot be flatten "
            "because all the fields have fixed shape."
        )
        return self.view(-1)

    def contiguous(self: T) -> T:
        return self._recursively_map_nonempty(lambda t: t.contiguous())

    def cpu(self: T) -> T:
        return self._recursively_map_nonempty(lambda t: t.cpu())

    @property
    def is_cuda(self) -> bool:
        return self._device != torch.device('cpu')

    def cuda(self: T) -> T:
        return self._recursively_map_nonempty(lambda t: t.cuda())

    def detach(self: T) -> T:
        return self._recursively_map_nonempty(lambda t: t.detach())

    def clone(self: T) -> T:
        return self._recursively_map_nonempty(lambda t: t.clone())

    def as_dict(self) -> Dict[str, Union[Tensor, int, NoneType]]:
        kwargs: Dict[str, Union[Tensor, int, NoneType]] = {}
        for field in self._data_annotations:
            field_value = getattr(self, field.name)
            assert not isinstance(field_value, TensorDataclass), "Nested structure is not allowed here."
            kwargs[field.name] = field_value
        return kwargs

    def serialize(self, path: Path) -> None:
        torch.save(self.detach().cpu().as_dict(), path)

    @classmethod
    def deserialize(cls: Type[T], path: Path) -> T:
        kwargs = torch.load(path, map_location='cpu')
        return cls(**kwargs)

    @property
    def requires_grad(self) -> bool:
        for field in self._data_annotations:
            if not field.trainable:
                continue
            field_value = getattr(self, field.name)
            if field_value.requires_grad:
                return True
        return False

    def requires_grad_(self: T, mode: bool = True) -> T:
        for field in self._data_annotations:
            field_value = getattr(self, field.name)
            if field_value is not None and field.trainable:
                field_value.requires_grad_(mode)
        return self

    def expand(self: T, *shape: int) -> T:
        assert self._shape is not None, (
            f"{self.__class__.__name__} cannot be expanded "
            "because all the fields have fixed shape."
        )
        shape = _convert_args_to_tuple_shape(shape)
        return self._recursively_map_shaped(lambda t, s: t.expand(*shape, *s))

    def __setitem__(self, indices: Any, value: Any) -> NoReturn:
        raise RuntimeError("Index assignment is not supported")

    def __len__(self) -> int:
        assert self._shape is not None, (
            f"{self.__class__.__name__} cannot apply len() "
            "because all the fields have fixed shape."
        )
        return self._shape[0]

    @property
    def shape(self) -> Tuple[int, ...]:
        return () if self._shape is None else self._shape

    @property
    def device(self) -> torch.device:
        return self._device

    def size(self, dim: Optional[int] = None) -> Union[int, Tuple[int, ...]]:
        if dim is None:
            if len(self.shape) == 0:
                return (1, )
            return self.shape
        return self.shape[dim]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def length(self) -> int:
        assert self._shape is not None, (
            f"{self.__class__.__name__} has no length "
            "because all the fields have fixed shape."
        )
        if len(self.shape) == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def annotate(self: T, **kwargs: TensorLike) -> T:
        return self._recursively_map_empty(lambda n: kwargs.get(n, None))

    def annotate_(self: T, **kwargs: TensorLike) -> T:
        dummy: T = self.annotate(**kwargs)
        self.swap_(dummy)
        return self

    def replace(self: T, **kwargs: Optional[TensorLike]) -> T:
        for field in self._data_annotations:
            if field.name in kwargs:
                continue
            kwargs[field.name] = getattr(self, field.name)
        return self.__class__(**kwargs)

    def replace_(self: T, **kwargs: Optional[TensorLike]) -> T:
        dummy = self.replace(**kwargs)
        self.swap_(dummy)
        return self

    def swap_(self: T, rhs: T) -> None:
        assert isinstance(rhs, type(self))
        inner_attrs = ['_device', '_shape', '_data_annotations']
        for key in inner_attrs + [f.name for f in dataclasses.fields(self)]:
            self_val, rhs_val = getattr(self, key), getattr(rhs, key)
            object.__setattr__(self, key, rhs_val)
            object.__setattr__(rhs, key, self_val)

    def __setattr__(self, name: str, value: Any, /) -> None:
        if hasattr(self, '_data_annotations'):
            raise RuntimeError("Attribute assignment is not supported")
        super().__setattr__(name, value)

    @classmethod
    def zeros(
        cls: Type[T],
        size: Union[int, Tuple[int, ...]],
        *,
        device: Optional[torch.device] = None,
        **kwargs: int,
    ) -> T:
        return cls._external_init(size, device, torch.zeros, **kwargs)

    @classmethod
    def ones(
        cls: Type[T],
        size: Union[int, Tuple[int, ...]],
        *,
        device: Optional[torch.device] = None,
        **kwargs: int,
    ) -> T:
        return cls._external_init(size, device, torch.ones, **kwargs)

    @classmethod
    def empty(
        cls: Type[T],
        size: Union[int, Tuple[int, ...]],
        *,
        device: Optional[torch.device] = None,
        **kwargs: int,
    ) -> T:
        return cls._external_init(size, device, torch.empty, **kwargs)

    @classmethod
    def _batch_operate(
        cls: Type[T],
        tensors: Iterable[T],
        dim: int,
        batch_operate_fn: Callable[..., torch.Tensor],
    ) -> T:
        kwargs: Dict[str, Optional[TensorLike]]

        tpl = tuple(tensors)
        assert len(tpl) > 0, "Empty input"
        if dim < 0:
            dim += tpl[0].ndim
            assert dim >= 0, f"Invalid dim: {dim - tpl[0].ndim}"
        kwargs = {}
        for field in tpl[0]._data_annotations:
            field_values = tuple(getattr(t, field.name) for t in tpl)
            if None in field_values:
                assert field.optional
                assert all([v is None for v in field_values]), \
                    "Optional fields must either all be None, or all have values"
                kwargs[field.name] = None
            elif field.fixed:
                assert all([(field_values[0] == v).all().item() for v in field_values]), \
                    "Broadcasting fields must have same values"
                kwargs[field.name] = field_values[0]
            elif isinstance(field.dtype, torch.dtype):
                kwargs[field.name] = batch_operate_fn(field_values, dim=dim)
            else:
                assert issubclass(field.dtype, TensorDataclass)
                kwargs[field.name] = field.dtype._batch_operate(field_values, dim, batch_operate_fn)
        return cls(**kwargs)

    @classmethod
    def cat(cls: Type[T], tensors: Iterable[T], *, dim: int) -> T:
        return cls._batch_operate(tensors, dim, torch.cat)

    @classmethod
    def stack(cls: Type[T], tensors: Iterable[T], *, dim: int) -> T:
        return cls._batch_operate(tensors, dim, torch.stack)


class _TrainableTensorDescriptor:

    def __init__(self, dtype: torch.dtype) -> None:
        self._dtype = dtype

    def __getitem__(self, shape: Any) -> torch.Tensor:
        return cast(torch.Tensor, (self._dtype, shape, True))

class _TensorDescriptor:

    def __init__(self, dtype: torch.dtype) -> None:
        self._dtype = dtype

    def __getitem__(self, shape: Any) -> torch.Tensor:
        return cast(torch.Tensor, (self._dtype, shape, False))

    @property
    def Trainable(self) -> _TrainableTensorDescriptor:
        return _TrainableTensorDescriptor(self._dtype)


TensorLike: TypeAlias = Union[TensorDataclass, torch.Tensor]

Float = _TensorDescriptor(torch.float)
Double = _TensorDescriptor(torch.double)
Int = _TensorDescriptor(torch.int)
Long = _TensorDescriptor(torch.long)
Bool = _TensorDescriptor(torch.bool)
Half = _TensorDescriptor(torch.half)
Byte = _TensorDescriptor(torch.uint8)
