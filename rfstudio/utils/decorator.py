from functools import wraps
from inspect import isfunction, isgenerator
from typing import Any, Callable, List, Optional, TypeVar, Union, overload

import numpy as np
import torch

from rfstudio.utils.tensor_dataclass import TensorDataclass


def get_wrapped_chains(fn: Callable[..., Any]) -> List[Callable[..., Any]]:
    lst = [fn]
    while hasattr(fn, '__wrapped__'):
        fn = fn.__wrapped__
        lst.append(fn)
    return lst


def check_for_decorated_method(self, fn: Callable[..., Any], wrapper: Callable[..., Any]) -> bool:
    assert not isinstance(self, type)
    class_method = getattr(self.__class__, fn.__name__, None)
    if isinstance(class_method, property):
        return class_method.fget is wrapper
    return isfunction(class_method) and class_method is wrapper


F = TypeVar('F', bound=Callable[..., Any])
P = TypeVar('P')


@overload
def lazy(fn: F, /, *, manually_decide: bool = False) -> F:
    ...


@overload
def lazy(fn: None = None, /, *, manually_decide: bool = False) -> Callable[[F], F]:
    ...


def lazy(fn: Optional[F] = None, /, *, manually_decide: bool = False) -> Union[F, Callable[[F], F]]:
    if fn is None:
        return lambda x: lazy(x, manually_decide=manually_decide)

    name = '__cache_fn__' + fn.__name__

    if manually_decide:

        @wraps(fn)
        def manually_cached_fn(self, *args, **kwargs) -> Any:
            closure: F = fn
            assert check_for_decorated_method(self, closure, manually_cached_fn)
            g = closure(self, *args, **kwargs)
            assert isgenerator(g), f"Function {closure.__name__} must yield twice when manually_decide is set to True"
            condition = next(g)
            old_condition = getattr(self, name)[0] if hasattr(self, name) else None
            if condition == old_condition:
                return getattr(self, name)[1]
            result = next(g)
            setattr(self, name, (condition, result))
            return result

        return manually_cached_fn

    else:

        @wraps(fn)
        def cached_fn(self, *args, **kwargs) -> Any:
            closure: F = fn
            assert check_for_decorated_method(self, closure, cached_fn)
            if hasattr(self, name):
                return getattr(self, name)
            result = closure(self, *args, **kwargs)
            setattr(self, name, result)
            return result

        return cached_fn


def lazy_property(fn: Callable[..., P], /) -> property:
    return property(lazy(fn))


@overload
def batchify(
    fn: F,
    /,
    *,
    prop: str = 'batch_size',
    batch_size: Optional[int] = None,
    callback: Optional[str] = None,
) -> F:
    ...


@overload
def batchify(
    fn: None = None,
    /,
    *,
    prop: str = 'batch_size',
    batch_size: Optional[int] = None,
    callback: Optional[str] = None,
) -> Callable[[F], F]:
    ...


def batchify(
    fn: Optional[F] = None,
    /,
    *,
    prop: str = 'batch_size',
    batch_size: Optional[int] = None,
    callback: Optional[str] = None,
) -> Union[Callable[[F], F], F]:

    if fn is None:
        return lambda x: batchify(x, prop=prop, batch_size=batch_size, callback=callback)

    @wraps(fn)
    def wrapper(self, *args, **kwargs) -> Any:
        closure: F = fn
        assert check_for_decorated_method(self, closure, wrapper)
        num_batches: Optional[int] = None
        for arg_lst in (args, kwargs.values()):
            for arg in arg_lst:
                assert isinstance(arg, (torch.Tensor, TensorDataclass)), \
                    "All arguments should be tensor or tensor dataclass"
                arg_num_batches = np.prod(arg.shape) if isinstance(arg, TensorDataclass) else arg.shape[0]
                if num_batches is None:
                    num_batches = arg_num_batches
                else:
                    assert num_batches == arg_num_batches, "Shape[0] of arguments must be same"
        assert num_batches is not None, "At least one tensor should be given"

        results = []
        effective_batch_size = getattr(self, prop) if batch_size is None else batch_size
        if callback is not None:
            getattr(self, callback)(0, num_batches)
        flatten_args = [(arg.flatten() if isinstance(arg, TensorDataclass) else arg) for arg in args]
        for i in range(0, num_batches, effective_batch_size):
            s = slice(i, i + effective_batch_size)
            results.append(closure(
                self,
                *tuple(arg[s] for arg in flatten_args),
                **{ k: v[s] for k, v in kwargs.items() },
            ))
            if callback is not None:
                getattr(self, callback)(min(i + effective_batch_size, num_batches), num_batches)

        if isinstance(results[0], tuple):
            tpl = []
            for i in range(len(results[0])):
                items = [x[i] for x in results]
                if isinstance(items[0], TensorDataclass):
                    items = type(items[0]).cat(items, dim=0)
                else:
                    items = torch.cat(items, dim=0)
                tpl.append(
                    items.view(*args[0].shape, *items.shape[1:])
                    if isinstance(args[0], TensorDataclass)
                    else items
                )
            return tuple(tpl)

        if isinstance(results[0], TensorDataclass):
            results = type(results[0]).cat(results, dim=0)
        else:
            results = torch.cat(results, dim=0)
        return results.view(*args[0].shape, *results.shape[1:]) if isinstance(args[0], TensorDataclass) else results

    return wrapper


def chains(fn: Callable[..., Callable[..., Any]]) -> Any:

    @wraps(fn)
    def wrapper(self, *args, **kwargs) -> Any:
        assert check_for_decorated_method(self, fn, wrapper)

        func: Callable[..., Any] = fn(self, *args, **kwargs)

        class PseduoCaller:
            pass

        @wraps(func)
        def pseduo_call(this: Any, *args, **kwargs) -> Any:
            return func(self, *args, **kwargs)

        setattr(PseduoCaller, func.__name__, pseduo_call)

        return PseduoCaller()

    return wrapper
