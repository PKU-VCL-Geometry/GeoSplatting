import inspect
from functools import wraps
from types import MethodType
from typing import Callable, Concatenate, ParamSpec, TypeVar

from rfstudio.utils.decorator import get_wrapped_chains

R = TypeVar('R')
P = ParamSpec('P')
Self = TypeVar('Self')


def inject(
    target: Callable[Concatenate[Self, P], R],
    hook: Callable[Concatenate[Self, P], R],
) -> None:
    assert inspect.ismethod(target) and callable(hook)
    instance = target.__self__

    def wrapper(self: Self, *args, **kwargs) -> R:
        chains = get_wrapped_chains(getattr(self, target.__name__).__func__)
        assert len(chains) > 1
        for f, wf in zip(chains[1:], chains[:-1]):
            if wf is wrapper and f is target.__func__:
                return hook(self, *args, **kwargs)
        raise ValueError("Too magical decorating...")

    setattr(instance, target.__name__, MethodType(wraps(target.__func__)(wrapper), instance))


def inject_once(
    target: Callable[Concatenate[Self, P], R],
    hook: Callable[Concatenate[Self, P], R],
) -> None:

    def wrapper(self: Self, *args, **kwargs) -> None:
        setattr(self, target.__name__, target)
        hook(self, *args, **kwargs)

    inject(target, wrapper)


def wrap_hook(
    target: Callable[Concatenate[Self, P], R],
    hook: Callable[[Self, Callable[[], R]], R],
) -> None:

    def wrapper(self: Self, *args, **kwargs) -> R:
        promise = lambda: target(*args, **kwargs)          # noqa: E731
        return hook(self, promise)

    inject(target, wrapper)


def enter_hook(
    target: Callable[Concatenate[Self, P], R],
    hook: Callable[Concatenate[Self, P], None],
) -> None:

    def wrapper(self: Self, *args, **kwargs) -> R:
        hook(self, *args, **kwargs)
        return target(*args, **kwargs)

    inject(target, wrapper)


def exit_hook(
    target: Callable[Concatenate[Self, P], R],
    hook: Callable[[Self, R], R],
) -> None:

    def wrapper(self: Self, *args, **kwargs) -> R:
        return hook(self, target(*args, **kwargs))

    inject(target, wrapper)
