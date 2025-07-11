from __future__ import annotations

from typing import Any, Optional, TypeVar

T = TypeVar('T')


class _OptionalWrapper:

    def __init__(self, target: Any, /) -> None:
        self._target = target

    def __getattr__(self, name: str) -> Optional[Any]:
        target = super().__getattribute__('_target')
        return None if target is None else getattr(target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, '_target') or name != '_target':
            raise NotImplementedError
        super().__setattr__(name, value)

    def __bool__(self) -> bool:
        return super().__getattribute__('_target') is not None


def get_optional_wrapper(target: Optional[T]) -> T:
    return _OptionalWrapper(target)
