import inspect
from dataclasses import is_dataclass


class Namespace:

    def __init_subclass__(cls) -> None:
        assert not is_dataclass(cls), "Namespace cannot be dataclass."
        existing_name = set(vars(Namespace).keys()) | { '__annotations__' }
        for key, value in vars(cls).items():
            if key in existing_name:
                continue
            assert not key.startswith(f'_{cls.__name__}__'), (
                "Private member is not allowed in Namespace. "
                f"(`{key.removeprefix(f'_{cls.__name__}__')}`)"
            )
            assert not key.startswith('__'), f"Magic is not allowed in Namespace. (`{key}`)"
            if inspect.isfunction(value):
                assert isinstance(value, staticmethod), (
                    "Only staticmethod is allowed in Namespace. "
                    f"(`{key}`)"
                )

    def __new__(cls, *args, **kwargs) -> None:
        raise RuntimeError("Namespace is a static class and cannot be instantiated.")

