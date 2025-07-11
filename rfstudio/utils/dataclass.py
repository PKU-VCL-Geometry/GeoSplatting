"""

"""

from __future__ import annotations

from dataclasses import Field, fields, is_dataclass
from pathlib import Path, PosixPath
from typing import Any, ClassVar, Dict, List, Protocol, Type, runtime_checkable

import __main__
from rfstudio.utils.dynamic_import import (
    get_module_file,
    import_from_file_module,
    is_from_file_module,
)


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def _dump_as_pycode(data: DataclassInstance, imports: List[Type[Any]], lines: List[str], indents: int) -> None:
    prefix = ' ' * (indents * 4)
    object_type = type(data)
    if object_type not in imports:
        imports.append(object_type)
    lines[-1] += object_type.__name__ + '('
    first = True
    for f in fields(data):
        if first:
            first = False
        else:
            lines[-1] += ','
        val = getattr(data, f.name)
        lines.append(prefix + ' ' * 4 + f.name + ' = ')
        if any([
            val is None,
            isinstance(val, (str, int, float, PosixPath, bool)),
            (isinstance(val, (list, tuple)) and all([True] + [isinstance(x, int) for x in val])),
        ]):
            lines[-1] += repr(val)
        else:
            assert is_dataclass(val), (
                f"Field [{object_type.__name__}.{f.name} : "
                f"{type(data).__name__}] cannot be dumped as python code."
            )
            _dump_as_pycode(val, imports, lines, indents + 1)
    lines.append(prefix + ')')


def load_dataclass_from_str(source: str) -> DataclassInstance:
    exec(source, locals(), locals())
    dataclass = locals().get('dumped', None)
    assert isinstance(dataclass, DataclassInstance)
    return dataclass


def load_dataclass(path: Path) -> DataclassInstance:
    assert path.exists()
    dataclass = import_from_file_module(path, names=['dumped'])[0]
    assert isinstance(dataclass, DataclassInstance)
    return dataclass


def dump_dataclass_as_str(target: DataclassInstance) -> str:
    assert is_dataclass(target)

    imports = [PosixPath]
    lines = ['dumped = ']
    _dump_as_pycode(target, imports, lines, indents=0)
    import_dict: Dict[str, List[str]] = {}
    unique_dict: Dict[str, str] = {}
    for object_type in imports:
        lst = import_dict.get(object_type.__module__)
        if lst is None:
            import_dict[object_type.__module__] = []
        import_dict[object_type.__module__].append(object_type.__name__)
        assert object_type.__name__ not in unique_dict, (
            f"Fail to dump {type(target).__name__} "
            f"because {object_type.__name__} is ambiguous "
            f"(from {unique_dict[object_type.__name__]} and from {object_type.__module__})"
        )
        unique_dict[object_type.__name__] = object_type.__module__
    import_lines = []
    dynamic_imports = []
    for module_name, object_names in import_dict.items():
        if module_name != '__main__' and not is_from_file_module(module_name):
            import_lines.append(f'from {module_name} import ' + ', '.join(object_names))
        else:
            dynamic_imports.append(module_name)

    for module_name in dynamic_imports:
        object_names = import_dict[module_name]
        module_file = (
            Path(__main__.__file__)
            if module_name == '__main__'
            else get_module_file(module_name)
        )
        import_lines += [
            f'from {import_from_file_module.__module__} import {import_from_file_module.__name__}',
            ', '.join(object_names) + (
                f', *_ = {import_from_file_module.__name__}'
                f'({repr(module_file)}, names={repr(object_names)})'
            )
        ]
    return '\n'.join(import_lines + [''] + lines + [''])


def dump_dataclass(path: Path, target: DataclassInstance) -> None:
    assert path.parent.exists()
    with open(path, 'w') as f:
        f.write(dump_dataclass_as_str(target))


def has_no_missing_field(t: DataclassInstance) -> bool:
    for f in fields(t):
        if getattr(t, f.name) is Ellipsis:
            return False
    return True
