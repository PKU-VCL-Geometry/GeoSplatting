from __future__ import annotations

import pathlib
import sys
import typing
from importlib.util import module_from_spec, spec_from_file_location


def import_from_file_module(
    file_path: typing.Union[str, pathlib.Path],
    names: typing.List[str],
) -> typing.List[typing.Any]:
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)
    assert file_path.exists() and file_path.is_file()
    module_name = "_file_module" + str(file_path)
    spec = spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for name in names:
        assert hasattr(module, name), (
            f"Fail to import {name} from {file_path} "
            f"""(probably due to 'if __name__ == "__main__"' condition)"""
        )
    return [getattr(module, name) for name in names]


def is_from_file_module(module_name: str) -> bool:
    return module_name.startswith("_file_module")


def get_module_file(module_name: str) -> pathlib.Path:
    assert is_from_file_module(module_name)
    path = pathlib.Path(module_name[len("_file_module"):])
    assert path.exists()
    return path
