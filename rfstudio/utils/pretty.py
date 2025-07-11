import pathlib
import runpy
import sys
from string import Formatter
from types import ModuleType
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from rich.cells import cell_len
from rich.console import Console
from rich.markup import RE_TAGS

from rfstudio.utils.scalar import ScalarType, make_scalar_pure


def pretty_traceback(
    *,
    console: Optional[Console] = None,
    suppress: List[ModuleType] = [],
    show_locals: bool = True,
    max_frames: int = 10
) -> None:

    import gsplat
    import nvdiffrast
    import open3d
    import rfviser
    import tinycudann
    import torchvision
    import tyro

    from rfstudio.engine import task
    from rfstudio.utils import tensor_dataclass

    (Console(stderr=True) if console is None else console).print_exception(
        show_locals=show_locals,
        suppress=[
            torch,
            rfviser,
            open3d,
            gsplat,
            np,
            tyro,
            runpy,
            tensor_dataclass,
            task,
            nvdiffrast,
            tinycudann,
            torchvision,
        ] + suppress,
        max_frames=max_frames
    )


def depretty(string: str) -> str:
    return RE_TAGS.sub('', string)


def _markup_stripped_cell_len(line: str) -> int:
    length = cell_len(line)
    for match in RE_TAGS.finditer(line):
        start, end = match.span()
        length -= (end - start)
    return length


def _make_scalar_pretty(value: ScalarType, *, formatter: str) -> str:
    value = make_scalar_pure(value)
    if isinstance(value, bool):
        assert formatter == '', "Cannot formatter a boolean."
        return f'[yellow][i]{"true" if value else "false"}[/i][/yellow]'
    if isinstance(value, int):
        return f'[cyan]{value:{formatter}}[/cyan]'
    if isinstance(value, float):
        abs_val = abs(value)
        if formatter != '':
            return f'[magenta]{value:{formatter}}[/magenta]'
        if abs_val == 0:
            return '[magenta]0.0[/magenta]'
        if abs_val > 100:
            return f'[magenta]{value:.1f}[/magenta]'
        if abs_val > 1:
            return f'[magenta]{value:.3f}[/magenta]'
        if abs_val > 0.0001:
            return f'[magenta]{value:.5f}[/magenta]'
        return f'[magenta]{value:.4e}[/magenta]'
    if isinstance(value, str):
        assert formatter == '', "Cannot formatter a str."
        return f'[green]"{value:{formatter}}"[/green]'
    if isinstance(value, pathlib.Path):
        assert formatter == '', "Cannot formatter a path."
        return f'[bright_yellow][u]{value}[/u][/bright_yellow]'
    raise RuntimeError("Unreachable")


class _Pretty:

    def __matmul__(self, template: str) -> str:
        previous_frame = sys._getframe(1)
        results = []
        spaces = []
        max_lines = 1
        parts = Formatter().parse(template)
        for part in parts:
            literal_text, field_name, format_spec, conversion = part
            assert not conversion, f"Conversion is not allowed here, but get {conversion}."
            if literal_text:
                results.append([literal_text])
                spaces.append(' ' * len(literal_text))
            if not field_name:
                continue
            value = eval(field_name, previous_frame.f_locals, previous_frame.f_globals)
            lines = self.__call__(value, formatter=format_spec).splitlines()
            max_lines = max(len(lines), max_lines)
            results.append(lines)
            spaces.append(' ' * max(_markup_stripped_cell_len(line) for line in lines))
        lines = []
        for i in range(max_lines):
            lines.append(
                ''.join([
                    (
                        (r[i] + ' ' * (len(s) - _markup_stripped_cell_len(r[i])))
                        if i < len(r) else s
                    )
                    for r, s in zip(results, spaces)
                ])
            )
        return '\n'.join(lines)

    def __call__(self, value: Union[Dict[str, ScalarType], ScalarType], *, formatter: str = '') -> str:
        if isinstance(value, dict):
            assert formatter == '', "Cannot manually format a dict."
            max_len = max(len(k) for k in value.keys())
            return '\n'.join([
                k + ' ' * (max_len - len(k)) + ' = ' + _make_scalar_pretty(v, formatter='')
                for k, v in value.items()
            ])
        return _make_scalar_pretty(value, formatter=formatter)


P = _Pretty()
