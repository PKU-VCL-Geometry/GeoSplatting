from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from rich.console import RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from rfstudio.utils.pretty import P
from rfstudio.utils.scalar import ScalarType, is_scalar

from .layout import _Layout


@dataclass
class _Table(_Layout):

    variables: Dict[str, ScalarType]
    title: Optional[str] = None

    def _make(self, *, ratio: int = 1) -> RenderableType:
        max_length = max([len(var_name) for var_name in self.variables.keys()]) if self.variables != {} else 0
        text: List[Text] = []
        for var_name, var_value in self.variables.items():
            text.append(Text(var_name, style="bold yellow", overflow='ellipsis'))
            text[-1].align('right', max_length)
            text[-1].append(' = ', style="white").append_text(Text.from_markup(P(var_value)))
        return Layout(Panel(Text('\n').join(text), title=self.title, title_align='left'), ratio=ratio)


@dataclass
class _TableUpdater:

    _target: _Table

    def update(self, **variables: ScalarType) -> None:
        cpu_variables = {}
        for k, v in variables.items():
            assert is_scalar(v), f"Variable {k} with type {v.__class__} is not a valid scalar."
            if isinstance(v, torch.Tensor):
                v = v.cpu()
            cpu_variables[k] = v
        self._target.variables = cpu_variables


@dataclass
class _TableMaker:

    _source: Dict[str, _Table]

    def __getitem__(self, key: str) -> Union[_Table, _TableUpdater]:
        if key not in self._source:
            self._source[key] = _Table({}, title=key)
            return self._source[key]
        return _TableUpdater(self._source[key])
