from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import plotext as plt
import torch
from rich.ansi import AnsiDecoder
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.jupyter import JupyterMixin
from rich.layout import Layout
from rich.panel import Panel

from rfstudio.utils.scalar import FloatScalarType
from rfstudio.utils.tensor_container import TensorList

from .layout import _Layout


class _PlotextMixin(JupyterMixin):

    def __init__(self, plot_fn: Callable[[int, int], str]) -> None:
        self._decoder = AnsiDecoder()
        self._plot_fn = plot_fn

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        canvas = self._plot_fn(
            width=options.max_width or console.width,
            height=options.height or console.height,
        )
        yield Group(*self._decoder.decode(canvas))


@dataclass
class _Plot(_Layout):

    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xticks: Union[Iterable[float], Tuple[Iterable[float], Iterable[str]], None] = None
    yticks: Union[Iterable[float], Tuple[Iterable[float], Iterable[str]], None] = None
    xlim: Tuple[Union[int, float, None], Union[int, float, None]] = (None, None)
    ylim: Tuple[Union[int, float, None], Union[int, float, None]] = (None, None)
    title: Optional[str] = None
    resolution: int = 256
    moving_factor: float = 0.99

    def __post_init__(self) -> None:
        self.x = TensorList()
        self.y = TensorList()
        self.last = None
        self.last_size = None

    def _insert(
        self,
        x: FloatScalarType,
        y: Union[FloatScalarType, Dict[str, FloatScalarType]],
    ) -> None:
        self.last = None
        if isinstance(y, dict):
            if isinstance(self.y, TensorList) and not self.y:
                self.y = {}
            assert isinstance(self.y, dict)
            for key, value in y.items():
                index_lst, value_lst = self.y.setdefault(key, (TensorList(), TensorList()))
                index_lst.append(x)
                if len(value_lst) == 0:
                    value_lst.append(value)
                else:
                    value_lst.append(value_lst[-1] * self.moving_factor + (1 - self.moving_factor) * value)
        else:
            assert isinstance(self.y, TensorList)
            if len(self.y) == 0:
                self.y.append(y)
            else:
                self.y.append(self.y[-1] * self.moving_factor + (1 - self.moving_factor) * y)
        self.x.append(x)

    def _make(self, *, ratio: int = 1) -> RenderableType:
        return Layout(
            Panel(
                _PlotextMixin(plot_fn=self._make_plot),
                title=self.title,
                title_align='left',
            ),
            ratio=ratio,
        )

    def _make_plot(self, width: int, height: int) -> str:
        if self.last is not None and self.last_size == (width, height):
            return self.last
        plt.clf()
        plt.plotsize(width, height)
        plt.theme('pro')
        min_y = None
        max_y = None
        if isinstance(self.y, dict):
            for y_name, (y_index, y_value) in self.y.items():
                y_index = y_index.as_tensor()
                y_value = y_value.as_tensor()
                if len(y_index) > 2 * self.resolution:
                    y_index = torch.nn.functional.interpolate(
                        y_index[None, None, :],
                        size=self.resolution,
                    ).flatten()
                    y_value = torch.nn.functional.interpolate(
                        y_value[None, None, :],
                        size=self.resolution,
                    ).flatten()
                if self.yticks is None:
                    min_y = y_value.min() if min_y is None else min(y_value.min(), min_y)
                    max_y = y_value.max() if max_y is None else max(y_value.max(), max_y)
                plt.plot(y_index, y_value, marker="hd", label=y_name)
        else:
            x, y = self.x.as_tensor(), self.y.as_tensor()
            if len(x) > 2 * self.resolution:
                x = torch.nn.functional.interpolate(
                    x[None, None, :],
                    size=self.resolution,
                ).flatten()
                y = torch.nn.functional.interpolate(
                    y[None, None, :],
                    size=self.resolution,
                ).flatten()
            plt.plot(x, y, marker="hd")
            if self.yticks is None:
                min_y = y.min()
                max_y = y.max()
        plt.ylim(*self.ylim)
        plt.xlim(*self.xlim)
        if self.xticks is not None:
            if isinstance(self.xticks, tuple) and len(self.xticks) == 2 and isinstance(self.xticks[0], Iterable):
                plt.xticks(*self.xticks)
            else:
                plt.xticks(self.xticks)
        if self.yticks is not None:
            if isinstance(self.yticks, tuple) and len(self.yticks) == 2 and isinstance(self.yticks[0], Iterable):
                plt.yticks(*self.yticks)
            else:
                plt.yticks(self.yticks)
        else:
            assert (min_y is not None and max_y is not None)
            yticks = torch.linspace(0, 1, 5) * (max_y - min_y) + min_y
            plt.yticks(yticks.tolist())
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        self.last = plt.build()
        self.last_size = (width, height)
        return self.last


@dataclass
class _PlotUpdater:

    _target: _Plot

    def update(
        self,
        *,
        x: FloatScalarType,
        y: Union[FloatScalarType, Dict[str, FloatScalarType]],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xticks: Union[Iterable[float], Tuple[Iterable[float], Iterable[str]], None] = None,
        yticks: Union[Iterable[float], Tuple[Iterable[float], Iterable[str]], None] = None,
        xlim: Tuple[Union[int, float, None], Union[int, float, None]] = (None, None),
        ylim: Tuple[Union[int, float, None], Union[int, float, None]] = (None, None),
    ) -> None:
        self._target.xlabel = xlabel
        self._target.ylabel = ylabel
        self._target.xticks = xticks
        self._target.yticks = yticks
        self._target.xlim = xlim
        self._target.ylim = ylim
        self._target._insert(x, y)


@dataclass
class _PlotMaker:

    _source: Dict[str, _Plot]

    def __getitem__(self, key: str) -> _PlotUpdater:
        if key not in self._source:
            self._source[key] = _Plot(title=key)
            return self._source[key]
        return _PlotUpdater(self._source[key])
