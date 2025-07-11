from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console

from .basic import _Text
from .layout import _Layout, _Lines, _Splits, _SplitsMaker
from .plot import _PlotMaker
from .progress import _ProgressMaker
from .table import _TableMaker


@dataclass
class _ScreenHandle:

    _desc: str
    _console: Console
    _update_interval: float = 2.0

    def __post_init__(self) -> None:
        self._live = None
        self._layout = None
        self._tables = {}
        self._plots = {}
        self._progresses = {}
        self._live_time = None

    def _start(self) -> None:
        self._layout = _Text('[bold][bright_yellow]' + self._desc, justify='center', line_height=None)
        self._live = self._console.screen()
        self._live.__enter__()
        self._live_time = time.time()
        self._live.update(self._layout._make())

    def _stop(self) -> None:
        self._live.__exit__(None, None, None)
        self._live = None
        self._live_time = None
        self._layout = None

    def hold(self, desc: str) -> None:
        renderable = _Text(text='[bold][bright_yellow]' + desc, line_height=None, justify='center')._make()
        while True:
            time.sleep(0.5)
            self._live.update(renderable)

    @property
    def cols(self) -> _SplitsMaker:
        return _SplitsMaker(split='col')

    @property
    def rows(self) -> _SplitsMaker:
        return _SplitsMaker(split='row')

    @property
    def table(self) -> _TableMaker:
        return _TableMaker(self._tables)

    @property
    def progress(self) -> _ProgressMaker:
        return _ProgressMaker(self._progresses)

    @property
    def plot(self) -> _PlotMaker:
        return _PlotMaker(self._plots)

    def sync(self, *, force: bool = False) -> None:
        curr_time = time.time()
        if force or (curr_time > self._live_time + self._update_interval):
            self._live.update(self._get_layout()._make())
            self._live_time = curr_time

    def _get_layout(self) -> _Layout:
        if not self._progresses:
            return self._layout
        max_width = max([len([] if p.title is None else p.title) for p in self._progresses.values()]) + 1
        for p in self._progresses.values():
            p.title_width = max_width
        progresses = _Lines(list(self._progresses.values()), title='progress')
        return _Splits(children=[self._layout, progresses], ratios=None, split='row')

    def set_layout(self, layout: _Layout) -> None:
        self._layout = layout
