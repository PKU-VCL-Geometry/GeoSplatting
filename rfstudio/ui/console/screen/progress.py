from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Union

from rich.console import RenderableType
from rich.layout import Layout
from rich.progress_bar import ProgressBar
from rich.text import Text


@dataclass
class _Progress:

    value: Union[int, float, None] = None
    total: Optional[int] = None
    start_value: Optional[int] = None
    start_time: Optional[float] = None
    title: Optional[str] = None
    title_width: Optional[int] = None

    def __post_init__(self) -> None:
        self._fin = None

    def _make(self, *, ratio: int = 1) -> RenderableType:
        percent = 0
        if self.value is not None:
            if self.total is None:
                percent = self.value
            else:
                percent = self.value / self.total
            assert 0 <= percent <= 1 and self.start_time is not None

            if percent == 1:
                if self._fin is None:
                    self._fin = time.time()
                spent = self._fin - self.start_time
            else:
                spent = time.time() - self.start_time
            if int(spent // 3600) > 99:
                spent = None
            if self.start_value is None or self.value == self.start_value or spent is None:
                eta = None
            else:
                eta = ((1 if self.total is None else self.total) - self.value) * spent / (self.value - self.start_value)
                if int(eta // 3600) > 99:
                    eta = None
            info = f'{self.value}/{self.total} ' if self.total is not None else ''
        else:
            spent = None
            eta = None
            info = ''
        if spent is None:
            spent = '--:--:--'
        else:
            spent = f'{int(spent//3600):02d}:{int((spent%3600)//60):02d}:{int(spent%60):02d}'
        if eta is None:
            eta = '--:--:--'
        else:
            eta = f'{int(eta//3600):02d}:{int((eta%3600)//60):02d}:{int(eta%60):02d}'
        info = (
            Text(' ', overflow='ellipsis')
            .append(info, style='progress.download')
            .append(f'{percent:.0%} ', style='progress.percentage')
            .append(f'{spent} ', style='progress.elapsed')
            .append(f'{eta}', style='progress.remaining')
        )
        desc = Text.from_markup(('[bold][bright_yellow]' + self.title + ' ') or ' ')
        title_width = desc.cell_len if self.title_width is None else self.title_width
        layout = Layout()
        layout.split_row(
            Layout(desc, size=title_width),
            Layout(
                ProgressBar(completed=percent * 100, width=None),
                ratio=1,
                minimum_size=4,
            ),
            Layout(info, size=info.cell_len)
        )
        return layout


@dataclass
class _ProgressUpdater:

    _target: _Progress

    def update(self, *, curr: Union[int, float], total: Optional[int] = None) -> None:
        self._target.value = curr
        self._target.total = total
        if self._target.start_value is None:
            self._target.start_value = curr
            self._target.start_time = time.time()


@dataclass
class _ProgressMaker:

    _source: Dict[str, _Progress]

    def __getitem__(self, key: str) -> _ProgressUpdater:
        if key not in self._source:
            self._source[key] = _Progress(title=key)
        return _ProgressUpdater(self._source[key])
