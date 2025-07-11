from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rich import box
from rich.console import RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from .layout import _Layout


@dataclass
class _Text(_Layout):

    text: str
    title: Optional[str] = None
    justify: Literal['left', 'center', 'right'] = 'left'
    overflow: Literal['fold', 'crop', 'ellipsis'] = 'fold'
    line_height: Optional[int] = 1

    def _make(self, *, ratio: int = 1) -> RenderableType:
        text = Text.from_markup(
            self.text,
            justify=self.justify,
            overflow=self.overflow,
        )
        if self.line_height is not None:
            return Layout(
                Panel(text, title=self.title, title_align='left'),
                size=self.line_height + 2,
            )
        layout = Layout()
        layout.split_column(
            Layout(Panel(Text(), box=box.MINIMAL)),
            Layout(text, size=1),
            Layout(Panel(Text(), box=box.MINIMAL)),
        )
        return Layout(Panel(layout, title=self.title), ratio=ratio)
