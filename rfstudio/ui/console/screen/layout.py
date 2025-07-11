from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from rich.console import RenderableType
from rich.layout import Layout
from rich.panel import Panel


@dataclass
class _Layout:

    def _make(self, *, ratio: int = 1) -> RenderableType:
        raise NotImplementedError


@dataclass
class _Splits(_Layout):

    children: List[_Layout]
    ratios: Optional[List[int]]
    split: Literal['col', 'row']

    def _make(self, *, ratio: int = 1) -> RenderableType:
        layout = Layout(ratio=ratio)
        ratios = ([1] * len(self.children)) if self.ratios is None else self.ratios
        children = [child._make(ratio=ratio) for child, ratio in zip(self.children, ratios)]
        if self.split == 'row':
            layout.split_column(*children)
        else:
            layout.split_row(*children)
        return layout


@dataclass
class _SplitsMaker:

    split: Literal['col', 'row']
    ratios: Optional[List[int]] = None

    def __getitem__(self, ratios: Tuple[int]) -> _SplitsMaker:
        assert self.ratios is None, "Duplicated ratio setting."
        self.ratios = list(ratios)
        return self

    def __call__(self, *layouts: _Layout) -> _Splits:
        return _Splits(layouts, ratios=self.ratios, split=self.split)


@dataclass
class _Lines(_Layout):

    children: List[_Layout]
    title: Optional[str] = None

    def _make(self, *, ratio: int = 1) -> RenderableType:
        inner_layout = Layout(size=len(self.children))
        inner_layout.split_column(*[child._make() for child in self.children])
        return Layout(
            Panel(inner_layout, title=self.title, title_align='left'),
            size=len(self.children) + 2,
        )
