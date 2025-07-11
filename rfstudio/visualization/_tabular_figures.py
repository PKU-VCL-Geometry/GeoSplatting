from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.graphics import RGBAImages, RGBImages
from rfstudio.utils.font import Font
from rfstudio.utils.typing import IntLike


class _CellConfigurator:
    def __init__(self, table: TabularFigures, r_ind: slice, c_ind: slice) -> None:
        self._table = table
        self._r_ind = r_ind
        self._c_ind = c_ind
        mask = table._mask[self._r_ind, self._c_ind]
        self._span = mask.shape
        self._info = None
        self._image = None
        self._color = None

    def _register(self) -> None:
        if self._table is not None:
            assert (self._table._mask[self._r_ind, self._c_ind] == -1).all()
            self._table._mask[self._r_ind, self._c_ind] = len(self._table._cells)
            self._table._cells.append(self)
            self._table = None

    def load(self, image: Union[RGBImages, Tensor], *, info: Optional[str] = None) -> _CellConfigurator:
        assert self._color is None
        self._register()
        self._info = ' ' + info + ' '
        self._image = image
        return self

    def fill(self, color: Tuple[float, float, float]) -> _CellConfigurator:
        assert self._image is None
        self._register()
        self._color = color
        return self

class TabularFigures:

    def __init__(self, num_rows: int, num_cols: int, *, device: Optional[torch.device] = None) -> None:
        self._rn = num_rows
        self._cn = num_cols
        self._cells: List[_CellConfigurator] = []
        self._mask = -torch.ones(num_rows, num_cols, dtype=torch.int64)
        self._device = device

    def __getitem__(self, index: Tuple[Union[IntLike, slice], Union[IntLike, slice]]) -> _CellConfigurator:
        r_index, c_index = index
        if not isinstance(r_index, slice):
            r_index = slice(r_index, r_index + 1)
        if not isinstance(c_index, slice):
            c_index = slice(c_index, c_index + 1)
        return _CellConfigurator(self, r_index, c_index)

    def draw(
        self,
        *,
        text_height_in_cellsize: float = 0.1,
        text_padding_in_cellsize: float = 0.05,
        text_fg_color: Tuple[float, float, float, float] = (0, 0, 0, 1),
        text_bg_color: Optional[Tuple[float, float, float, float]] = None,
        text_anchor: Literal['left-top', 'left-bottom', 'right-top', 'right-bottom'] = 'left-bottom',
        default_bg_color: Tuple[float, float, float] = (1, 1, 1),
        cellsize: Optional[int] = None,
        object_fit: Literal['fill', 'cover', 'disallowed'] = 'disallowed',
    ) -> Tensor:
        if object_fit != 'disallowed':
            raise NotImplementedError
        assert self._cells != []
        if cellsize is None:
            for cell in self._cells:
                if cell._image is None:
                    continue
                shape = cell._image.item().shape
                assert shape[0] % cell._span[0] == 0
                cellsize = shape[0] // cell._span[0]
                assert cellsize * cell._span[1] == shape[1]
                break
        assert cellsize is not None
        bg_color = torch.tensor(default_bg_color, dtype=torch.float32, device=self._device)
        canvas = bg_color.expand(cellsize * self._rn, cellsize * self._cn, 3).contiguous()
        font = Font.from_name('LinLibertine')
        padding = int(cellsize * text_padding_in_cellsize)
        line_height = int(cellsize * text_height_in_cellsize)
        for cell in self._cells:
            r_slice = slice(
                None if cell._r_ind.start is None else cell._r_ind.start * cellsize,
                None if cell._r_ind.stop is None else cell._r_ind.stop * cellsize,
                None if cell._r_ind.step is None else cell._r_ind.step * cellsize,
            )
            c_slice = slice(
                None if cell._c_ind.start is None else cell._c_ind.start * cellsize,
                None if cell._c_ind.stop is None else cell._c_ind.stop * cellsize,
                None if cell._c_ind.step is None else cell._c_ind.step * cellsize,
            )
            if cell._image is not None:
                item = cell._image.detach().to(self._device)
                if isinstance(item, RGBImages):
                    item = item.item()
                elif isinstance(item, RGBAImages):
                    item = item.blend(bg_color).item()
                else:
                    assert isinstance(item, Tensor)
                    assert item.ndim == 3 and item.shape[-1] in [3, 4]
                    if item.shape[-1] == 4:
                        item = item[..., :3] * item[..., 3:] + (1 - item[..., 3:]) * bg_color
                if cell._info is not None:
                    alpha = font.write(cell._info, line_height=line_height).to(item) * text_fg_color[3]
                    if text_anchor == 'left-top':
                        indices = (slice(padding, alpha.shape[0] + padding), slice(padding, alpha.shape[1] + padding))
                    elif text_anchor == 'left-bottom':
                        indices = (slice(-padding-alpha.shape[0], -padding), slice(padding, alpha.shape[1] + padding))
                    elif text_anchor == 'right-top':
                        indices = (slice(padding, alpha.shape[0] + padding), slice(-padding-alpha.shape[1], -padding))
                    elif text_anchor == 'right-bottom':
                        indices = (slice(-padding-alpha.shape[0], -padding), slice(-padding-alpha.shape[1], -padding))
                    else:
                        raise ValueError(text_anchor)
                    if text_bg_color is not None:
                        item[indices] = torch.add(
                            item[indices] * (1 - text_bg_color[3]),
                            torch.tensor(text_bg_color[:3]).to(item) * text_bg_color[3],
                        )
                    item[indices] = item[indices] * (1 - alpha) + torch.tensor(text_fg_color[:3]).to(alpha) * alpha
            else:
                item = torch.tensor(cell._color).to(canvas)
            canvas[r_slice, c_slice, :] = item
        return canvas
