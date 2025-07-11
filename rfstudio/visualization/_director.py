from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor

from rfstudio.graphics import RGBAImages
from rfstudio.graphics._cameras import _CachedPixelCoords
from rfstudio.io import dump_float32_image, load_float32_image, load_float32_masked_image, open_video_renderer
from rfstudio.ui import console
from rfstudio.utils.font import Font


@runtime_checkable
class Animatable(Protocol):
    def render_frame(
        self,
        idx: int,
        *,
        suggested_size: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
        preview: bool = False,
    ) -> Tensor:
        ...

class Animation:

    @dataclass
    class Radio:
        options: List[Animatable]
        switch_indices: List[int]
        min_opacity: float = 0.2
        duration: int = 20

        def __post_init__(self) -> None:
            assert len(self.options) == len(self.switch_indices) + 1
            for i, j in zip(self.switch_indices[:1], self.switch_indices[1:]):
                assert i < j

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            assert suggested_size is not None
            W, H = suggested_size
            unit = W // len(self.options)
            orin_x = (W - unit * len(self.options)) // 2
            if preview:
                for i, option in enumerate(self.options[:-1]):
                    if idx < self.switch_indices[i]:
                        return option.render_frame(idx, suggested_size=(unit, H), device=device, preview=True)
                return self.options[-1].render_frame(idx, suggested_size=(unit, H), device=device, preview=True)
            ratios = [1.]
            for switch_idx in self.switch_indices:
                ratios.append(max(0, min(1, (idx - switch_idx + self.duration // 2) / self.duration)))
            ratios.append(0.)
            canvas = torch.zeros(H, W, 4, device=device)
            for (i, option), lr, rr in zip(enumerate(self.options), ratios[:-1], ratios[1:], strict=True):
                x = min(lr, 1 - rr) * (1 - self.min_opacity) + self.min_opacity
                content = option.render_frame(idx, suggested_size=(unit, H), device=device)
                content[..., 3:] *= x
                canvas[:, orin_x+unit*i:orin_x+unit+unit*i, :] = content
            return canvas


    @dataclass
    class GridContainer:
        padding: int
        text_line_height: int
        rows: Dict[str, float]
        cols: Dict[str, float]
        align: Literal['left', 'center', 'right'] = 'center'

        def __post_init__(self) -> None:
            self._children: Dict[str, Dict[str, Optional[Animatable]]] = {}
            self._cached_slices: Dict[str, Dict[str, Tuple[slice, slice]]] = {}
            for row_name in self.rows.keys():
                self._children[row_name] = { col_name: None for col_name in self.cols.keys() }
                self._cached_slices[row_name] = {}
            self._cache: Optional[Tensor] = None
            self._cached_size: Optional[Tuple[int, int]] = None

        def __getitem__(self, row_name: str) -> Dict[str, Optional[Animatable]]:
            return self._children[row_name]

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            assert suggested_size is not None
            if self._cache is None or self._cached_size != suggested_size:
                W, H = suggested_size
                n_rows = len(self.rows.keys())
                n_cols = len(self.cols.keys())
                rest_width = W - self.text_line_height - n_cols * self.padding
                rest_height = H - self.text_line_height - (n_rows + 1) * self.padding
                unit_in_w = int(rest_width / sum(self.cols.values()))
                unit_in_h = int(rest_height / sum(self.rows.values()))
                unit = min(unit_in_h, unit_in_w)
                canvas = torch.zeros((H, W, 4), device=device)
                col_spans = torch.tensor([int(unit * col_span) for col_span in self.cols.values()])
                row_spans = torch.tensor([int(unit * row_span) for row_span in self.rows.values()])
                rest_width = rest_width - col_spans.sum().item()
                rest_height = rest_height - row_spans.sum().item()
                orin_x = (
                    rest_width // 2
                    if self.align == 'center'
                    else (0 if self.align == 'left' else rest_width)
                )
                orin_y = rest_height // 2
                assert orin_x >= 0 and orin_y >= 0
                col_ends = col_spans.cumsum(0)
                col_starts = col_ends.clone().roll(1, dims=0)
                col_starts[0] = 0
                row_ends = row_spans.cumsum(0)
                row_starts = row_ends.clone().roll(1, dims=0)
                row_starts[0] = 0
                for row_idx, row_name in enumerate(self.rows.keys()):
                    row_text = Animation.StaticText(text=row_name, bold=True).render_frame(
                        idx,
                        suggested_size=(row_spans[row_idx].item(), self.text_line_height),
                        device=device,
                        preview=preview,
                    ).transpose(0, 1).flip(0).contiguous()
                    npad = self.padding * (row_idx + 1)
                    canvas[
                        orin_y + npad + row_starts[row_idx].item() : orin_y + npad + row_ends[row_idx].item(),
                        orin_x: orin_x + self.text_line_height,
                    ] = row_text
                for col_idx, col_name in enumerate(self.cols.keys()):
                    col_text = Animation.StaticText(text=col_name, bold=True).render_frame(
                        idx,
                        suggested_size=(col_spans[col_idx].item(), self.text_line_height),
                        device=device,
                        preview=preview,
                    )
                    npad = self.text_line_height + self.padding * (col_idx + 1)
                    canvas[
                        canvas.shape[0] - orin_y - self.text_line_height : canvas.shape[0] - orin_y,
                        orin_x + npad + col_starts[col_idx].item() : orin_x + npad + col_ends[col_idx].item(),
                    ] = col_text

                self._cache = canvas
                self._cached_slices = {}
                self._cached_size = suggested_size
                for row_idx, row_name in enumerate(self.rows.keys()):
                    rs = orin_y + self.padding * (row_idx + 1) + row_starts[row_idx]
                    re = orin_y + self.padding * (row_idx + 1) + row_ends[row_idx]
                    self._cached_slices[row_name] = {}
                    for col_idx, col_name in enumerate(self.cols.keys()):
                        cs = orin_x + self.text_line_height + self.padding * (col_idx + 1) + col_starts[col_idx]
                        ce = orin_x + self.text_line_height + self.padding * (col_idx + 1) + col_ends[col_idx]
                        self._cached_slices[row_name][col_name] = (slice(rs, re), slice(cs, ce))

            self._cache = self._cache.to(device)
            cached_canvas = self._cache
            for row_name, row_item in self._children.items():
                for col_name, child in row_item.items():
                    if child is None:
                        continue
                    rslice, cslice = self._cached_slices[row_name][col_name]
                    H, W = cached_canvas[rslice, cslice].shape[:2]
                    cached_canvas[rslice, cslice] = child.render_frame(
                        idx,
                        suggested_size=(W, H),
                        device=device,
                        preview=preview,
                    )
            return cached_canvas


    @dataclass
    class StaticPureColor:
        color: Tuple[int, int, int] = (255, 255, 255)
        alpha: float = 1.0

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            assert suggested_size is not None
            W, H = suggested_size
            r, g, b = self.color
            return torch.tensor([r/255, g/255, b/255, self.alpha]).float().to(device).view(1, 1, 4).repeat(H, W, 1)

    @dataclass
    class StaticImage:
        content: Tensor

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            if preview:
                if suggested_size is None:
                    H, W = self.content.shape[:2]
                else:
                    W, H = suggested_size
                return (torch.tensor([36, 123, 95, 255], device=device) / 255).expand(H, W, 4).contiguous()
            content = self.content.to(device)
            if content.shape[-1] == 3:
                content = torch.cat((content, torch.ones_like(content[..., :1])), dim=-1)
            if suggested_size is not None:
                content = RGBAImages([content]).resize_to(*suggested_size).item()
            return content

    @dataclass
    class Highlight:
        content: Animatable
        start: int
        stop: int
        source: Tuple[slice[float, float], slice[float, float]]
        target: Tuple[slice[float, float], slice[float, float]]
        thickness: int
        color: Tuple[int, int, int] = (90, 116, 143)
        source_in: int = 10
        target_in: int = 10
        fade_out: int = 10

        def __post_init__(self) -> None:
            assert self.stop - self.start > self.source_in + self.target_in + self.fade_out

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            if not preview and (idx <= self.start or idx >= self.stop):
                return self.content.render_frame(idx, suggested_size=suggested_size, device=device)
            if suggested_size is None:
                content = self.content.render_frame(idx, suggested_size=suggested_size, device=device)
                H, W = content.shape[:2]
            else:
                content = None
                W, H = suggested_size
            border = torch.tensor((*self.color, 255), device=device) / 255
            sy1 = int(self.source[0].start * H)
            sy2 = int(self.source[0].stop * H)
            sx1 = int(self.source[1].start * W)
            sx2 = int(self.source[1].stop * W)
            ty1 = int(self.target[0].start * H)
            ty2 = int(self.target[0].stop * H)
            tx1 = int(self.target[1].start * W)
            tx2 = int(self.target[1].stop * W)
            if preview:
                if content is None:
                    content = self.content.render_frame(idx, suggested_size=suggested_size, device=device, preview=True)
                content[sy1:sy2, sx1:sx1+self.thickness, :] = border
                content[sy1:sy2, sx2-self.thickness:sx2, :] = border
                content[sy1:sy1+self.thickness, sx1:sx2, :] = border
                content[sy2-self.thickness:sy2, sx1:sx2, :] = border
                content[ty1:ty2, tx1:tx1+self.thickness, :] = border
                content[ty1:ty2, tx2-self.thickness:tx2, :] = border
                content[ty1:ty1+self.thickness, tx1:tx2, :] = border
                content[ty2-self.thickness:ty2, tx1:tx2, :] = border
                return content

            source = content
            ssy1, ssy2, ssx1, ssx2 = sy1, sy2, sx1, sx2
            tthickness = self.thickness
            if content is None:
                ratio = max((ty2 - ty1) / (sy2 - sy1), (tx2 - tx1) / (ty2 - ty1))
                content = self.content.render_frame(idx, suggested_size=suggested_size, device=device)
                if ratio > 1:
                    suggested_size = (round(W * ratio), round(H * ratio))
                    ssy1 = int(self.source[0].start * suggested_size[1])
                    ssy2 = int(self.source[0].stop * suggested_size[1])
                    ssx1 = int(self.source[1].start * suggested_size[0])
                    ssx2 = int(self.source[1].stop * suggested_size[0])
                    tthickness = round(self.thickness * ratio)
                    source = self.content.render_frame(idx, suggested_size=suggested_size, device=device)
            if idx <= self.start + self.source_in:
                x = (idx - self.start) / self.source_in
                canvas = content[sy1:sy2, sx1:sx2].clone()
                canvas[:, :self.thickness, :] = border
                canvas[:, -self.thickness:, :] = border
                canvas[-self.thickness:, :, :] = border
                canvas[:self.thickness, :, :] = border
                content[sy1:sy2, sx1:sx2, :] = _switch(content[sy1:sy2, sx1:sx2], canvas, ratio=x, mode='flush')
                return content
            if idx < self.start + self.source_in + self.target_in:
                x = (idx - self.start - self.source_in) / self.target_in
                x = -(math.cos(math.pi * x) - 1) / 2 # ease inout sine
                content[sy1:sy2, sx1:sx1+self.thickness, :] = border
                content[sy1:sy2, sx2-self.thickness:sx2, :] = border
                content[sy1:sy1+self.thickness, sx1:sx2, :] = border
                content[sy2-self.thickness:sy2, sx1:sx2, :] = border
                cx1 = round((1 - x) * sx1 + tx1 * x)
                cx2 = round((1 - x) * sx2 + tx2 * x)
                cy1 = round((1 - x) * sy1 + ty1 * x)
                cy2 = round((1 - x) * sy2 + ty2 * x)
                cthickness = round(tthickness * x + self.thickness * (1 - x))
                content[cy1:cy2, cx1:cx2, :] = RGBAImages(
                    [source[ssy1:ssy2, ssx1:ssx2]]
                ).resize_to(cx2 - cx1, cy2 - cy1).item()
                content[cy1:cy2, cx1:cx1+cthickness, :] = border
                content[cy1:cy2, cx2-cthickness:cx2, :] = border
                content[cy1:cy1+cthickness, cx1:cx2, :] = border
                content[cy2-cthickness:cy2, cx1:cx2, :] = border
                return content
            if idx > self.stop - self.fade_out:
                origin = content.clone()
            content[sy1:sy2, sx1:sx1+self.thickness, :] = border
            content[sy1:sy2, sx2-self.thickness:sx2, :] = border
            content[sy1:sy1+self.thickness, sx1:sx2, :] = border
            content[sy2-self.thickness:sy2, sx1:sx2, :] = border
            content[ty1:ty2, tx1:tx2, :] = RGBAImages(
                [source[ssy1:ssy2, ssx1:ssx2]]
            ).resize_to(tx2 - tx1, ty2 - ty1).item()
            content[ty1:ty2, tx1:tx1+tthickness, :] = border
            content[ty1:ty2, tx2-tthickness:tx2, :] = border
            content[ty1:ty1+tthickness, tx1:tx2, :] = border
            content[ty2-tthickness:ty2, tx1:tx2, :] = border
            if idx > self.stop - self.fade_out:
                x = (idx - self.stop + self.fade_out) / self.fade_out
                content = origin * x + content * (1 - x)
            return content

    @dataclass
    class ImageFrames:
        folder: Path
        start: int = 0
        cut_in: Optional[int] = None
        cut_out: Optional[int] = None
        clip: Optional[Tuple[slice, slice]] = None
        padding_lbrt: Optional[Tuple[int, int, int, int]] = None
        pattern: str = "*.png"
        use_rgba: bool = True

        def __getitem__(self, index: slice) -> Animation.ImageFrames:
            assert isinstance(index, slice)
            curr_slice = slice(self.cut_in, self.cut_out)
            images = list(self.folder.glob(self.pattern))
            cut_in, cut_out, _ = index.indices(len(images[curr_slice]))
            offest, _, _ = curr_slice.indices(len(images))
            return Animation.ImageFrames(
                start=self.start,
                folder=self.folder,
                cut_in=cut_in+offest,
                cut_out=cut_out+offest,
                clip=self.clip,
            )

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            image_paths = list(self.folder.glob(self.pattern))
            clip = ... if self.clip is None else self.clip
            if preview:
                if suggested_size is None:
                    H, W = load_float32_masked_image(image_paths[0])[clip].shape[:2]
                else:
                    W, H = suggested_size
                return (torch.tensor([36, 123, 95, 255], device=device) / 255).expand(H, W, 4).contiguous()
            image_paths.sort(key=lambda p: p.stem)
            image_paths = image_paths[slice(self.cut_in, self.cut_out)]
            filename = image_paths[max(0, min(len(image_paths) - 1, idx - self.start))]
            if self.use_rgba:
                content = load_float32_masked_image(filename)
            else:
                content = load_float32_image(filename)
                content = torch.cat((content, torch.ones_like(content[..., :1])), dim=-1)
            if self.padding_lbrt is not None:
                pl, pb, pr, pt = self.padding_lbrt
                content = torch.nn.functional.pad(
                    content,
                    (0, 0, pl, pr, pt, pb),
                    mode='constant',
                    value=0.0 if self.use_rgba else 1.0,
                )
            content = content[clip].contiguous().to(device)
            if suggested_size is not None:
                content = RGBAImages([content]).resize_to(*suggested_size).item()
            return content

    @dataclass
    class StaticText:
        text: str
        bold: bool
        align: Literal['left', 'center', 'right'] = 'center'

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            assert suggested_size is not None
            if preview:
                return (torch.tensor([20, 115, 230, 255], device=device) / 255).expand(
                    suggested_size[1], suggested_size[0], 4
                ).contiguous()
            font = Font.from_name('TimesNewRoman_B') if self.bold else Font.from_name('TimesNewRoman')
            content = font.write(self.text, line_height=suggested_size[1])
            if content.shape[1] > suggested_size[0]:
                line_height = int(suggested_size[1] * suggested_size[0] / content.shape[1])
                content = font.write(self.text, line_height=line_height)
            padded = content.new_zeros(suggested_size[1], suggested_size[0], 4)
            dW = padded.shape[1] - content.shape[1]
            dH = padded.shape[0] - content.shape[0]
            assert dW >= 0 and dH >= 0
            if self.align == 'center':
                padded[dH//2:dH//2+content.shape[0], dW//2:dW//2+content.shape[1], 3:] = content
            elif self.align == 'left':
                padded[dH//2:dH//2+content.shape[0], :content.shape[1], 3:] = content
            elif self.align == 'right':
                padded[dH//2:dH//2+content.shape[0], dW:dW+content.shape[1], 3:] = content
            else:
                raise ValueError(self.align)
            return padded.to(device)

    @dataclass
    class FadeIn:
        content: Animatable
        start: int
        stop: int
        no_preview: bool = False

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            if preview:
                content = self.content.render_frame(idx, suggested_size=suggested_size, device=device, preview=True)
                return torch.zeros_like(content) if self.no_preview else content
            if idx <= self.start:
                if suggested_size is not None:
                    return torch.zeros(suggested_size[1], suggested_size[0], 4, device=device)
                return torch.zeros(1, 1, 4, device=device)
            content = self.content.render_frame(idx, suggested_size=suggested_size, device=device)
            if idx <= self.stop:
                x = (idx - self.start) / (self.stop - self.start)
                content[..., 3:] *= x
            return content

    @dataclass
    class FadeOut:
        content: Animatable
        start: int
        stop: int
        no_preview: bool = False

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            if preview:
                content = self.content.render_frame(idx, suggested_size=suggested_size, device=device, preview=True)
                return torch.zeros_like(content) if self.no_preview else content
            if idx >= self.stop:
                if suggested_size is not None:
                    return torch.zeros(suggested_size[1], suggested_size[0], 4, device=device)
                return torch.zeros(1, 1, 4, device=device)
            content = self.content.render_frame(idx, suggested_size=suggested_size, device=device)
            if idx >= self.start:
                x = (idx - self.start) / (self.stop - self.start)
                content[..., 3:] *= (1 - x)
            return content

    @dataclass
    class Switch:
        before: Animatable
        after: Animatable
        start: int
        stop: int
        mode: Literal['fade', 'flush', 'push'] = 'flush'

        def render_frame(
            self,
            idx: int,
            *,
            suggested_size: Optional[Tuple[int, int]] = None,
            device: Optional[torch.device] = None,
            preview: bool = False,
        ) -> Tensor:
            if idx <= self.start:
                return self.before.render_frame(idx, suggested_size=suggested_size, device=device, preview=preview)
            if idx >= self.stop:
                return self.after.render_frame(idx, suggested_size=suggested_size, device=device, preview=preview)
            if preview:
                return self.before.render_frame(idx, suggested_size=suggested_size, device=device, preview=True)
            x = (idx - self.start) / (self.stop - self.start)
            before = self.before.render_frame(idx, suggested_size=suggested_size, device=device)
            after = self.after.render_frame(idx, suggested_size=suggested_size, device=device)
            return _switch(before, after, ratio=x, mode=self.mode)


class _StageDirector:

    def __init__(self, duration: int, resolution: Tuple[int, int]) -> None:
        self._duration = duration
        self._posed_animations: List[Tuple[int, int, Optional[Tuple[int, int]], Animatable]] = []
        self._resolution = resolution
        self._stage_effects: List[Animatable] = []

    def fade_in(self, duration: int) -> _StageDirector:
        self._stage_effects.append(Animation.FadeOut(
            Animation.StaticPureColor(),
            start=0,
            stop=duration,
            no_preview=True,
        ))
        return self

    def fade_out(self, duration: int) -> _StageDirector:
        self._stage_effects.append(Animation.FadeIn(
            Animation.StaticPureColor(),
            start=self._duration - duration,
            stop=self._duration,
            no_preview=True,
        ))
        return self

    def __setitem__(self, index: Tuple[Union[int, float, slice], Union[int, float, slice]], value: Animatable) -> None:
        assert isinstance(value, Animatable)
        y, x = index
        W, H = self._resolution
        yspan, xspan = None, None
        assert isinstance(x, (int, float, slice))
        assert isinstance(y, (int, float, slice))
        if isinstance(y, float):
            y = round(H * y)
        elif isinstance(y, slice):
            assert (
                (y.step is None) and
                (y.start is None or isinstance(y.start, (int, float))) and
                (y.stop is None or isinstance(y.stop, (int, float)))
            )
            y = slice(
                round(H * y.start) if isinstance(y.start, float) else y.start,
                round(H * y.stop) if isinstance(y.stop, float) else y.stop,
            )
            ystart, ystop, _ = y.indices(H)
            yspan = ystop - ystart
            y = ystart
            assert yspan > 0
        if isinstance(x, float):
            x = round(W * x)
        elif isinstance(x, slice):
            assert (
                (x.step is None) and
                (x.start is None or isinstance(x.start, (int, float))) and
                (x.stop is None or isinstance(x.stop, (int, float)))
            )
            x = slice(
                round(W * x.start) if isinstance(x.start, float) else x.start,
                round(W * x.stop) if isinstance(x.stop, float) else x.stop,
            )
            xstart, xstop, _ = x.indices(W)
            xspan = xstop - xstart
            x = xstart
            assert xspan > 0
        assert (yspan is None) == (xspan is None)
        suggested_size = None if xspan is None else (xspan, yspan)
        self._posed_animations.append((x, y, suggested_size, value))

def _blend(fg: Tensor, bg: Tensor) -> Tensor:
    return fg[..., :3] * fg[..., 3:] + (1 - fg[..., 3:]) * bg

def _switch(before: Tensor, after: Tensor, *, ratio: float, mode: Literal['fade', 'flush']) -> Tensor:
    assert 0 <= ratio <= 1
    assert before.shape == after.shape
    H, W = before.shape[:2]
    if mode == 'push':
        x = math.sin(math.pi * ratio / 2) # ease out sine
        W_after = round(x * W)
        result = torch.cat((before[:, W_after:, :], after[:, :W_after, :]), dim=1)
        assert result.shape[:2] == (H, W)
        return result
    x = ratio # linear
    if mode == 'fade':
        pass
    elif mode == 'flush':
        yx_grid = _CachedPixelCoords.get(height=H, width=W, device=before.device) # [H, W, 2]
        x = (yx_grid[..., 0:1] * W + yx_grid[..., 1:2] * H >= 2 * H * W * (1 - x)).float()
    else:
        raise ValueError(mode)
    return x * after + (1 - x) * before

@dataclass
class Director:

    workspace: Path = ...
    preview: bool = False
    resolution: Literal['4K', '2K', '1080p', '720p', '360p'] = '1080p'
    aspect: Literal['16:9', '4:3'] = '16:9'
    stage_only: Optional[str] = None
    frame_only: Optional[int] = None
    overwritten: Optional[int] = None

    def __setup__(self) -> None:
        self.workspace.mkdir(exist_ok=True, parents=True)
        self._stages: List[str] = []
        self._device = None
        self._num_total_frames = 0

    def to(self, device: Optional[torch.device]) -> None:
        self._device = device

    @property
    def _resolution(self) -> Tuple[int, int]:
        assert self.aspect in ('16:9', '4:3')
        if self.resolution in ('4K', '2K'):
            W = 3840 if self.resolution == '4K' else 1920
            H = W // 16 * 9 if self.aspect == '16:9' else W // 4 * 3
            return W, H
        if self.resolution in ('1080p', '720p', '360p'):
            H = int(self.resolution[:-1])
            W = H // 9 * 16 if self.aspect == '16:9' else H // 3 * 4
            return W, H
        raise ValueError(self.resolution)

    def _realize(self, name: str, sd: _StageDirector) -> None:
        (self.workspace / name).mkdir(parents=True, exist_ok=True)
        W, H = self._resolution
        canvas = torch.empty(H, W, 3, device=self._device)
        with console.progress(f'Rendering stage {name}') as ptrack:
            for i in ptrack(
                range(1 if self.preview else sd._duration)
                if self.frame_only is None
                else [self.frame_only]
            ):
                output = self.workspace / name / ('preview.png' if self.preview else f'{i:04d}.png')
                if (self.overwritten is None and not self.preview and not self.frame_only and output.exists()) or (
                    self.overwritten is not None and self._num_total_frames + i < self.overwritten
                ):
                    continue
                canvas.fill_(1)
                for x, y, suggested_size, animatable in sd._posed_animations:
                    content = animatable.render_frame(
                        i,
                        suggested_size=suggested_size,
                        device=self._device,
                        preview=self.preview,
                    )
                    canvas[y:y+content.shape[0], x:x+content.shape[1]] = _blend(
                        fg=content,
                        bg=canvas[y:y+content.shape[0], x:x+content.shape[1]],
                    )
                if not self.preview:
                    for animatable in sd._stage_effects:
                        content = animatable.render_frame(
                            i,
                            suggested_size=(W, H),
                            device=self._device,
                            preview=self.preview,
                        )
                        canvas = _blend(fg=content, bg=canvas)
                dump_float32_image(output, canvas)

    @contextmanager
    def stage(self, name: str, *, duration: int) -> Iterator[_StageDirector]:
        assert name not in self._stages
        sd = _StageDirector(duration=duration, resolution=self._resolution)
        try:
            yield sd
            if self.stage_only is None or name == self.stage_only:
                self._realize(name, sd)
        finally:
            if self.stage_only is None or name == self.stage_only:
                self._num_total_frames += duration
                self._stages.append(name)

    def export(self, *, file: Path = None, fps: float = 20.0, target_mb: float = 128.0) -> None:
        if self.preview:
            return
        if file is None:
            file = self.workspace / 'output.mp4'
        else:
            file.parent.mkdir(exist_ok=True, parents=True)
        total_frames = []
        for stage in self._stages:
            image_paths = list((self.workspace / stage).glob("*.png"))
            image_paths.sort(key=lambda p: p.stem)
            total_frames += image_paths
        with open_video_renderer(file, fps=fps, target_mb=target_mb) as renderer:
            with console.progress('Fusing') as ptrack:
                for path in ptrack(total_frames):
                    renderer.write(load_float32_image(path))
