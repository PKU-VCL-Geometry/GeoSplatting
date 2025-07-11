from __future__ import annotations

from typing import Optional, Tuple, TypeVar

import cv2
import numpy as np
import torch

from rfstudio.graphics import RGBAImages, RGBImages

IMG = TypeVar('IMG', RGBAImages, RGBImages)


def highlight(
    figures: IMG,
    *,
    window: Tuple[int, int, int, int],
    border_width: int,
    border_color: Tuple[int, int, int],
    padding: Optional[int] = None,
) -> Tuple[IMG, IMG]:
    main_figures = []
    highlighted = []
    padding = border_width if padding is None else padding
    if isinstance(figures, RGBAImages):
        border_color = border_color + (255,)
    for fig in figures.detach().cpu():
        np_img = (fig.numpy() * 255).astype(np.uint8)
        cv2.rectangle(np_img, window[:2], window[2:], border_color, border_width, cv2.LINE_AA)
        main_figures.append(torch.from_numpy(np_img) / 255)
        np_window = np.ones(
            (window[3] - window[1] + 2 * padding, window[2] - window[0] + 2 * padding, np_img.shape[-1]),
            dtype=np.uint8,
        ) * border_color
        np_window[padding:-padding, padding:-padding, :] = np_img[window[1]:window[3], window[0]:window[2], :]
        highlighted.append(torch.from_numpy(np_window) / 255)
    return figures.__class__(main_figures).to(figures.device), figures.__class__(highlighted).to(figures.device)
