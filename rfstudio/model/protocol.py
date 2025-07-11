from __future__ import annotations

from typing import Protocol, runtime_checkable

from rfstudio.graphics import Cameras, DepthImages, RGBImages


@runtime_checkable
class RGBImageRenderable(Protocol):

    def render_rgb(self, inputs: Cameras) -> RGBImages:
        ...

@runtime_checkable
class DepthImageRenderable(Protocol):

    def render_depth(self, inputs: Cameras) -> DepthImages:
        ...
