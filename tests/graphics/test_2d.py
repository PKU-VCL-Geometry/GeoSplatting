from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics._2d import Cameras2D, CircleShape2D, Viser2D
from rfstudio.io import dump_float32_image


@dataclass
class TestCircle(Task):

    def run(self) -> None:
        viser = Viser2D()
        circles = CircleShape2D.random(5)
        cameras = Cameras2D.from_lookat(eye=(-0.9, -0.9), target=(0, 0), far=2)
        rays = cameras.generate_rays(downsample_to=8)
        ray_samples = rays.get_samples(torch.linspace(0, 1, 8)[None].repeat(rays.shape[0], 1).to(rays.origins))
        viser.show(circles).show(cameras).show(rays).show(ray_samples).export(Path('temp.circle.png'))
        dump_float32_image(Path('temp.png'), circles.render(cameras).visualize(width=800, height=800).item())

if __name__ == '__main__':
    TaskGroup(
        circle=TestCircle(cuda=0),
        ray=TestCircle(cuda=0),
    ).run()
