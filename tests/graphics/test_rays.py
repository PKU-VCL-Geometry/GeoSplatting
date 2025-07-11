from __future__ import annotations

from dataclasses import dataclass

import torch

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, Points
from rfstudio.model.density_field.components.sampler import NeuSSampler, UniformSampler
from rfstudio.visualization import Visualizer


@dataclass
class RayTester(Task):

    def run(self) -> None:
        camera = Cameras.from_hemisphere(
            center=(0, 1, 0),
            up=(1, 1, 1),
            radius=0.5,
            num_samples=1,
            resolution=(32, 32),
            near=1.,
            far=4.,
        )[0]
        rays = camera.generate_rays()
        pts = rays.get_samples(torch.linspace(0, 1, 8))

        with Visualizer(port=6789).customize() as handle:
            handle['camera'].show(camera)
            handle['raysamples'].show(pts.view_as(Points)).configurate(point_size=0.02)

@dataclass
class RaySamplerTester(Task):

    def run(self) -> None:
        camera = Cameras.from_hemisphere(
            center=(0, 1, 0),
            up=(1, 1, 1),
            radius=0.5,
            num_samples=1,
            resolution=(32, 32),
            near=1.,
            far=4.,
        )[0]
        rays = camera.generate_rays()
        sampler = UniformSampler(8).train()
        pts = sampler(rays)

        neus_sampler = NeuSSampler(4, 16).train()
        neus_pts = neus_sampler(rays, sdf_fn=(lambda x: (x ** 2).sum(-1, keepdim=True) ** 0.5 - 2.5))

        with Visualizer(port=6789).customize() as handle:
            handle['camera'].show(camera)
            handle['uniform'].show(pts.view_as(Points)).configurate(point_size=0.02)
            handle['neus'].show(neus_pts.view_as(Points)).configurate(point_size=0.02)

if __name__ == '__main__':
    TaskGroup(
        rays=RayTester(),
        raysampler=RaySamplerTester(),
    ).run()
