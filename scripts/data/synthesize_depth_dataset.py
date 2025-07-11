from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.data.dataparser import DepthBlenderDataparser
from rfstudio.engine.task import Task
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, DepthImages
from rfstudio.model.protocol import DepthImageRenderable
from rfstudio.ui import console


@dataclass
class Synthesis(Task):

    load: Path = ...

    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, DepthImageRenderable)
        cameras = Cameras.from_sphere(
            center=(0, 0, 0),
            up=(0, 0, 1),
            radius=2.67,
            num_samples=400,
            resolution=(800, 800),
            hfov_degree=40,
            device=self.device,
        )
        self.output.mkdir(parents=True, exist_ok=True)

        depths = []
        with console.progress('Rendering Depth Maps') as handle:
            for camera in handle(cameras.view(-1, 1)):
                depths.append(model.render_depth(camera).item())
        depths = torch.stack(depths)
        DepthBlenderDataparser.dump(cameras[:100], DepthImages(depths[:100]), None, path=self.output, split='train')
        DepthBlenderDataparser.dump(cameras[100:200], DepthImages(depths[100:200]), None, path=self.output, split='val')
        DepthBlenderDataparser.dump(cameras[200:], DepthImages(depths[200:]), None, path=self.output, split='test')

if __name__ == '__main__':
    Synthesis(cuda=0).run()
