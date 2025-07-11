from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import MultiViewDataset, SfMDataset
from rfstudio.engine.task import Task
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras
from rfstudio.io import open_video_renderer
from rfstudio.model.protocol import RGBImageRenderable
from rfstudio.ui import console


@dataclass
class VideoRenderTask(Task):

    load: Path = ...

    output: Path = ...

    duration: float = ...

    fps: float = 60

    downsample: Optional[float] = None

    @torch.no_grad()
    def run(self) -> None:

        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, RGBImageRenderable)
        assert isinstance(train_task.dataset, (MultiViewDataset, SfMDataset))

        with console.status('Loading dataset...'):
            train_task.dataset.load(split='train')
        cameras: Cameras = train_task.dataset.get_inputs(split='train')
        cameras = cameras.sample_sequentially(
            int(self.fps * self.duration),
            uniform_by='distance'
        )

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open_video_renderer(
            self.output,
            fps=self.fps,
            downsample=self.downsample
        ) as renderer:
            with console.progress('Exporting...') as ptrack:
                for i in ptrack(range(cameras.shape[0])):
                    inputs = cameras[i:i+1]
                    img = model.render_rgb(inputs).item()
                    renderer.write(img)

if __name__ == '__main__':
    VideoRenderTask(cuda=0).run()
