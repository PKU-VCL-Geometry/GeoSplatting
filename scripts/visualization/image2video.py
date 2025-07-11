from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from rfstudio.engine.task import Task
from rfstudio.io import load_float32_image, open_video_renderer
from rfstudio.ui import console


@dataclass
class Image2Video(Task):

    input: Path = ...

    output: Path = ...

    fps: float = 12

    max_duration: Optional[float] = None

    downsample: Optional[float] = None

    target_mb: Optional[float] = None

    extension: Literal['png', 'jpg', 'jpeg'] = 'png'

    def run(self) -> None:

        image_list = list(self.input.glob(f"*.{self.extension}"))
        image_list.sort(key=lambda p: p.name)

        if self.max_duration is not None:
            image_list = image_list[:int(self.max_duration * self.fps)]

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open_video_renderer(
            self.output,
            fps=self.fps,
            downsample=self.downsample,
            target_mb=self.target_mb,
        ) as renderer:
            with console.progress('Exporting...') as ptrack:
                for image_path in ptrack(image_list):
                    renderer.write(load_float32_image(image_path))

if __name__ == '__main__':
    Image2Video().run()
