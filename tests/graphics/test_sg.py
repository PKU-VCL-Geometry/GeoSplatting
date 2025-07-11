from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import TextureCubeMap, TextureSG
from rfstudio.io import dump_float32_image


@dataclass
class Tester(Task):

    envmap: Path = Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr'
    output: Path = Path('temp.png')

    def run(self) -> None:
        cubemap = TextureCubeMap.from_image_file(self.envmap, device=self.device)
        invertible = cubemap.as_sg(128)
        random = TextureSG.from_random(128, device=self.device)
        img = torch.cat((
            cubemap.visualize().item(),
            invertible.visualize().item(),
            random.visualize().item(),
        ), dim=0)
        dump_float32_image(self.output, img.clamp(0, 1))

if __name__ == '__main__':
    Tester(cuda=0).run()
