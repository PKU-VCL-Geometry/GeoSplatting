from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import TextureCubeMap, TextureLatLng
from rfstudio.io import dump_float32_image


@dataclass
class Tester(Task):

    envmap: Path = Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr'
    output: Path = Path('temp.png')
    vis: bool = False

    def run(self) -> None:
        latlng = TextureLatLng.from_image_file(self.envmap, device=self.device)
        cubemap = TextureCubeMap.from_image_file(self.envmap, device=self.device)
        if self.vis:
            dump_float32_image(self.output, cubemap.visualize().clamp(0, 1).item())
            return
        invertible = (
            latlng
                .as_cubemap(resolution=cubemap.resolution)
                .as_latlng(width=latlng.width, height=latlng.height)
        )
        row1 = torch.cat((
            latlng.data,
            latlng.visualize().item(),
            cubemap.visualize(width=latlng.width, height=latlng.height).item(),
            invertible.visualize().item(),
        ), dim=1)
        latlng.z_up_to_y_up_()
        cubemap.z_up_to_y_up_()
        invertible = (
            latlng
                .as_cubemap(resolution=cubemap.resolution)
                .as_latlng(width=latlng.width, height=latlng.height)
        )
        row2 = torch.cat((
            latlng.data / (latlng.data + 1),
            latlng.visualize().item(),
            cubemap.visualize(width=latlng.width, height=latlng.height).item(),
            invertible.visualize().item(),
        ), dim=1)
        img = torch.cat((row1, row2), dim=0)
        dump_float32_image(self.output, img.clamp(0, 1))

if __name__ == '__main__':
    Tester(cuda=0).run()
