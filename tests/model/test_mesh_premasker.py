from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.data import MultiViewDataset
from rfstudio.data.dataparser import RFMaskedRealDataparser
from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, RGBAImages, RGBImages, TriangleMesh
from rfstudio.graphics.shaders import NormalShader


@dataclass
class Script(Task):

    mesh: Path = ...
    dataset: MultiViewDataset = MultiViewDataset(path=...)
    output: Path = ...

    def run(self) -> None:
        self.dataset.to(self.device)
        mesh = TriangleMesh.from_file(self.mesh).to(self.device)
        # T = torch.tensor([-0.349, -0.158, -0.065]).to(self.device)
        # R = torch.tensor([[ 0.987,  0.119, -0.104],
        #         [ 0.119, -0.135,  0.984],
        #         [ 0.104, -0.984, -0.148]]).to(self.device)
        # S = torch.tensor(0.095).to(self.device)
        T = torch.tensor([-0.655, -0.317,  0.647]).to(self.device)
        R = torch.tensor([[ 0.992,  0.080, -0.097],
            [ 0.080,  0.192,  0.978],
            [ 0.097, -0.978,  0.184]]).to(self.device)
        S = torch.tensor(0.338).to(self.device)
        mesh.replace_(vertices=(R @ (mesh.vertices - T).unsqueeze(-1)).squeeze(-1) * S)
        mesh.export(self.mesh.parent / f'{self.mesh.stem}_adjusted.ply', only_geometry=True)
        inputs = Cameras.cat([
            self.dataset.get_inputs(split='train'),
            self.dataset.get_inputs(split='val'),
            self.dataset.get_inputs(split='test'),
        ], dim=0)
        normals = mesh.render(cameras=inputs, shader=NormalShader())
        gt_outputs = RGBImages(
            [img.item() for img in self.dataset.get_gt_outputs(split='train')] +
            [img.item() for img in self.dataset.get_gt_outputs(split='val')] +
            [img.item() for img in self.dataset.get_gt_outputs(split='test')]
        )
        gt_outputs = RGBAImages([
            torch.cat((rgb * normal[..., 3:], normal[..., 3:]), dim=-1)
            for rgb, normal in zip(gt_outputs, normals)
        ])
        RFMaskedRealDataparser.dump(inputs, gt_outputs, None, path=self.output, split='all')


if __name__ == '__main__':
    Script(cuda=0).run()
