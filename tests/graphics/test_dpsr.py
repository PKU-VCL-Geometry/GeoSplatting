from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Points, TriangleMesh
from rfstudio.ui import console
from rfstudio.visualization import Visualizer


@dataclass
class Mesh2Mesh(Task):

    model: Path = ...
    viser: Visualizer = Visualizer(port=6789)
    resolution: int = 256
    smoothness: int = 10
    subdivide: bool = False

    @torch.no_grad()
    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.model).to(self.device).normalize()
        if self.subdivide:
            mesh = mesh.subdivide()
        if mesh.face_normals is None:
            mesh.compute_face_normals_(fix=True)
        pts = Points(
            positions=mesh.vertices[mesh.indices.flatten(), :].view(-1, 3, 3).mean(-2), # [F, 3]
            normals=mesh.face_normals.mean(-2), # [F, 3]
        )
        with console.status('DPSR'):
            fused = TriangleMesh.from_diff_poisson_reconstruction(
                pts,
                resolution=self.resolution,
                smoothness=self.smoothness,
            )
        with self.viser.customize() as handle:
            handle['gt'].show(mesh)
            handle['fused'].show(fused)

if __name__ == '__main__':
    Mesh2Mesh(cuda=0).run()
