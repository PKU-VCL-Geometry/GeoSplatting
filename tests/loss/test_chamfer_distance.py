from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Points, TriangleMesh
from rfstudio.loss import ChamferDistanceFscoreMetric
from rfstudio.ui import console
from rfstudio.utils.pretty import P


@dataclass
class Mesh2Mesh(Task):

    a: Path = ...

    b: Path = ...

    num_chamfer_samples: Optional[int] = 1000000

    threshold: float = 0.001

    @torch.no_grad()
    def run(self) -> None:
        mesh_a = TriangleMesh.from_file(self.a).to(self.device)
        mesh_b = TriangleMesh.from_file(self.b).to(self.device)
        threshold = self.threshold
        with console.status(desc='Computing Chamfer Distance'):
            chamfer, fscore = ChamferDistanceFscoreMetric(
                target_num_points=self.num_chamfer_samples,
                threshold=threshold,
            )(mesh_a, mesh_b)
        console.print(P@'Chamfer Distance: {chamfer:.6f}')
        console.print(P@'Fscore @ {threshold}: {fscore:.6f}')

@dataclass
class Point2Mesh(Task):

    a: Path = ...

    b: Path = ...

    num_chamfer_samples: Optional[int] = 1000000

    threshold: float = 0.001

    @torch.no_grad()
    def run(self) -> None:
        pc_a = Points.from_file(self.a).to(self.device)
        mesh_b = TriangleMesh.from_file(self.b).to(self.device)
        threshold = self.threshold
        with console.status(desc='Computing Chamfer Distance'):
            chamfer, fscore = ChamferDistanceFscoreMetric(
                target_num_points=self.num_chamfer_samples,
                threshold=threshold,
            )(pc_a, mesh_b)
        console.print(P@'Chamfer Distance: {chamfer:.6f}')
        console.print(P@'Fscore @ {threshold}: {fscore:.6f}')

@dataclass
class Point2Point(Task):

    a: Path = ...

    b: Path = ...

    num_chamfer_samples: Optional[int] = 1000000

    threshold: float = 0.001

    @torch.no_grad()
    def run(self) -> None:
        pc_a = TriangleMesh.from_file(self.a).to(self.device)
        pc_b = TriangleMesh.from_file(self.b).to(self.device)
        threshold = self.threshold
        with console.status(desc='Computing Chamfer Distance'):
            chamfer, fscore = ChamferDistanceFscoreMetric(
                target_num_points=self.num_chamfer_samples,
                threshold=threshold,
            )(pc_a, pc_b)
        console.print(P@'Chamfer Distance: {chamfer:.6f}')
        console.print(P@'Fscore @ {threshold}: {fscore:.6f}')

if __name__ == '__main__':
    TaskGroup(
        m2m=Mesh2Mesh(cuda=0),
        p2m=Point2Mesh(cuda=0),
        p2p=Point2Point(cuda=0),
    ).run()
