from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, RelightDataset
from rfstudio.data.dataparser import IDRDataparser, MaskedBlenderDataparser, MaskedIDRDataparser
from rfstudio.engine.task import Task, TaskGroup


@dataclass
class MVS2Blender(Task):

    dataset: MeshViewSynthesisDataset = MeshViewSynthesisDataset(path=...)

    output: Path = ...

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=MaskedBlenderDataparser)

@dataclass
class Relight2Blender(Task):

    dataset: RelightDataset = RelightDataset(path=...)

    output: Path = ...

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=MaskedBlenderDataparser)

@dataclass
class MVS2IDR(Task):

    dataset: MeshViewSynthesisDataset = MeshViewSynthesisDataset(path=...)

    output: Path = ...

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=MaskedIDRDataparser)

@dataclass
class MV2IDR(Task):

    dataset: MultiViewDataset = MultiViewDataset(path=...)

    output: Path = ...

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=IDRDataparser)

if __name__ == '__main__':
    TaskGroup(
        mvs2blender=MVS2Blender(cuda=0),
        mvs2idr=MVS2IDR(cuda=0),
        mv2idr=MV2IDR(cuda=0),
        relight2blender=Relight2Blender(cuda=0),
    ).run()
