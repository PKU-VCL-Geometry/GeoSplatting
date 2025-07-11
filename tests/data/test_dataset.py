from dataclasses import dataclass

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, RelightDataset, SfMDataset
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.visualization import Visualizer


@dataclass
class SfMTester(Task):

    viser: Visualizer = Visualizer(port=6789)

    dataset: SfMDataset = SfMDataset(path=...)

    def run(self) -> None:
        self.dataset.to(self.device)
        self.viser.show(root=self.dataset)


@dataclass
class MultiViewTester(Task):

    viser: Visualizer = Visualizer(port=6789)

    dataset: MultiViewDataset = MultiViewDataset(path=...)

    def run(self) -> None:
        self.dataset.to(self.device)
        self.viser.show(root=self.dataset)


@dataclass
class MeshViewSynthesisTester(Task):

    viser: Visualizer = Visualizer(port=6789)

    dataset: MeshViewSynthesisDataset = MeshViewSynthesisDataset(path=...)

    def run(self) -> None:
        self.dataset.to(self.device)
        self.viser.show(root=self.dataset)


@dataclass
class RelightTester(Task):

    viser: Visualizer = Visualizer(port=6789)

    dataset: RelightDataset = RelightDataset(path=...)

    def run(self) -> None:
        self.dataset.to(self.device)
        self.viser.show(root=self.dataset)


if __name__ == '__main__':
    TaskGroup(
        sfm=SfMTester(),
        mv=MultiViewTester(),
        mvs=MeshViewSynthesisTester(),
        relight=RelightTester(),
    ).run()
