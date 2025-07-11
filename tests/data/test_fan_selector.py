from dataclasses import dataclass
from pathlib import Path

from rfstudio.data import SfMDataset
from rfstudio.data.selector import FanSelector
from rfstudio.engine.task import Task
from rfstudio.visualization import Visualizer


@dataclass
class Tester(Task):

    viser: Visualizer = Visualizer(port=6789)

    path: Path = ...

    dataset_0: SfMDataset = SfMDataset(
        path=...,
        selector=FanSelector(fan_index=0, num_fans=5, num_fans_per_cover=2)
    )

    dataset_1: SfMDataset = SfMDataset(
        path=...,
        selector=FanSelector(fan_index=1, num_fans=5, num_fans_per_cover=2)
    )

    dataset_2: SfMDataset = SfMDataset(
        path=...,
        selector=FanSelector(fan_index=2, num_fans=5, num_fans_per_cover=2)
    )

    dataset_3: SfMDataset = SfMDataset(
        path=...,
        selector=FanSelector(fan_index=3, num_fans=5, num_fans_per_cover=2)
    )

    dataset_4: SfMDataset = SfMDataset(
        path=...,
        selector=FanSelector(fan_index=4, num_fans=5, num_fans_per_cover=2)
    )

    dataset_5: SfMDataset = SfMDataset(
        path=...,
    )

    def __post_init__(self) -> None:
        self.dataset_0.path = self.path
        self.dataset_1.path = self.path
        self.dataset_2.path = self.path
        self.dataset_3.path = self.path
        self.dataset_4.path = self.path
        self.dataset_5.path = self.path

    def run(self) -> None:
        self.dataset_0.to(self.device)
        self.dataset_1.to(self.device)
        self.dataset_2.to(self.device)
        self.dataset_3.to(self.device)
        self.dataset_4.to(self.device)
        self.dataset_5.to(self.device)
        self.viser.show(
            d0=self.dataset_0,
            d1=self.dataset_1,
            d2=self.dataset_2,
            d3=self.dataset_3,
            d4=self.dataset_4,
            full=self.dataset_5
        )

if __name__ == '__main__':
    Tester(path=Path('data/tnt/barn')).run()
