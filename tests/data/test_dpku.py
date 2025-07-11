from dataclasses import dataclass
from pathlib import Path

from rfstudio.data import SfMDataset
from rfstudio.engine.task import Task
from rfstudio.visualization import Visualizer


@dataclass
class Tester(Task):

    viser: Visualizer = Visualizer(port=6789)

    dataset: SfMDataset = ...

    def run(self) -> None:
        self.viser.show(dpku=self.dataset)

if __name__ == '__main__':
    Tester(dataset=SfMDataset(path=Path('/data/dpku/DigitalPKUBackend/backend/data/projects/0000000087'))).run()
