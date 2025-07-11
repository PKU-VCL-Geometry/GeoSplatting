from dataclasses import dataclass

from rfstudio.data import SfMDataset
from rfstudio.engine.task import Task
from rfstudio.visualization import Visualizer


@dataclass
class Tester(Task):

    viser: Visualizer = Visualizer(port=6789)

    dataset: SfMDataset = SfMDataset(path=...)

    def run(self) -> None:
        self.dataset.to(self.device)
        self.viser.show(root=self.dataset)

if __name__ == '__main__':
    Tester().run()
