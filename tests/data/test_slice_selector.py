from dataclasses import dataclass
from pathlib import Path

from rfstudio.data import SfMDataset
from rfstudio.data.selector import SliceSelector
from rfstudio.engine.task import Task
from rfstudio.visualization import Visualizer


@dataclass
class VisBlock(Task):

    num_slices: int = 4

    path: Path = Path('data') / 'wild'

    viser: Visualizer = Visualizer(port=6789)

    def run(self) -> None:
        datasets = {
            f'block{i}': SfMDataset(
                path=self.path,
                selector=SliceSelector(i, self.num_slices, cover_factor=1.5)
            )
            for i in range(self.num_slices)
        }
        for d in datasets.values():
            d.__setup__()
            d.to(self.device)
        self.viser.show(**datasets)


if __name__ == '__main__':
    VisBlock().run()
