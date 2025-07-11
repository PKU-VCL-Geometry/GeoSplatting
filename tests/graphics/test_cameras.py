from pathlib import Path

import torch

from rfstudio.data import MultiViewDataset
from rfstudio.graphics import Cameras
from rfstudio.visualization import Visualizer


def test_sample_sequentially():
    dataset = MultiViewDataset(path=Path('data') / 'blender' / 'lego')
    dataset.__setup__()
    cameras = dataset.get_inputs(split='test')
    paths = cameras.sample_sequentially(num_samples=1000, uniform_by='distance')
    Visualizer(port=6789).show(paths=paths)

def test_hemisphere():
    cameras111 = Cameras.from_hemisphere(
        center=(0, 1, 0),
        up=(1, 1, 1),
        radius=0.5,
        num_samples=128
    )
    cameras001 = Cameras.from_hemisphere(
        center=(0, 0, 0),
        up=(0, 0, 1),
        radius=1,
        num_samples=128,
        device=torch.device('cuda:0')
    )
    Visualizer(port=6789).show(cam111=cameras111, cam001=cameras001)

if __name__ == '__main__':
    test_sample_sequentially()
    test_hemisphere()
