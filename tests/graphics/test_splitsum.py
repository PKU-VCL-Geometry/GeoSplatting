from __future__ import annotations

import torch

from rfstudio.graphics import TextureCubeMap

if __name__ == '__main__':
    device = torch.device('cuda:1')
    R = 64
    a = torch.rand(6, R, R, 3).to(device)
    envmap = TextureCubeMap(data=a, transform=None).as_splitsum()
    assert envmap.base.isfinite().all()
    assert envmap.mipmaps.isfinite().all()
