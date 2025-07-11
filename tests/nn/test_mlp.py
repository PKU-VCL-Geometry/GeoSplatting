from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from rfstudio.nn import MLP


def test_optimize(initialization: Literal['default', 'kaiming-uniform']):
    torch.manual_seed(1)

    mlp1 = MLP(layers=[-1, 3, 1], initialization=initialization)
    mlp1.__setup__()
    mlp1.cuda()
    mlp2 = MLP(layers=[3, 3, 1], initialization=initialization)
    mlp2.__setup__()
    mlp2.cuda()
    mlp3 = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 1)).cuda()

    opt = None

    B = 64
    for _ in range(1000):
        x = torch.rand(B, 3).cuda()
        y = (x[..., 0] ** 3 - x[..., 1] * 2 + x[..., 2] - torch.pi).unsqueeze(-1)

        if opt is not None:
            opt[0].zero_grad()
            opt[1].zero_grad()
            opt[2].zero_grad()

        y1 = mlp1(x)
        y2 = mlp2(x)
        y3 = mlp3(x)

        if opt is None:
            opt = (
                torch.optim.Adam(mlp1.parameters(), lr=1e-2),
                torch.optim.Adam(mlp2.parameters(), lr=1e-2),
                torch.optim.Adam(mlp3.parameters(), lr=1e-2),
            )

        loss1 = F.mse_loss(y1, y)
        loss1.backward()
        opt[0].step()

        loss2 = F.mse_loss(y2, y)
        loss2.backward()
        opt[1].step()

        loss3 = F.mse_loss(y3, y)
        loss3.backward()
        opt[2].step()

        if _ % 10 == 0:
            print(loss1.item(), loss2.item(), loss3.item())


    print(loss1.item(), loss2.item(), loss3.item())


if __name__ == '__main__':
    mlp = MLP(layers=[-1, 3, 4, 1])
    mlp.__setup__()
    mlp.cuda()
    assert mlp.training
    mlp.eval()
    assert not mlp.training
    data = torch.rand((5, 9)).cuda()
    assert mlp(data).shape == (5, 1)
    assert mlp(data).device == data.device

    dummy = MLP(layers=[-1, 3, 4, 1])
    dummy.__setup__()
    dummy.load_state_dict(mlp.state_dict())

    test_optimize('default')
    test_optimize('kaiming-uniform')
