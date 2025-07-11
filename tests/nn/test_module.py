from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from rfstudio.nn import Module

if __name__ == '__main__':

    @dataclass
    class MLP(Module):

        a: torch.nn.Parameter

        b: Optional[torch.nn.Module]

        def __setup__(self) -> None:
            self.c = torch.nn.Linear(2, 3)

        def __call__(self) -> Any:
            self.d = torch.nn.Linear(3, 4)


    mlp1 = MLP(
        a = torch.nn.Parameter(torch.zeros(5)),
        b = None
    )
    mlp1.__setup__()
    named_params = dict(mlp1.named_parameters())
    assert named_params['a'].shape == (5,)

    mlp2 = MLP(
        a = torch.nn.Parameter(torch.zeros(7)),
        b = mlp1
    )
    mlp2.__setup__()
    named_params = dict(mlp2.named_parameters())
    assert named_params['a'].shape == (7,)
    assert named_params['b.a'].shape == (5,)

    mlp2.cuda()
    assert mlp2.device == mlp1.a.device
