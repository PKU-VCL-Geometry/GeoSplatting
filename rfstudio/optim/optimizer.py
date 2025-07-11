from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Type, Union

import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from rfstudio.utils.decorator import chains


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    half_life: Optional[int],
    warm_up: Optional[int],
    mode: Literal['exp', 'cos'],
) -> LRScheduler:

    def exp_decay(step: int) -> Any:
        if warm_up is not None:
            if step < warm_up:
                return (step / warm_up) ** 2
            if half_life is None:
                return 1
        assert half_life is not None
        lambda_ = np.log(2) / half_life
        return np.exp(-lambda_ * (step - (0 if warm_up is None else warm_up)))

    def cos_decay(step: int) -> Any:
        if warm_up is not None:
            if step < warm_up:
                return step / warm_up
            if half_life is None:
                return 1
        progress = (step - (0 if warm_up is None else warm_up)) / half_life
        alpha = 0.05
        return (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

    scheduler = LambdaLR(optimizer, lr_lambda=exp_decay if mode == 'exp' else cos_decay)
    return scheduler


@dataclass
class Optimizer:

    category: Type[torch.optim.Optimizer]

    modules: Union[torch.nn.Module, List[torch.nn.Module]]

    lr: float

    eps: float = 1e-15

    max_norm: Optional[float] = None

    lr_decay: Optional[int] = None

    warm_up: Optional[int] = None

    lr_decay_mode: Literal['exp', 'cos'] = 'exp'

    def get_parameters(self) -> List[torch.Tensor]:
        modules = self.modules if isinstance(self.modules, list) else [self.modules]
        return sum([list(m.parameters()) for m in modules], [])


class ModuleOptimizers:

    def __init__(
        self,
        mixed_precision: bool,
        optim_dict: Dict[str, Optimizer],
    ) -> None:
        self.optim_dict = optim_dict
        self.grad_scaler = GradScaler(enabled=mixed_precision)
        self.optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None

    def zero_grad(self) -> None:
        if self.optimizers is not None:
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        if self.optimizers is None:
            self.parameters = {
                key: optim.get_parameters()
                for key, optim in self.optim_dict.items()
            }
            self.optimizers = {
                key: optim.category(self.parameters[key], lr=optim.lr, eps=optim.eps)
                for key, optim in self.optim_dict.items()
            }
            self.schedulers = {
                key: get_scheduler(
                    self.optimizers[key],
                    half_life=optim.lr_decay,
                    warm_up=optim.warm_up,
                    mode=optim.lr_decay_mode,
                )
                for key, optim in self.optim_dict.items()
                if optim.lr_decay is not None or optim.warm_up is not None
            }
        self.grad_scaler.scale(loss).backward()

    def step(self) -> None:
        assert self.optimizers is not None
        for key, optimizer in self.optimizers.items():
            max_norm = self.optim_dict[key].max_norm
            if max_norm is not None:
                self.grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters[key], max_norm, error_if_nonfinite=True)
            self.grad_scaler.step(optimizer)
        self.grad_scaler.update()
        for scheduler in self.schedulers.values():
            scheduler.step()

    @chains
    def __getitem__(self, key: Union[str, Iterable[str]]) -> Any:

        keys = [key] if isinstance(key, str) else key
        assert isinstance(keys, Iterable)

        @torch.no_grad()
        def mutate_params(
            self: ModuleOptimizers,
            *,
            clear: bool = False,
            reset: bool = False,
            indices: Optional[Int[Tensor, "N ndim"]] = None,
        ) -> None:
            assert self.optimizers is not None
            assert reset or clear or indices is not None
            for key in keys:
                optimizer = self.optimizers[key]
                optim = self.optim_dict[key]
                category = optim.category

                new_params = list(optim.get_parameters())
                self.parameters[key] = new_params

                if reset:
                    self.optimizers[key] = category(new_params, lr=optim.lr, eps=optim.eps)
                    if key in self.schedulers:
                        self.schedulers[key] = get_scheduler(
                            self.optimizers[key],
                            half_life=optim.lr_decay,
                            warm_up=optim.warm_up,
                        )
                    continue

                if category is torch.optim.Adam:
                    items = ['exp_avg', 'exp_avg_sq']
                else:
                    raise NotImplementedError

                if indices is not None:
                    index_slice = tuple(
                        indices[:, i]
                        for i in range(indices.shape[1])
                    ) + (..., )
                    for param, new_param in zip(optimizer.param_groups[0]["params"], new_params):
                        param_state = optimizer.state[param]
                        for item in items:
                            if item not in param_state:
                                continue
                            # param_state[item]: [..., *C]
                            new_state = param_state[item][index_slice] # [N, *C]
                            new_state[(indices < 0).any(1), ...] = 0
                            param_state[item] = new_state
                        del optimizer.state[param]
                        optimizer.state[new_param] = param_state

                if clear:
                    for param in optimizer.param_groups[0]["params"]:
                        param_state = optimizer.state[param]
                        for item in items:
                            if item not in param_state:
                                continue
                            param_state[item][...] = 0

                optimizer.param_groups[0]["params"] = new_params

        return mutate_params

    def mutate_params(self, *, clear: bool = False, reset: bool = False, indices: Optional[Int[Tensor, "N ndim"]] = None) -> None:
        assert self.optimizers is not None
        self[self.optimizers.keys()].mutate_params(clear=clear, reset=reset, indices=indices)
