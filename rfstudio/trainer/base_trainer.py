from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.data import BaseDataset
from rfstudio.nn import Module
from rfstudio.optim import ModuleOptimizers


@dataclass
class BaseTrainer(ABC):

    num_steps: int = ...

    num_steps_per_vis: Optional[int] = None

    num_steps_per_val: Optional[int] = None

    num_steps_per_save: Optional[int] = None

    full_test_after_train: bool = True

    hold_after_train: bool = True

    batch_size: int = ...

    mixed_precision: bool = False

    detect_anomaly: bool = False

    num_accums_per_batch: int = 1

    @abstractmethod
    def setup(self, model: Module, dataset: BaseDataset[Any, Any, Any]) -> ModuleOptimizers:
        ...

    @abstractmethod
    def step(
        self,
        model: Module,
        inputs: Any,
        gt_outputs: Any,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        ...

    def visualize(
        self,
        model: Module,
        inputs: Any,
    ) -> Tensor:
        return torch.zeros(1, 1, 3)

    def before_update(
        self,
        model: Module,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        pass

    def after_update(
        self,
        model: Module,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        pass

    def after_backward(self, model: Module) -> None:
        pass

    def after_train(self, model: Module, dataset: BaseDataset[Any, Any, Any]) -> None:
        pass
