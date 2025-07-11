from dataclasses import dataclass

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from rfstudio.utils.tensor_dataclass import TensorLike


@dataclass
class BaseLoss:

    def __call__(self, outputs: TensorLike, gt_outputs: TensorLike) -> Float[Tensor, "1"]:
        raise NotImplementedError


@dataclass
class L2Loss(BaseLoss):

    def __call__(self, outputs: Tensor, gt_outputs: Tensor) -> Float[Tensor, "1"]:
        return F.mse_loss(outputs, gt_outputs)


@dataclass
class L1Loss(BaseLoss):

    def __call__(self, outputs: Tensor, gt_outputs: Tensor) -> Float[Tensor, "1"]:
        return F.l1_loss(outputs, gt_outputs)
