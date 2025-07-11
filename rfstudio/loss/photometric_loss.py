from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
from jaxtyping import Float
from torch import Tensor
from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity,
    structural_similarity_index_measure,
)

from rfstudio.graphics import PBRAImages, PBRImages, RGBAImages, RGBImages

from .base_loss import BaseLoss, L1Loss, L2Loss


@dataclass
class BasePhotometricLoss(BaseLoss):

    data_range: float = 1.0

    def __call__(
        self,
        outputs: Union[RGBImages, PBRImages],
        gt_outputs: Union[RGBImages, PBRImages],
    ) -> Float[Tensor, "1"]:
        results = []
        for output, gt_output in zip(outputs, gt_outputs, strict=True):
            results.append(self._impl(output, gt_output))
        return torch.stack(results).mean()

    def _impl(self, output: Float[Tensor, "H W 3"], gt_output: Float[Tensor, "H W 3"]) -> Float[Tensor, "1"]:
        raise NotImplementedError


@dataclass
class ImageL1Loss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return L1Loss.__call__(self, output, gt_output) / self.data_range ** 2


@dataclass
class ImageL2Loss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return L2Loss.__call__(self, output, gt_output) / self.data_range ** 2


@dataclass
class PSNRLoss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return -10 * (L2Loss.__call__(self, output, gt_output) / self.data_range ** 2).log10()


@dataclass
class SSIMLoss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return 1 - structural_similarity_index_measure(
            gt_output.permute(2, 0, 1)[None],
            output.permute(2, 0, 1)[None],
            data_range=self.data_range,
        )


@dataclass
class LPIPSLoss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return learned_perceptual_image_patch_similarity(
            gt_output.permute(2, 0, 1)[None] / self.data_range,
            output.permute(2, 0, 1)[None] / self.data_range,
            normalize=True,
        )

@dataclass
class SSIML1Loss(BasePhotometricLoss):

    ssim_lambda: float = 0.2

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        ssim_loss = SSIMLoss._impl(self, output, gt_output)
        l1_loss = L1Loss.__call__(self, output, gt_output)
        return ssim_loss * self.ssim_lambda + l1_loss * (1.0 - self.ssim_lambda)


@dataclass
class MaskedPhotometricLoss(BaseLoss):

    photometric_term: BasePhotometricLoss = ...
    coverage_coeff: float = 1.0
    coverage_loss: BaseLoss = L2Loss()

    def __call__(
        self,
        *,
        outputs: Union[RGBAImages, PBRAImages],
        gt_outputs: RGBAImages,
    ) -> Float[Tensor, "1"]:
        results = []
        if isinstance(outputs, PBRAImages):
            gt_outputs = gt_outputs.srgb2rgb()
        for output, gt_output in zip(outputs, gt_outputs, strict=True):
            results.append(
                torch.add(
                    self.photometric_term._impl(
                        output[..., :3] * gt_output[..., 3:],
                        gt_output[..., :3] * gt_output[..., 3:]
                    ),
                    self.coverage_coeff * self.coverage_loss(output[..., 3:], gt_output[..., 3:])
                )
            )
        return torch.stack(results).mean()


@dataclass
class HDRLoss(BasePhotometricLoss):

    def _rgb2srgb(self, f: Tensor) -> Tensor:
        return torch.where(
            f > 0.0031308,
            torch.pow(torch.clamp(f, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * f,
        )

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:

        return L1Loss()(
            self._rgb2srgb(torch.log(output.clamp(0, 65535) + 1)) / self.data_range,
            self._rgb2srgb(torch.log(gt_output.clamp(0, 65535) + 1)) / self.data_range,
        )
