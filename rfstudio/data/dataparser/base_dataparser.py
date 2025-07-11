from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, Tuple, TypeVar

import torch

from rfstudio.utils.typing import Indexable

DT = TypeVar('DT')
GT = TypeVar('GT')
MT = TypeVar('MT')


@dataclass
class BaseDataparser(ABC, Generic[DT, GT, MT]):

    @abstractmethod
    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[DT], Indexable[GT], MT]:
        ...

    @staticmethod
    @abstractmethod
    def recognize(path: pathlib.Path) -> bool:
        ...

    @staticmethod
    def dump(
        inputs: DT,
        gt_outputs: GT,
        meta: MT,
        *,
        path: pathlib.Path,
        split: Literal['train', 'test', 'val'],
    ) -> None:
        raise NotImplementedError
