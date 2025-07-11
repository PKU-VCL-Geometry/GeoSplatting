"""
Module [rfstudio.data.dataset]

Define dataset class here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from jaxtyping import Bool, Float32, Int32
from torch import Tensor

from rfstudio.graphics import Cameras, DepthImages, RGBAImages, RGBImages, SfMPoints, TriangleMesh
from rfstudio.graphics._2d import Cameras2D, RGBA2DImages
from rfstudio.utils.typing import Indexable, IntArrayLike

from .dataparser import (
    BaseDataparser,
    BlenderDataparser,
    ColmapDataparser,
    DepthBlenderDataparser,
    DPKUDataparser,
    IDRDataparser,
    LLFFDataparser,
    MaskedBlenderDataparser,
    MaskedIDRDataparser,
    MaskedLLFFDataparser,
    MeshDRDataparser,
    MeshPBRDataparser,
    MeshViewSynthesisDataparser,
    RFMaskedRealDataparser,
    ShapeNetDataparser,
    ShinyBlenderDataparser,
    StanfordORBDataparser,
    Syn4RelightDataparser,
    Synthetic2DDataparser,
    TensoIRDataparser,
)
from .selector import BaseSequentialSelector, BaseSpatialSelector

DT = TypeVar('DT')
GT = TypeVar('GT')
MT = TypeVar('MT')


@dataclass
class BaseDataset(ABC, Generic[DT, GT, MT]):

    """
    Base dataset class for loading something from a specific path and parsing it with a data parser.

    Attributes: path, dataparser
    Methods: get_loader
    """

    path: Path
    """
    path to data
    """

    dataparser: Optional[BaseDataparser[DT, GT, MT]] = None
    """
    dataparser to parse data; auto to decide when not specified
    """

    selector: Union[BaseSpatialSelector, BaseSequentialSelector, None] = None

    @classmethod
    @abstractmethod
    def get_dataparser_list(cls) -> List[Type[BaseDataparser[DT, GT, MT]]]:
        ...

    def __setup__(self) -> None:
        if self.dataparser is None:
            for dataparser_class in self.__class__.get_dataparser_list():
                if dataparser_class.recognize(self.path):
                    self.dataparser = dataparser_class()
                    break
            else:
                raise RuntimeError("Cannot decide dataparser automatically.")
        self._pairs: Dict[Literal['train', 'test', 'val'], Tuple[Indexable[DT], Indexable[GT], MT]]
        self._selections: Dict[Literal['train', 'test', 'val'], Tuple[IntArrayLike, MT]]

        self._pairs = {}
        self._selections = {}
        self._device = None

    def to(self, device: torch.device) -> None:
        self._device = device

    def load(self, *, split: Literal['train', 'test', 'val']) -> None:
        if split not in self._pairs:
            assert self.dataparser is not None
            self._pairs[split] = self.dataparser.parse(self.path, split=split, device=self._device)
            if self.selector is not None:
                if isinstance(self.selector, BaseSpatialSelector):
                    indices = self.get_spatial_selection(self.selector.filter, split=split)
                elif isinstance(self.selector, BaseSequentialSelector):
                    indices = self.get_sequential_selection(self.selector.filter, split=split)
                else:
                    raise NotImplementedError
                self._selections[split] = (indices, self.get_selected_meta(indices, split=split))

    def replace_selector_(self, selector: Union[BaseSpatialSelector, BaseSequentialSelector]) -> None:
        splits = self._selections
        self._selections = {}
        for split in splits:
            if isinstance(selector, BaseSpatialSelector):
                indices = self.get_spatial_selection(selector.filter, split=split)
            elif isinstance(selector, BaseSequentialSelector):
                indices = self.get_sequential_selection(selector.filter, split=split)
            else:
                raise NotImplementedError
            self._selections[split] = (indices, self.get_selected_meta(indices, split=split))

    def get_size(self, *, split: Literal['train', 'test', 'val']) -> int:
        return len(self.get_inputs(split=split))

    def get_meta(self, *, split: Literal['train', 'test', 'val']) -> MT:
        self.load(split=split)
        if split in self._selections:
            return self._selections[split][1]
        return self._pairs[split][2]

    def get_inputs(self, *, split: Literal['train', 'test', 'val']) -> Indexable[DT]:
        self.load(split=split)
        dt = self._pairs[split][0]
        if split in self._selections:
            indices = self._selections[split][0]
            dt = dt[indices]
        return dt

    def get_gt_outputs(self, *, split: Literal['train', 'test', 'val']) -> Indexable[GT]:
        self.load(split=split)
        gt = self._pairs[split][1]
        if split in self._selections:
            indices = self._selections[split][0]
            gt = gt[indices]
        return gt

    def get_train_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        infinite: bool = True,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('train', batch_size=batch_size, shuffle=shuffle, infinite=infinite)

    def get_test_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        infinite: bool = False,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('test', batch_size=batch_size, shuffle=shuffle, infinite=infinite)

    def get_val_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        infinite: bool = True,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('val', batch_size=batch_size, shuffle=shuffle, infinite=infinite)

    def get_iter(
        self,
        split: Literal['train', 'test', 'val'],
        *,
        batch_size: int,
        shuffle: bool,
        infinite: bool,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        self.load(split=split)
        data = self.get_inputs(split=split)
        gt = self.get_gt_outputs(split=split)
        idx = 0
        if shuffle:
            indices = torch.randperm(len(data), device=self._device)
        else:
            indices = torch.arange(len(data), device=self._device)
        while True:
            idx += batch_size
            sampled_indices = indices[idx - batch_size:idx]

            yield data[sampled_indices], gt[sampled_indices], sampled_indices

            if idx >= len(data):
                idx = 0
                if not infinite:
                    break
                if shuffle:
                    indices = torch.randperm(len(data), device=self._device)

    def get_sequential_selection(
        self,
        sequential_filter: Callable[[Int32[Tensor, "N 1"]], Bool[Tensor, "N 1"]],
        *,
        split: Literal['train', 'test', 'val']
    ) -> IntArrayLike:
        raise NotImplementedError

    def get_spatial_selection(
        self,
        spatial_filter: Callable[[Float32[Tensor, "N 3"]], Bool[Tensor, "N 1"]],
        *,
        split: Literal['train', 'test', 'val']
    ) -> IntArrayLike:
        raise NotImplementedError

    def get_selected_meta(
        self,
        indices: IntArrayLike,
        *,
        split: Literal['train', 'test', 'val'],
    ) -> MT:
        raise NotImplementedError

    def dump(
        self,
        path: Path,
        *,
        exist_ok: bool = False,
        dataparser: Optional[Type[BaseDataparser[DT, MT, GT]]] = None,
    ) -> None:
        path.mkdir(exist_ok=exist_ok, parents=True)
        if dataparser is None:
            dataparser = self.dataparser.__class__
        for split in ['train', 'test', 'val']:
            dataparser.dump(
                inputs=self.get_inputs(split=split),
                gt_outputs=self.get_gt_outputs(split=split),
                meta=self.get_meta(split=split),
                path=path,
                split=split,
            )

@dataclass
class MultiViewDataset(BaseDataset[Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[BaseDataparser[Cameras, RGBImages, Any]]]:
        return [
            BlenderDataparser,
            DPKUDataparser,
            ColmapDataparser,
            LLFFDataparser,
            IDRDataparser,
        ]

    def get_sequential_selection(
        self,
        sequential_filter: Callable[[Int32[Tensor, "N 1"]], Bool[Tensor, "N 1"]],
        *,
        split: Literal['train', 'test', 'val']
    ) -> IntArrayLike:
        N = self.get_size(split=split)
        indices = torch.arange(N, device=self._device).unsqueeze(-1)
        mask = sequential_filter(indices)
        return indices[mask].cpu().numpy()

    def get_spatial_selection(
        self,
        spatial_filter: Callable[[Float32[Tensor, "N 3"]], Bool[Tensor, "N 1"]],
        *,
        split: Literal['train', 'test', 'val']
    ) -> IntArrayLike:
        cameras: Cameras = self.get_inputs(split=split)
        assert cameras.ndim == 1
        indices = spatial_filter(cameras.c2w[:, :, 3]).squeeze(-1).nonzero().squeeze(-1)
        return indices.cpu().numpy()

    def get_selected_meta(
        self,
        indices: IntArrayLike,
        *,
        split: Literal['train', 'test', 'val'],
    ) -> Any:
        pass


@dataclass
class SfMDataset(BaseDataset[Cameras, RGBImages, SfMPoints]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[BaseDataparser[Cameras, RGBImages, SfMPoints]]]:
        return [
            DPKUDataparser,
            ColmapDataparser,
        ]

    def get_sequential_selection(
        self,
        sequential_filter: Callable[[Int32[Tensor, "N 1"]], Bool[Tensor, "N 1"]],
        *,
        split: Literal['train', 'test', 'val']
    ) -> IntArrayLike:
        N = self.get_size(split=split)
        indices = torch.arange(N, device=self._device).unsqueeze(-1)
        mask = sequential_filter(indices)
        return indices[mask].cpu().numpy()

    def get_spatial_selection(
        self,
        spatial_filter: Callable[[Float32[Tensor, "N 3"]], Bool[Tensor, "N 1"]],
        *,
        split: Literal['train', 'test', 'val']
    ) -> IntArrayLike:
        cameras: Cameras = self.get_inputs(split=split)
        assert cameras.ndim == 1
        indices = spatial_filter(cameras.c2w[:, :, 3]).squeeze(-1).nonzero().squeeze(-1)
        return indices.cpu().numpy()

    def get_selected_meta(
        self,
        indices: IntArrayLike,
        *,
        split: Literal['train', 'test', 'val'],
    ) -> SfMPoints:
        return self.get_meta(split=split).seen_by(torch.from_numpy(indices).to(self._device))


@dataclass
class MeshViewSynthesisDataset(BaseDataset[Cameras, RGBAImages, TriangleMesh]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[BaseDataparser[Cameras, RGBAImages, TriangleMesh]]]:
        return [
            MeshPBRDataparser,
            MeshViewSynthesisDataparser,
            ShinyBlenderDataparser,
            MaskedBlenderDataparser,
            MaskedIDRDataparser,
            MaskedLLFFDataparser,
            ShapeNetDataparser,
            StanfordORBDataparser,
            RFMaskedRealDataparser,
        ]

@dataclass
class DepthSynthesisDataset(BaseDataset[Cameras, DepthImages, TriangleMesh]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[BaseDataparser[Cameras, DepthImages, TriangleMesh]]]:
        return [
            MeshDRDataparser,
            DepthBlenderDataparser,
        ]

@dataclass
class RelightDataset(BaseDataset[Cameras, RGBAImages, Tuple[
    Indexable[RGBAImages],
    Optional[Indexable[RGBAImages]],
    Tuple[Indexable[RGBAImages], ...],
    Tuple[Path, ...],
]]):

    '''
    Meta:
        Albedo: RGBAImages
        Roughness: Optional[RGBAImages]
        Relight 1-N: Tuple[RGBAImages, ...]
        Envmap 1-N: Tuple[Path, ...]
    '''

    @classmethod
    def get_dataparser_list(cls) -> List[Type[BaseDataparser[
        Cameras,
        RGBAImages,
        Tuple[Indexable[RGBAImages], Indexable[RGBAImages], Tuple[Indexable[RGBAImages], ...], Tuple[Path, ...]],
    ]]]:
        return [
            Syn4RelightDataparser,
            TensoIRDataparser,
        ]

@dataclass
class MultiView2DDataset(BaseDataset[Cameras2D, RGBA2DImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[BaseDataparser[Cameras2D, RGBA2DImages, Any]]]:
        return [
            Synthetic2DDataparser,
        ]
