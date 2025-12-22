from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

ImageTensor = torch.Tensor
Images = List[ImageTensor]

Target = Dict[str, torch.Tensor]
Targets = List[Target]

LossDict = Dict[str, torch.Tensor]
Predictions = List[Dict[str, torch.Tensor]]


class SegmentationDatasetAdapter(
    Dataset[Tuple[ImageTensor, Target]],
    ABC,
):
    num_classes: int
    num_attr_classes: int

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[ImageTensor, Target]:
        pass


class TransformAdapter(ABC):
    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        target: Target,
    ) -> Tuple[np.ndarray, Target]:
        pass


class DetectionModelAdapter(ABC):
    @abstractmethod
    def train(self, mode: bool = True) -> "DetectionModelAdapter":
        pass

    @abstractmethod
    def to(self, device: str) -> "DetectionModelAdapter":
        pass

    @abstractmethod
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        pass

    @abstractmethod
    def forward(
        self,
        images: Images,
        targets: Optional[Targets] = None,
    ) -> Union[LossDict, Predictions]:
        pass


class TrainerAdapter(ABC):
    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass
