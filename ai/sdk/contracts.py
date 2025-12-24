from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

ImageTensor = torch.Tensor
Images = List[ImageTensor]

LossDict = Dict[str, torch.Tensor]

DetectionTarget = Dict[str, torch.Tensor]
DetectionTargets = List[DetectionTarget]

DetectionPrediction = Dict[str, torch.Tensor]
DetectionPredictions = List[DetectionPrediction]


class StorageType(StrEnum):
    json = auto()
    pg = auto()


class SegmentationDatasetAdapter(
    Dataset[Tuple[ImageTensor, DetectionTarget]],
    ABC,
):
    num_classes: int

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[ImageTensor, DetectionPrediction]: ...


class TransformAdapter(ABC):
    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        target: DetectionTarget,
    ) -> Tuple[np.ndarray, DetectionTarget]: ...


class DetectionModelAdapter(ABC):
    @abstractmethod
    def train(self, mode: bool) -> "DetectionModelAdapter": ...

    @abstractmethod
    def to(self, device: str) -> "DetectionModelAdapter": ...

    @abstractmethod
    def parameters(self) -> Iterable[torch.nn.Parameter]: ...

    @abstractmethod
    def forward(
        self,
        images: Images,
        targets: Optional[DetectionTargets] = None,
    ) -> Union[LossDict, DetectionPredictions]: ...


class TrainerAdapter(ABC):
    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def save(self, model_path: str, labels_path: str) -> None: ...

    @abstractmethod
    def load(self, model_path: str, labels_path: str) -> None: ...

    @abstractmethod
    def predict(self, image_path: str, score_threshold: float) -> None: ...


class StorageAdapter(ABC):
    @abstractmethod
    def save(self, path: str, state_dict: Dict) -> None: ...

    @abstractmethod
    def load(self, path: str) -> Any: ...


class DatasetValidatorAdapter(ABC):

    @abstractmethod
    def validate(self) -> None: ...
