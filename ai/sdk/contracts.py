from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
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


class BackboneType(StrEnum):
    resnet50 = auto()
    resnet101 = auto()
    resnet152 = auto()
    # efficientnet_b3 = auto()
    # efficientnet_b4 = auto()
    # convnext = auto()


@dataclass
class BackboneConfig:
    name: BackboneType
    pretrained: bool = True


class BackboneSpec(Protocol):
    backbone: nn.Module
    out_channels: int


class BackboneAdapter(ABC):
    out_channels: int

    @abstractmethod
    def build(self, cfg: BackboneConfig) -> nn.Module:
        pass


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
    def save(self, model_path: str, labels_path: str, metrics_path: str) -> None: ...

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
    def validate(self, *args, **kwargs) -> None: ...


class MetricResult(Dict[str, float]):
    pass


class ModelConfig(TypedDict):
    architecture: str
    num_classes: int
    num_attr_classes: int
    object_labels: List[str]
    object_attrs: List[str]
    weights_storage: str
    weights_path: str


class RegistryCredentials(TypedDict):
    host: str
    login: str
    password: str
    output_annotations: str
    output_images: str


class RegistryAdapter(ABC):

    @abstractmethod
    def find_annotations(self) -> List[TypedDict]: ...

    @abstractmethod
    def find_annotation(self, annotation_id: int) -> TypedDict: ...

    @abstractmethod
    def save_annotations(self) -> None: ...


class MetricAdapter(ABC):
    name: str

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, preds: Any, targets: Any) -> None: ...

    @abstractmethod
    def compute(self) -> Dict: ...
