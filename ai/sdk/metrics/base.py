from abc import ABC, abstractmethod
from ast import Dict


class MetricAdapter(ABC):
    name: str

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self, preds, targets): ...

    @abstractmethod
    def compute(self) -> Dict: ...
