from typing import Dict, List, Optional

from framework.contracts import MetricAdapter
from framework.metrics.calibration import ConfidenceStats
from framework.metrics.classification import DetectionAccuracy
from framework.metrics.dice import MeanDice
from framework.metrics.iou import MeanIoU


class MetricsRegistry(MetricAdapter):

    def __init__(self):
        self.metrics: List[MetricAdapter] = [MeanIoU(), MeanDice(), ConfidenceStats(), DetectionAccuracy()]
        self.history: List[Dict] = []

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def update(self, predictions, targets):
        for metric in self.metrics:
            try:
                metric.update(predictions, targets)
            except Exception as e:
                pass

    def compute(self) -> Dict:
        result = {}

        for metric in self.metrics:
            try:
                result.update(metric.compute())
            except Exception as e:
                result[metric.name] = f"error: {e}"

        self.history.append(result)
        return result

    def last(self) -> Optional[Dict]:
        return self.history[-1] if self.history else None

    def all(self) -> List[Dict]:
        return self.history
