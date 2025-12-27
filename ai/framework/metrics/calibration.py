from framework.contracts import MetricAdapter


class ConfidenceStats(MetricAdapter):
    name = "confidence"

    def reset(self):
        self.scores = []

    def update(self, predictions, targets=None):
        for pred in predictions:
            if "scores" in pred:
                self.scores.extend(pred["scores"].detach().cpu().tolist())

    def compute(self):
        if not self.scores:
            return {
                "confidence_mean": 0.0,
                "confidence_max": 0.0,
            }

        return {
            "confidence_mean": sum(self.scores) / len(self.scores),
            "confidence_max": max(self.scores),
        }
