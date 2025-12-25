from sdk.contracts import MetricAdapter, MetricResult


class DetectionAccuracy(MetricAdapter):
    name = "accuracy"

    def __init__(self):
        self.reset()

    def reset(self):
        self.correct_class = 0
        self.total_class = 0

        self.correct_attr = 0
        self.total_attr = 0

    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            pred_labels = pred.get("labels", [])
            gt_labels = target.get("labels", [])

            pred_attrs = pred.get("attr_pred", [])
            gt_attrs = target.get("attr_labels", [])

            n = min(len(pred_labels), len(gt_labels))
            if n > 0:
                pred_labels = pred_labels[:n]
                gt_labels = gt_labels[:n]

                self.correct_class += int((pred_labels == gt_labels).sum())
                self.total_class += n

            m = min(len(pred_attrs), len(gt_attrs))
            if m > 0:
                pred_attrs = pred_attrs[:m]
                gt_attrs = gt_attrs[:m]

                self.correct_attr += int((pred_attrs == gt_attrs).sum())
                self.total_attr += m

    def compute(self) -> MetricResult:
        return {
            "accuracy_class": (self.correct_class / self.total_class if self.total_class > 0 else 0.0),
            "accuracy_severity": (self.correct_attr / self.total_attr if self.total_attr > 0 else 0.0),
        }
