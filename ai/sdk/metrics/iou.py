import torch
from sdk.contracts import MetricAdapter


def box_iou(box1, box2):
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else torch.tensor(0.0)


class MeanIoU(MetricAdapter):
    name = "mean_iou"

    def reset(self):
        self.ious = []

    def update(self, predictions, targets):
        for pred, tgt in zip(predictions, targets):
            p_boxes = pred.get("boxes")
            t_boxes = tgt.get("boxes")

            if p_boxes is None or t_boxes is None:
                continue

            for pb, tb in zip(p_boxes, t_boxes):
                self.ious.append(box_iou(pb.cpu(), tb.cpu()).item())

    def compute(self):
        return {self.name: sum(self.ious) / len(self.ious) if self.ious else 0.0}
