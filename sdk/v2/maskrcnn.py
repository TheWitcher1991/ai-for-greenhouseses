from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from sdk.contracts import DetectionModelAdapter


class MaskRCNN(nn.Module, DetectionModelAdapter):
    def __init__(self, num_classes: int, num_disease: int, num_severity: int, weights_path: str = None):
        super().__init__()

        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        if weights_path and Path(weights_path).exists():
            self.model = maskrcnn_resnet50_fpn(weights=None)
            checkpoint = torch.load(weights_path)
            self.model.load_state_dict(checkpoint)
        else:
            self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

        self.disease_head = nn.Linear(in_features, num_disease)
        self.severity_head = nn.Linear(in_features, num_severity)

    def forward(self, images, targets=None):
        if self.training:
            losses = self.model(images, targets)

            features = self.model.backbone(images)
            if isinstance(features, torch.Tensor):
                features = {"0": features}

            roi = self.model.roi_heads.box_roi_pool(features, [t["boxes"] for t in targets], images[0].shape[-2:])
            roi = self.model.roi_heads.box_head(roi)

            disease_logits = self.disease_head(roi)
            severity_logits = self.severity_head(roi)

            disease_gt = torch.cat([t["disease"] for t in targets])
            severity_gt = torch.cat([t["severity"] for t in targets])

            loss_attr = nn.CrossEntropyLoss()(disease_logits, disease_gt) + nn.CrossEntropyLoss()(
                severity_logits, severity_gt
            )

            losses["loss_attr"] = loss_attr

            return losses

        output = self.model(images)
        if sum(len(o["boxes"]) for o in output) == 0:
            return output

        with torch.no_grad():
            features = self.model.backbone(images)
            if isinstance(features, torch.Tensor):
                features = {"0": features}

            roi = self.model.roi_heads.box_roi_pool(features, [o["boxes"] for o in output], images[0].shape[-2:])
            roi = self.model.roi_heads.box_head(roi)

            disease = self.disease_head(roi).argmax(1)
            severity = self.severity_head(roi).argmax(1)

        idx = 0
        for o in output:
            n = len(o["boxes"])
            o["disease"] = disease[idx : idx + n]
            o["severity"] = severity[idx : idx + n]
            idx += n

        return output
