from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from sdk.contracts import DetectionModelAdapter


class MaskRCNN(nn.Module, DetectionModelAdapter):
    def __init__(self, num_classes: int, num_attr_classes: int, weights_path: str = None):
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

        self.attr_head = nn.Linear(in_features, num_attr_classes)

    def forward(self, images, targets=None):
        if self.training:
            losses = self.model(images, targets)

            features = self.model.backbone(images)
            if isinstance(features, torch.Tensor):
                features = {"0": features}

            roi = self.model.roi_heads.box_roi_pool(features, [t["boxes"] for t in targets], images[0].shape[-2:])
            roi = self.model.roi_heads.box_head(roi)

            logits = self.attr_head(roi)
            labels = torch.cat([t["attr_labels"] for t in targets])

            losses["loss_attr"] = nn.CrossEntropyLoss()(logits, labels)
            return losses

        output = self.model(images)
        if sum(len(o["boxes"]) for o in output) == 0:
            return output

        features = self.model.backbone(images)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        roi = self.model.roi_heads.box_roi_pool(features, [o["boxes"] for o in output], images[0].shape[-2:])
        roi = self.model.roi_heads.box_head(roi)

        attr = self.attr_head(roi).argmax(1)

        idx = 0
        for o in output:
            n = len(o["boxes"])
            o["attr_pred"] = attr[idx : idx + n]
            idx += n

        return output
