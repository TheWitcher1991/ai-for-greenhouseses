from pathlib import Path

import torch
import torch.nn as nn
from sdk.backbone import BackboneBuilder
from sdk.contracts import BackboneConfig, DetectionModelAdapter
from sdk.logger import logger
from torchvision.models.detection import MaskRCNN as MaskRCNNDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNN(nn.Module, DetectionModelAdapter):
    def __init__(self, num_classes: int, num_attr_classes: int, backbone_cfg: BackboneConfig, weights_path: str = None):
        super().__init__()

        logger.info(f"Инициализация MaskRCNN | " f"classes={num_classes}, attr_classes={num_attr_classes}")

        backbone = BackboneBuilder.build(backbone_cfg)

        self.model = MaskRCNNDetection(
            backbone=backbone,
            num_classes=num_classes,
        )

        if weights_path and Path(weights_path).exists():
            logger.info(f"Загрузка весов из файла: {weights_path}")
            checkpoint = torch.load(weights_path)
            self.model.load_state_dict(checkpoint)
            logger.info("Веса успешно загружены")

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

        self.attr_head = nn.Linear(in_features, num_attr_classes)
        self.attr_loss = nn.CrossEntropyLoss()

    def forward(self, images, targets=None):
        if self.training:
            logger.info(f"Forward(train) | batch_size={len(images)}")

            losses = self.model(images, targets)

            logger.info("Base losses: " + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))

            features = self.model.backbone(images)
            if isinstance(features, torch.Tensor):
                features = {"0": features}

            roi = self.model.roi_heads.box_roi_pool(features, [t["boxes"] for t in targets], images[0].shape[-2:])
            roi = self.model.roi_heads.box_head(roi)

            logits = self.attr_head(roi)
            labels = torch.cat([t["attr_labels"].to(images[0].device) for t in targets])

            loss_attr = self.attr_loss(logits, labels)
            losses["loss_attr"] = loss_attr

            logger.info(f"loss_attr={loss_attr.item():.4f}")

            return losses

        logger.info(f"Forward(infer) | batch_size={len(images)}")

        output = self.model(images)

        total_boxes = sum(len(o["boxes"]) for o in output)
        logger.info(f"Detected objects: {total_boxes}")

        if total_boxes == 0:
            logger.warning("Объекты не найдены")
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

        logger.info("Attribute prediction добавлен в output")

        return output
