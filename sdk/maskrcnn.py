import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class MaskRCNNWithAttr(nn.Module):
    def __init__(self, num_classes, num_attr_classes):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT")

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

        self.attr_head = nn.Linear(in_features, num_attr_classes)

    def forward(self, images, targets=None):
        if self.training:
            output = self.model(images, targets)

            # ROI features для attr_head
            # roi_features = self.model.roi_heads.box_head(
            #     self.model.roi_heads.box_roi_pool(
            #         [img for img in images], [t["boxes"] for t in targets], images[0].shape[-2:]
            #     )
            # )

            features = self.model.backbone(images)

            if isinstance(features, torch.Tensor):
                features = {"0": features}

            roi_features = self.model.roi_heads.box_roi_pool(
                features, [t["boxes"] for t in targets], images[0].shape[-2:]
            )
            roi_features = self.model.roi_heads.box_head(roi_features)

            attr_logits = self.attr_head(roi_features)
            attr_labels = torch.cat([t["attr_labels"] for t in targets], dim=0).to(attr_logits.device)
            attr_loss = nn.CrossEntropyLoss()(attr_logits, attr_labels)

            # объединяем с основной loss
            loss = sum(loss for loss in output.values()) + attr_loss
            return {"loss": loss}
        else:
            output = self.model(images)

            if sum(len(o["boxes"]) for o in output) == 0:
                return output

            # вычисление attr_pred
            with torch.no_grad():
                roi_features = self.model.roi_heads.box_head(
                    self.model.roi_heads.box_roi_pool(
                        [img for img in images], [o["boxes"] for o in output], images[0].shape[-2:]
                    )
                )
                attr_logits = self.attr_head(roi_features)
                attr_pred = torch.argmax(attr_logits, dim=1)

            # добавляем предсказание в output
            for o, a in zip(output, attr_pred.split([len(b) for b in [o["boxes"] for o in output]])):
                o["attr_pred"] = a
            return output
