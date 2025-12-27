import torch
from framework.backbone import BackboneBuilder
from framework.contracts import BackboneConfig, BackboneType
from framework.detection.v1.dataset.coco import CocoSegmentationDataset
from framework.transforms import ComposeTransforms
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
EPOCHS = 1

dataset = CocoSegmentationDataset(
    images_dir="data/v1/images", annotation_file="data/v1/annotations.json", transforms=ComposeTransforms()
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

backbone_cfg = BackboneConfig(name=BackboneType.resnet50, pretrained=True)

backbone = BackboneBuilder.build(backbone_cfg)

num_classes = dataset.num_classes

model = MaskRCNN(
    backbone=backbone,
    num_classes=num_classes,
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler(enabled=USE_AMP)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for step, (images, targets, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        try:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        except Exception as e:
            print(f"Ошибка при обработке батча: {e}")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
