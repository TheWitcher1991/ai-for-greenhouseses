import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from dataset import CocoSegmentationDataset
from transforms import ComposeTransforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-4

dataset = CocoSegmentationDataset(
    images_dir="data/images", annotation_file="data/annotations.json", transforms=ComposeTransforms()
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = maskrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = dataset.num_classes

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler(device=DEVICE)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        try:
            with torch.amp.autocast(device_type=DEVICE):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
        except Exception as e:
            print(f"Ошибка при обработке батча: {e}")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
print("Модель сохранена как model.pth")
