import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from dataset import CocoSegmentationDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CocoSegmentationDataset(images_dir="data/images", annotation_file="data/cvat_merged_coco.json")

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = maskrcnn_resnet50_fpn(pretrained=True)

# 2 класса + фон
num_classes = 3

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for images, targets in tqdm(loader):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
