import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = maskrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 12
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

LABELS = {
    1: "Здоровый лист 0",
    2: "Лист, зараженный мучнистой росой 1",
    3: "Лист, зараженный мучнистой росой 2",
    4: "Лист, зараженный мучнистой росой 3",
    5: "Лист, зараженный мучнистой росой 4",
    6: "Лист, зараженный мучнистой росой 5",
    7: "Лист, зараженный мучнистой росой 6",
    8: "Лист, зараженный мучнистой росой 7",
    9: "Лист, зараженный мучнистой росой 8",
    10: "Стебель",
    11: "Плод",
}


def predict(image_path, score_threshold=0.5):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)[0]

    results = []
    for score, label, mask in zip(output["scores"], output["labels"], output["masks"]):
        if score < score_threshold:
            continue

        results.append({"label": LABELS[int(label)], "score": float(score), "mask": mask[0].cpu().numpy()})

    return results
