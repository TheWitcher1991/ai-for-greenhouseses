import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def imread_unicode(path: str):
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class CocoSegmentationDataset(Dataset):
    def __init__(self, images_dir, annotation_file):
        self.coco = COCO(annotation_file)
        self.images_dir = images_dir
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = f"{self.images_dir}/{img_info['file_name']}"

        image = imread_unicode(path)

        if image is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks = [], [], []

        for ann in anns:
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue

            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(mask)

        if len(boxes) == 0:
            raise ValueError(f"Пустая разметка: {path}")

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(masks, dtype=torch.uint8),
        }

        return image, target
