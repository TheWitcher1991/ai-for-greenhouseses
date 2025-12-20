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

        # Список изображений с масками
        self.image_ids = []
        for img_id in self.coco.imgs:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            valid = any(self.coco.annToMask(ann).sum() > 0 for ann in anns)
            if valid:
                self.image_ids.append(img_id)

        if len(self.image_ids) == 0:
            raise ValueError("Нет изображений с аннотациями и ненулевыми масками!")

        category_ids = sorted([c["id"] for c in self.coco.cats.values()])
        self.category_id_map = {cid: i + 1 for i, cid in enumerate(category_ids)}
        self.num_classes = len(self.category_id_map) + 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = f"{self.images_dir}/{img_info['file_name']}"

        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")
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
            labels.append(self.category_id_map[ann["category_id"]])
            masks.append(mask)

        if len(boxes) == 0:
            raise ValueError(f"Пустая разметка: {path}")

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8),
        }

        return image, target
