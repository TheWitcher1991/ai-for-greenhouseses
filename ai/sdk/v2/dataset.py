import json
import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from sdk.contracts import DetectionTarget, SegmentationDatasetAdapter


class CocoSegmentationDataset(SegmentationDatasetAdapter):
    def __init__(self, images_dir, annotation_file, transforms=None):
        with open(annotation_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        self.coco = COCO.__new__(COCO)
        self.coco.dataset = coco_data
        self.coco.createIndex()

        self.images_dir = images_dir
        self.transforms = transforms

        self.image_ids = [
            img_id
            for img_id in self.coco.imgs
            if any(self.coco.annToMask(ann).sum() > 0 for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)))
        ]

        if len(self.image_ids) == 0:
            raise ValueError("Нет изображений с масками!")

        categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_id_map = {c["id"]: i + 1 for i, c in enumerate(categories)}
        self.class_names = {i + 1: c["name"] for i, c in enumerate(categories)}
        self.num_classes = len(self.category_id_map) + 1

        disease_values = set()

        for ann in self.coco.loadAnns(self.coco.getAnnIds()):
            attrs = ann.get("attributes", {})
            disease_values.add(1 if "disease" in attrs else 0)

        self.max_severity = 0
        for ann in self.coco.loadAnns(self.coco.getAnnIds()):
            sev = ann.get("attributes", {}).get("severity", 0)
            self.max_severity = max(self.max_severity, int(sev))

        self.num_severity_classes = self.max_severity + 1
        self.num_disease_classes = len(disease_values)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.images_dir, img_info["file_name"])

        image = self.imread_unicode(path)
        if image is None:
            raise ValueError(f"Не удалось прочитать изображение: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, severity, disease = [], [], [], [], []

        for ann in anns:
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue

            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            labels.append(self.category_id_map[ann["category_id"]])

            attrs = ann.get("attributes", {})
            
            disease.append(1 if "Disease" in attrs else 0)
            severity.append(int(attrs.get("severity", 0)))

            masks.append(mask)

        if len(boxes) == 0:
            raise ValueError(f"Пустая разметка: {path}")

        target: DetectionTarget = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks), dtype=torch.float32),
            "severity": torch.tensor(severity, dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return image, target

    @staticmethod
    def imread_unicode(path: str):
        try:
            with open(path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
