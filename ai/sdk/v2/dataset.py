import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from sdk.contracts import SegmentationDatasetAdapter


class CocoSegmentationDataset(SegmentationDatasetAdapter):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.coco = COCO(annotation_file)
        self.images_dir = images_dir
        self.transforms = transforms

        self.image_ids = [
            img_id
            for img_id in self.coco.imgs
            if any(self.coco.annToMask(ann).sum() > 0 for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)))
        ]

        if len(self.image_ids) == 0:
            raise ValueError("Нет изображений с масками!")

        category_ids = sorted([c["id"] for c in self.coco.cats.values()])
        self.category_id_map = {cid: i + 1 for i, cid in enumerate(category_ids)}
        self.num_classes = len(self.category_id_map) + 1

        disease_present = set()
        severity_values = set()

        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                attrs = ann.get("attributes", {})
                if attrs:
                    disease_present.add(1)
                else:
                    disease_present.add(0)

                for k in attrs:
                    if "балл" in k:
                        severity_values.add(int(k.split()[0]))

        self.num_disease_classes = len(disease_present)
        self.num_severity_classes = max(severity_values) + 1 if severity_values else 1

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

        boxes, labels, masks = [], [], []
        disease_labels, severity_labels = [], []

        for ann in anns:
            mask = self.coco.annToMask(ann)
            if mask.sum() == 0:
                continue

            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])

            cat = ann["category_id"]
            labels.append(self.category_id_map[ann[cat]])
            disease_labels.append(self.category_id_map.get(cat, 0))

            severity = 0
            for k in ann.get("attributes", {}):
                if "балл" in k:
                    severity = int(k.split()[0])
            severity_labels.append(severity)

            masks.append(mask)

        if len(boxes) == 0:
            raise ValueError(f"Пустая разметка: {path}")

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks), dtype=torch.float32),
            "disease": torch.tensor(disease_labels, dtype=torch.int64),
            "severity": torch.tensor(severity_labels, dtype=torch.int64),
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
