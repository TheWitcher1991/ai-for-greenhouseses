import os

import cv2
import numpy as np
import torch
from sdk.contracts import SegmentationDatasetAdapter


class YoloDetectionDataset(SegmentationDatasetAdapter):
    def __init__(
        self,
        dataset_root: str,
        transforms=None,
    ):
        self.root = dataset_root
        self.transforms = transforms

        self.images = self._load_images()
        self.class_names = self._load_class_names()

        self.num_classes = len(self.class_names) + 1
        self.num_attr_classes = 1

    def _load_images(self):
        train_file = os.path.join(self.root, "train.txt")
        with open(train_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_class_names(self):
        path = os.path.join(self.root, "obj.names")
        with open(path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        return {i + 1: name for i, name in enumerate(names)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = img_path.replace(".jpg", ".txt")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить {img_path}")

        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels, masks, attr_labels = [], [], [], []

        with open(label_path, "r") as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

                mask = np.zeros((h, w), dtype=np.uint8)
                mask[int(y1) : int(y2), int(x1) : int(x2)] = 1
                masks.append(mask)

                attr_labels.append(0)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks), dtype=torch.float32),
            "attr_labels": torch.tensor(attr_labels, dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)

        return image, target, self.class_names
