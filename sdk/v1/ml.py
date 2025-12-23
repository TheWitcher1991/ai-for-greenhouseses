import json

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sdk.contracts import TrainerAdapter
from sdk.logger import logger

from .dataset import CocoSegmentationDataset
from .maskrcnn import MaskRCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLM(TrainerAdapter):
    def __init__(
        self,
        dataset: CocoSegmentationDataset = None,
        device: str = DEVICE,
        epochs: int = 10,
        batch_size: int = 2,
        lr: int = 1e-4,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        logger.info(f"Устройство: {self.device}")
        logger.info(f"Эпохи={epochs}, batch_size={batch_size}, lr={lr}")

        if dataset:
            self.loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
            )

            self.object_labels = {i: name for i, name in dataset.category_id_map.items()}
            self.object_attrs = {i: str(i) for i in range(dataset.num_attr_classes)}

            logger.info(
                f"Датасет загружен: {len(dataset)} изображений, "
                f"classes={dataset.num_classes}, attr_classes={dataset.num_attr_classes}"
            )

            self.model = MaskRCNN(
                num_classes=dataset.num_classes or 1, num_attr_classes=dataset.num_attr_classes or 1
            ).to(self.device)

            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            self.scaler = torch.amp.GradScaler(device=self.device)

    def train(self):
        logger.info("Начало обучения")

        for epoch in range(self.epochs):
            logger.info(f"Эпоха {epoch + 1}/{self.epochs} старт")
            self.model.train()
            epoch_loss = 0

            for step, (images, targets) in enumerate(tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.epochs}")):
                logger.info(f"Батч {step + 1} получен")

                try:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                except Exception as e:
                    logger.error(f"Ошибка переноса данных на device: {e}")
                    continue

                self.optimizer.zero_grad()
                try:
                    with torch.amp.autocast(device_type=self.device):
                        losses = self.model(images, targets)
                        loss = sum(losses.values())

                    logger.info("Losses: " + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
                except Exception as e:
                    logger.error(f"Ошибка forward/backward: {e}")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            logger.info(f"Эпоха {epoch + 1} завершена, loss={epoch_loss:.4f}")

        logger.info("Обучение завершено")

    def save(self, model_path="model.pth", labels_path="labels.json"):
        torch.save(self.model.state_dict(), model_path)

        labels_dict = {
            "object_labels": self.object_labels,
            "object_attrs": self.object_attrs,
        }

        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Модель сохранена: {model_path}, labels: {labels_path}")

    def load(self, model_path="model.pth", labels_path="labels.json"):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)

        self.object_labels = labels_dict["object_labels"]
        self.object_attrs = labels_dict["object_attrs"]

        self.model = MaskRCNN(
            num_classes=len(self.object_labels),
            num_attr_classes=len(self.object_attrs),
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Модель загружена: {model_path}")

    def predict(self, image_path, score_threshold=0.5):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = self.model(tensor)[0]

        results = []
        for score, label, mask in zip(output["scores"], output["labels"], output["masks"]):
            if score < score_threshold:
                continue

            results.append(
                {
                    "label": int(label),
                    "score": float(score),
                    "confidence_percent": float(score.item() * 100),
                    "mask": mask[0].cpu().numpy(),
                }
            )

        return results
