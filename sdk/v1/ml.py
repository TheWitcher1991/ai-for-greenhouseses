from re import L

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

            logger.info(
                f"Датасет загружен: {len(dataset)} изображений, "
                f"classes={dataset.num_classes}, attr_classes={dataset.num_attr_classes}"
            )

        self.model = MaskRCNN(num_classes=dataset.num_classes or 1, num_attr_classes=dataset.num_attr_classes or 1).to(
            self.device
        )

        if dataset:
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

    def save(self):
        torch.save(self.model.state_dict(), "model.pth")
        logger.info("Модель сохранена: model.pth")

    def load(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Модель загружена: {path}")

    def predict(self, image_path, score_threshold=0.5):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)[0]

        results = []
        for i in range(len(output["boxes"])):
            score = float(output["scores"][i])
            if score < score_threshold:
                continue

            obj_label = self.object_labels.get(int(output["labels"][i]), "Unknown")
            disease = self.disease_labels.get(int(output["disease_pred"][i]), "Unknown")
            severity = int(output["severity_pred"][i])

            results.append(
                {
                    "object": obj_label,
                    "disease": disease,
                    "severity": severity,
                    "score": score,
                    "box": output["boxes"][i].cpu().numpy(),
                    "mask": output["masks"][i, 0].cpu().numpy(),
                }
            )

        return results
