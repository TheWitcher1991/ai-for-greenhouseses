from re import L

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sdk.dataset import CocoSegmentationDataset
from sdk.logger import logger
from sdk.maskrcnn import MaskRCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLModel:
    def __init__(
        self,
        dataset: CocoSegmentationDataset,
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

        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

        logger.info(
            f"Датасет загружен: {len(dataset)} изображений, "
            f"classes={dataset.num_classes}, attr_classes={dataset.num_attr_classes}"
        )

        self.model = MaskRCNN(num_classes=dataset.num_classes, num_attr_classes=dataset.num_attr_classes)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scaler = torch.amp.GradScaler(device=self.device)

    def train(self) -> None:
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

    def save(self) -> None:
        torch.save(self.model.state_dict(), "model.pth")
        logger.info("Модель сохранена: model.pth")
