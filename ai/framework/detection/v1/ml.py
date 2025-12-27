from typing import Optional

import cv2
import torch
from framework.contracts import (
    BackboneConfig,
    DetectionPrediction,
    DetectionPredictions,
    ImageTensor,
    ModelConfig,
    SegmentationDatasetAdapter,
    TrainerAdapter,
)
from framework.logger import logger
from framework.registry.metrics import MetricsRegistry
from framework.storage.json import json_storage
from torch.utils.data import DataLoader
from tqdm import tqdm

from .maskrcnn import MaskRCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLM(TrainerAdapter):
    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        dataset: SegmentationDatasetAdapter = None,
        device: str = DEVICE,
        epochs: int = 10,
        batch_size: int = 2,
        lr: int = 1e-4,
        weights_path: str = None,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weights_path = weights_path
        self.backbone_cfg = backbone_cfg

        self.metrics = MetricsRegistry()

        logger.info(f"Устройство: {self.device}")
        logger.info(f"Эпохи={epochs}, batch_size={batch_size}, lr={lr}")

        self.model: Optional[MaskRCNN] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.use_amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        if dataset is not None:
            self._init_from_dataset(dataset)

    def _init_from_dataset(self, dataset: SegmentationDatasetAdapter) -> None:
        self.loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
        )

        self.category_id_to_name_en = dataset.category_id_to_name_en
        self.object_labels = {i: self.category_id_to_name_en[cid] for cid, i in dataset.category_id_map.items()}
        self.object_attrs = {i: str(i) for i in range(dataset.num_attr_classes)}

        logger.info(
            f"Dataset: {len(dataset)} images | "
            f"classes={dataset.num_classes} | "
            f"attr_classes={dataset.num_attr_classes}"
        )

        self.model = MaskRCNN(
            num_classes=dataset.num_classes,
            num_attr_classes=dataset.num_attr_classes,
            backbone_cfg=self.backbone_cfg,
            weights_path=self.weights_path,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def train(self):
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Trainer is not initialized with dataset/model")

        logger.info("Начало обучения")

        for epoch in range(self.epochs):
            logger.info(f"Эпоха {epoch + 1}/{self.epochs} старт")
            self.model.train()
            epoch_loss = 0

            self.metrics.reset()

            for step, (images, targets, _) in enumerate(tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.epochs}")):
                logger.info(f"Батч {step + 1}/{len(self.loader)} получен")
                logger.info(f"Количество изображений в батче: {len(images)}")

                try:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    logger.info(f"Пример target[0]: { {k: v.shape for k, v in targets[0].items()} }")
                except Exception as e:
                    logger.error(f"Ошибка переноса данных на device: {e}")
                    continue

                self.optimizer.zero_grad(set_to_none=True)

                try:
                    with torch.amp.autocast(enabled=self.use_amp):
                        losses = self.model(images, targets)
                        loss = sum(losses.values())

                    logger.info("Losses: " + ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
                    logger.info(f"Текущий суммарный loss батча: {loss.item():.4f}")
                except Exception as e:
                    logger.error(f"Ошибка forward/backward: {e}")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

                self.model.eval()
                with torch.no_grad():
                    preds = self.model(images)
                    self.metrics.update(preds, targets)
                self.model.train()

                logger.info(f"Батч {step + 1} завершен, накопленный loss эпохи: {epoch_loss:.4f}")

            epoch_metrics = self.metrics.compute()

            logger.info(f"Эпоха {epoch + 1} завершена, loss={epoch_loss:.4f}, метрики={epoch_metrics}")

        logger.info("Обучение завершено")

    def save(self, model_path="model.pth", labels_path="labels.json", metrics_path="metrics.json"):
        torch.save(self.model.state_dict(), model_path)

        labels_dict = ModelConfig(
            architecture=self.backbone_cfg.name,
            num_classes=len(self.object_labels),
            num_attr_classes=len(self.object_attrs),
            object_labels=self.object_labels,
            object_attrs=self.object_attrs,
            weights_storage="local",
            weights_path=self.weights_path,
        )

        json_storage.save(labels_path, labels_dict)

        json_storage.save(metrics_path, self.metrics.all())

        logger.info(f"Модель сохранена: {model_path}, labels: {labels_path}")

    def load(self, model_path="model.pth", labels_path="labels.json"):
        labels_dict: ModelConfig = json_storage.load(labels_path)

        self.object_labels = labels_dict["object_labels"]
        self.object_attrs = labels_dict["object_attrs"]

        self.model = MaskRCNN(
            num_classes=len(self.object_labels),
            num_attr_classes=len(self.object_attrs),
            backbone_cfg=self.backbone_cfg,
        )

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Модель загружена: {model_path}")

    def predict(self, image_path, score_threshold=0.5):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor: ImageTensor = (
            torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)[0]

        results: DetectionPredictions = []
        for score, label, mask in zip(output["scores"], output["labels"], output["masks"]):
            if score.item() < score_threshold:
                continue

            label_id = int(label.item())
            label_name = self.object_labels.get(label_id, str(label_id))

            results.append(
                DetectionPrediction(
                    label_id=label_id,
                    label=label_name,
                    score=float(score.item()),
                    confidence_percent=float(score.item() * 100),
                    mask=mask[0].cpu(),
                )
            )

        return results
