from re import L

import torch
from torch.utils.data import DataLoader

from sdk.logger import logger
from sdk.ml import MLM as MLMv1

from .dataset import CocoSegmentationDataset
from .maskrcnn import MaskRCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLM(MLMv1):
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
            f"classes={dataset.num_classes}, num_disease_classes={dataset.num_disease_classes}, num_severity_classes={dataset.num_severity_classes}"
        )

        self.model = MaskRCNN(
            num_classes=dataset.num_classes,
            num_disease=dataset.num_disease_classes,
            num_severity=dataset.num_severity_classes,
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scaler = torch.amp.GradScaler(device=self.device)
