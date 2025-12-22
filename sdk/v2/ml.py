from re import L

import cv2
import torch
from torch.utils.data import DataLoader

from sdk.logger import logger
from sdk.v1.ml import MLM as MLMv1

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
                    "confidence_percent": float(score.item() * 100),
                    "mask": output["masks"][i, 0].cpu().numpy(),
                }
            )

        return results
