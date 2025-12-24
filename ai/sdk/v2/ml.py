import json
from re import L

import cv2
import torch
from sdk.logger import logger
from sdk.v1.ml import MLM as MLMv1
from torch.utils.data import DataLoader

from .dataset import CocoSegmentationDataset
from .maskrcnn import MaskRCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MLM(MLMv1):
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
            self.disease_labels = {0: "healthy", 1: "powdery"}
            self.severity_labels = {i: str(i) for i in range(dataset.num_attr_classes)}

            logger.info(
                f"Датасет загружен: {len(dataset)} изображений, "
                f"classes={dataset.num_classes}, num_disease_classes={dataset.num_disease_classes}, num_severity_classes={dataset.num_severity_classes}"
            )

            self.model = MaskRCNN(
                num_classes=dataset.num_classes,
                num_disease=dataset.num_disease_classes,
                num_severity=dataset.num_severity_classes,
            ).to(self.device)

            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            self.scaler = torch.amp.GradScaler(device=self.device)

    def save(self, model_path="model.pth", labels_path="labels.json"):
        torch.save(self.model.state_dict(), model_path)

        labels_dict = {
            "object_labels": self.object_labels,
            "disease_labels": self.disease_labels,
            "severity_labels": self.severity_labels,
        }

        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Модель сохранена: {model_path}, labels: {labels_path}")

    def load(self, model_path="model.pth", labels_path="labels.json"):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        self.object_labels = labels_dict["object_labels"]
        self.disease_labels = labels_dict["disease_labels"]
        self.severity_labels = labels_dict["severity_labels"]

        self.model = MaskRCNN(
            num_classes=len(self.object_labels),
            num_disease=len(self.disease_labels),
            num_severity=len(self.severity_labels),
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Модель загружена: {model_path}")

    def predict(self, image_path, score_threshold=0.5):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)[0]

        results = []
        for idx, (score, label, mask) in enumerate(zip(output["scores"], output["labels"], output["masks"])):
            score = float(output["scores"][i])
            if score < score_threshold:
                continue

            # obj_label = self.object_labels.get(int(output["labels"][i]), "Unknown")
            # disease = self.disease_labels.get(int(output["disease_pred"][i]), "Unknown")
            # severity = int(output["severity_pred"][i])

            disease_idx = output["disease_pred"][idx].item()
            severity_idx = output["severity_pred"][idx].item()

            results.append(
                {
                    "label": self.object_labels.get(label.item(), str(label.item())),
                    "disease": self.disease_labels.get(disease_idx, str(disease_idx)),
                    "severity": self.severity_labels.get(severity_idx, str(severity_idx)),
                    "confidence_percent": float(score.item() * 100),
                    "mask": mask[0].cpu().numpy(),
                }
            )

        return results
