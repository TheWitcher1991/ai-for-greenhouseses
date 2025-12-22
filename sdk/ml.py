import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sdk.dataset import CocoSegmentationDataset
from sdk.maskrcnn import MaskRCNNWithAttr

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

        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

        self.model = MaskRCNNWithAttr(num_classes=dataset.num_classes, num_attr_classes=dataset.num_attr_classes)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scaler = torch.amp.GradScaler(device=self.device)

    def instance(self) -> MaskRCNNWithAttr:
        return self.model

    def train(self) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for images, targets in tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                self.optimizer.zero_grad()
                try:
                    with torch.amp.autocast(device_type=self.device):
                        losses = self.model(images, targets)
                        loss = sum(losses.values())
                except Exception as e:
                    print(f"Ошибка при обработке батча: {e}")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

    def save(self) -> None:
        torch.save(self.model.state_dict(), "model.pth")
        print("Модель сохранена как model.pth")
