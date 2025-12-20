import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class RandomDataset(Dataset):
    def __init__(self, num_samples=1000, num_features=10, num_classes=3):
        self.X = np.random.randn(num_samples, num_features).astype(np.float32)
        self.y = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int64)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = RandomDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class SimpleNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

os.makedirs("output", exist_ok=True)
pth_path = "model.pth"
torch.save(model.state_dict(), pth_path)
print(f"Модель сохранена в {pth_path}")
