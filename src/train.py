import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_datasets
from model import build_model

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
MODEL_NAME = "cats-dogs-resnet18"

def train():
    mlflow.set_experiment("cats-vs-dogs")

    train_ds, val_ds, _ = get_datasets("data/processed")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "architecture": "ResNet18",
            "transfer_learning": True
        })

        for epoch in range(EPOCHS):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    preds = model(x).argmax(1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            acc = correct / total
            mlflow.log_metric("val_accuracy", acc, step=epoch)
            print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")

        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME
        )

if __name__ == "__main__":
    train()
