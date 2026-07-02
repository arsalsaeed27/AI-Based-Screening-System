import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from training.dataset import get_dataloaders
from training.model import RetinalCNN

CHECKPOINT_DIR = "/content/drive/MyDrive/retinal-models"


def parse_args():
    parser = argparse.ArgumentParser(description="Train RetinalCNN on the APTOS dataset")
    parser.add_argument("--csv", type=str, required=True, help="path to train.csv")
    parser.add_argument("--images", type=str, required=True, help="path to images folder")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    return parser.parse_args()


def compute_class_weights(csv_path, num_classes, device):
    df = pd.read_csv(csv_path)
    counts = df["diagnosis"].value_counts().reindex(range(num_classes), fill_value=0).values
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    torch.set_grad_enabled(True)

    return total_loss / total, correct / total


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders(args.csv, args.images, batch_size=args.batch)

    num_classes = 5
    model = RetinalCNN(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(args.csv, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val loss {best_val_loss:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
