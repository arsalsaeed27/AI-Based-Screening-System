import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.dr_dataset_v2 import get_folder_dataloaders
from training.model import DeepRetinalCNN

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/dr-v2-models")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepRetinalCNN on a folder-structured DR dataset"
    )
    parser.add_argument("--train", type=str, required=True, help="path to train folder")
    parser.add_argument("--val", type=str, required=True, help="path to val folder")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="epoch to start from")
    parser.add_argument("--best_val_loss", type=float, default=float("inf"), help="best val loss so far")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    def process_batches():
        nonlocal total_loss, correct, total
        for batch_idx, (images, labels) in enumerate(loader):
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

            if train and batch_idx % 20 == 0:
                print(
                    f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}",
                    flush=True,
                )

    if train:
        process_batches()
    else:
        with torch.no_grad():
            process_batches()

    return total_loss / total, correct / total


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_folder_dataloaders(
        args.train, args.val, batch_size=args.batch
    )

    num_classes = 5
    model = DeepRetinalCNN(num_classes=num_classes).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_val_loss = args.best_val_loss
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

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
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_dr_v2_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val loss {best_val_loss:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
