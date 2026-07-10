import argparse
import os
import time

import timm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from training.dr_dataset_v2 import get_csv_dataloaders

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/efficientnet-models")
IMAGE_SIZE = 300
NUM_CLASSES = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B3 on a folder-structured DR dataset"
    )
    parser.add_argument("--train", type=str, required=True, help="path to train CSV")
    parser.add_argument("--val", type=str, required=True, help="path to val CSV")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="epoch to start from")
    parser.add_argument("--best_val_loss", type=float, default=float("inf"), help="best val loss so far")
    return parser.parse_args()


def build_optimizer(model):
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" not in n],
            "lr": 0.00003,
            "weight_decay": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": 0.0003,
            "weight_decay": 1e-4,
        },
    ]
    return AdamW(param_groups)


def run_train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 20 == 0:
            print(
                f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}",
                flush=True,
            )

    return total_loss / total, correct / total


def run_val_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_csv_dataloaders(
        args.train, args.val, batch_size=args.batch, image_size=300
    )

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=True,
        num_classes=NUM_CLASSES,
        drop_rate=0.3,
        drop_path_rate=0.2,
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[0.0003, 0.003],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )
    scaler = GradScaler()

    best_val_loss = args.best_val_loss
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_acc = run_train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_loss, val_acc = run_val_epoch(model, val_loader, criterion, device)

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
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_efficientnet_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val loss {best_val_loss:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
