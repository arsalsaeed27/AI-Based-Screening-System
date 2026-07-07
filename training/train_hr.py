import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from training.hr_dataset import build_merged_dataframe, get_dataloaders
from training.model import RetinalCNN

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/hr-models")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RetinalCNN for hypertensive retinopathy binary classification"
    )
    parser.add_argument("--zoya_hr", type=str, required=True, help="path to Zoya77 HR images folder")
    parser.add_argument("--zoya_normal", type=str, required=True, help="path to Zoya77 Normal images folder")
    parser.add_argument("--hrdc_hr_csv", type=str, required=True, help="path to HRDC HR Classification CSV")
    parser.add_argument("--hrdc_hr_images", type=str, required=True, help="path to HRDC HR Classification images")
    parser.add_argument("--hrdc_hyp_csv", type=str, required=True, help="path to HRDC Hypertensive Classification CSV")
    parser.add_argument("--hrdc_hyp_images", type=str, required=True, help="path to HRDC Hypertensive Classification images")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="epoch to start from")
    parser.add_argument("--best_val_loss", type=float, default=float("inf"), help="best val loss from previous run")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    tp = tn = fp = fn = 0

    torch.set_grad_enabled(train)
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).float()

        if train:
            optimizer.zero_grad()

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (outputs > 0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if not train:
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

        # print every 10 batches so you know it's alive
        if train and batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    torch.set_grad_enabled(True)

    avg_loss = total_loss / total
    acc = correct / total

    if train:
        return avg_loss, acc, None, None

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return avg_loss, acc, sensitivity, specificity


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df = build_merged_dataframe(
        args.zoya_hr, args.zoya_normal,
        args.hrdc_hr_csv, args.hrdc_hr_images,
        args.hrdc_hyp_csv, args.hrdc_hyp_images,
    )

    train_loader, val_loader, class_weights = get_dataloaders(df, batch_size=args.batch)

    model = RetinalCNN(num_classes=1).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    # use a mild pos_weight instead of the full inverse ratio
    train_loader, val_loader = get_dataloaders(df, batch_size=args.batch)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = args.best_val_loss
    print(f"Starting with best val loss: {best_val_loss:.4f}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc, val_sensitivity, val_specificity = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Sensitivity: {val_sensitivity:.4f} | Val Specificity: {val_specificity:.4f}"
        )

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_hr_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val loss {best_val_loss:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
