import argparse
import os
import random
import time

import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split

from training.hr_dataset import build_merged_dataframe, HRDataset

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/hr-efficientnet-models")
IMAGE_SIZE = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B3 for hypertensive retinopathy binary classification"
    )
    parser.add_argument("--zoya_hr", type=str, required=True, help="path to Zoya77 HR images folder")
    parser.add_argument("--zoya_normal", type=str, required=True, help="path to Zoya77 Normal images folder")
    parser.add_argument("--hrdc_hr_csv", type=str, required=True, help="path to HRDC HR Classification CSV")
    parser.add_argument("--hrdc_hr_images", type=str, required=True, help="path to HRDC HR Classification images")
    parser.add_argument("--hrdc_hyp_csv", type=str, required=True, help="path to HRDC Hypertensive Classification CSV")
    parser.add_argument("--hrdc_hyp_images", type=str, required=True, help="path to HRDC Hypertensive Classification images")
    parser.add_argument("--eyepacs_csv", type=str, default=None, help="path to EyePACS CSV (optional, adds healthy images)")
    parser.add_argument("--eyepacs_images", type=str, default=None, help="path to EyePACS images (optional, adds healthy images)")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="epoch to start from")
    parser.add_argument("--best_val_loss", type=float, default=float("inf"), help="best val loss from previous run")
    return parser.parse_args()


def build_balanced_dataframe(df):
    random.seed(42)
    pos_df = df[df["label"] == 1].reset_index(drop=True)
    neg_df = df[df["label"] == 0].reset_index(drop=True)

    neg_sample = neg_df.sample(n=min(len(pos_df), len(neg_df)), random_state=42)
    balanced_df = pd.concat([pos_df, neg_sample], ignore_index=True)

    print("Balanced class distribution:")
    print(balanced_df["label"].value_counts().sort_index())

    return balanced_df


def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, train):
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
            with autocast():
                outputs = model(images).squeeze(1)

            # cast to float32 for stable loss calculation
            outputs = outputs.float()
            loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            preds = (outputs > 0).float()
        else:
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            preds = (torch.sigmoid(outputs) > 0.3).float()

        total_loss += loss.item() * images.size(0)
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
        eyepacs_csv=args.eyepacs_csv, eyepacs_images=args.eyepacs_images,
    )
    balanced_df = build_balanced_dataframe(df)

    n = len(balanced_df)
    n_train = int(0.8 * n)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(n), [n_train, n_val], generator=generator)

    train_df = balanced_df.iloc[train_indices.indices].reset_index(drop=True)
    val_df = balanced_df.iloc[val_indices.indices].reset_index(drop=True)

    train_dataset = HRDataset(train_df, image_size=IMAGE_SIZE, augment=True)
    val_dataset = HRDataset(val_df, image_size=IMAGE_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = timm.create_model(
        'efficientnet_b3',
        pretrained=True,
        num_classes=1,
        drop_rate=0.3,
        drop_path_rate=0.2,
    ).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n],
         "lr": 0.00003, "weight_decay": 1e-4},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n],
         "lr": 0.0003, "weight_decay": 1e-4},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[0.0003, 0.003],
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    scaler = GradScaler()

    best_val_loss = args.best_val_loss
    print(f"Starting with best val loss: {best_val_loss:.4f}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device, train=True
        )
        val_loss, val_acc, val_sensitivity, val_specificity = run_epoch(
            model, val_loader, criterion, optimizer, scaler, scheduler, device, train=False
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Sensitivity: {val_sensitivity:.4f} | Val Specificity: {val_specificity:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_hr_efficientnet_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val loss {best_val_loss:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
