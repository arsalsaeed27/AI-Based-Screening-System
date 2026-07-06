import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from training.glaucoma_dataset import get_dataloaders
from training.unet import UNet

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/glaucoma-models")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for optic disc/cup segmentation")
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--best_val_loss", type=float, default=float("inf"), help="best val loss from previous run")
    return parser.parse_args()


def dice_loss(preds, targets, smooth=1.0):
    preds = preds.contiguous().view(preds.size(0), preds.size(1), -1)
    targets = targets.contiguous().view(targets.size(0), targets.size(1), -1)
    intersection = (preds * targets).sum(dim=2)
    union = preds.sum(dim=2) + targets.sum(dim=2)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def dice_score(preds, targets, smooth=1.0):
    preds = (preds > 0.5).float().contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def run_epoch(model, loader, bce_criterion, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    total = 0
    disc_dice_sum = 0.0
    cup_dice_sum = 0.0
    num_batches = 0

    torch.set_grad_enabled(train)
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = bce_criterion(outputs, masks) + dice_loss(outputs, masks)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        if not train:
            disc_dice_sum += dice_score(outputs[:, 0], masks[:, 0])
            cup_dice_sum += dice_score(outputs[:, 1], masks[:, 1])
            num_batches += 1

        if train and batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    torch.set_grad_enabled(True)

    avg_loss = total_loss / total
    if train:
        return avg_loss, None, None
    return avg_loss, disc_dice_sum / num_batches, cup_dice_sum / num_batches


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        args.images, args.masks, batch_size=args.batch
    )

    model = UNet(in_channels=3, out_channels=2).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    bce_criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = args.best_val_loss
    print(f"Starting with best val loss: {best_val_loss:.4f}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, bce_criterion, optimizer, device, train=True
        )
        val_loss, disc_dice, cup_dice = run_epoch(
            model, val_loader, bce_criterion, optimizer, device, train=False
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice Disc: {disc_dice:.4f} | Val Dice Cup: {cup_dice:.4f}"
        )

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_glaucoma_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val loss {best_val_loss:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()