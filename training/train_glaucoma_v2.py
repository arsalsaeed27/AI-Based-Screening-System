import argparse
import os
import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from training.glaucoma_dataset_v2 import get_glaucoma_v2_dataloaders
from training.unet import UNet

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/glaucoma-v2-models")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet v2 for optic disc/cup segmentation on the merged dataset")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--best_val_loss", type=float, default=float("inf"), help="best val loss from previous run")
    return parser.parse_args()


def dice_loss(preds, targets, smooth=1.0):
    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def dice_score(preds, targets, smooth=1.0):
    preds = (preds > 0.5).float().contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def focal_loss(preds, targets, gamma=2.0, alpha=0.8):
    import torch.nn.functional as F
    bce = F.binary_cross_entropy(preds, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def combined_loss(outputs, masks, bce, epoch):
    disc_bce = bce(outputs[:, 0], masks[:, 0])
    disc_dice = dice_loss(outputs[:, 0], masks[:, 0])
    cup_bce = bce(outputs[:, 1], masks[:, 1])
    cup_dice = dice_loss(outputs[:, 1], masks[:, 1])

    # focal loss for cup
    cup_focal = focal_loss(outputs[:, 1], masks[:, 1])

    disc_loss = disc_bce + disc_dice
    cup_loss = cup_bce + 2.0 * cup_dice + cup_focal
    return disc_loss + 3.0 * cup_loss


def run_epoch(model, loader, bce_criterion, optimizer, scaler, device, epoch, train):
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
            with autocast():
                outputs = model(images)
                loss = combined_loss(outputs, masks, bce_criterion, epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = combined_loss(outputs, masks, bce_criterion, epoch)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        if not train:
            disc_dice_sum += dice_score(outputs[:, 0], masks[:, 0])
            cup_dice_sum += dice_score(outputs[:, 1], masks[:, 1])
            num_batches += 1

        # print every 10 batches so you know it's alive
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

    # arnavjain1/glaucoma-datasets on Kaggle — aggregates ORIGA, REFUGE, and G1020,
    # each with Images/ and Masks/ subfolders. Verify these paths once the dataset
    # is attached, since exact mount path/casing can vary.
    base = "/kaggle/input/datasets/arnavjain1/glaucoma-datasets"
    folder_list = [
        {"images": f"{base}/REFUGE/train/Images", "masks": f"{base}/REFUGE/train/Masks"},
        {"images": f"{base}/REFUGE/val/Images",   "masks": f"{base}/REFUGE/val/Masks"},
        {"images": f"{base}/REFUGE/test/Images",  "masks": f"{base}/REFUGE/test/Masks"},
        {"images": f"{base}/ORIGA/Images",        "masks": f"{base}/ORIGA/Masks"},
        {"images": f"{base}/G1020/Images",        "masks": f"{base}/G1020/Masks"},
    ]
    train_loader, val_loader = get_glaucoma_v2_dataloaders(folder_list, batch_size=args.batch)

    model = UNet(in_channels=3, out_channels=2).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    bce_criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0003, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6
    )
    scaler = GradScaler()

    best_val_loss = args.best_val_loss
    best_disc_dice = 0.0
    best_cup_dice = 0.0
    best_combined_dice = 0.0
    print(f"Starting with best val loss: {best_val_loss:.4f}")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, bce_criterion, optimizer, scaler, device, epoch, train=True
        )
        val_loss, disc_dice, cup_dice = run_epoch(
            model, val_loader, bce_criterion, optimizer, scaler, device, epoch, train=False
        )
        avg_dice = (disc_dice + cup_dice) / 2
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Dice Disc: {disc_dice:.4f} | Dice Cup: {cup_dice:.4f} | "
            f"Dice Average: {avg_dice:.4f} | LR: {current_lr:.6f}"
        )

        scheduler.step(epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if disc_dice > best_disc_dice:
            best_disc_dice = disc_dice
            best_disc_path = os.path.join(CHECKPOINT_DIR, "best_disc_model.pth")
            torch.save(model.state_dict(), best_disc_path)
            print(f"New best disc dice {best_disc_dice:.4f}, saved: {best_disc_path}")

        if cup_dice > best_cup_dice:
            best_cup_dice = cup_dice
            best_cup_path = os.path.join(CHECKPOINT_DIR, "best_cup_model.pth")
            torch.save(model.state_dict(), best_cup_path)
            print(f"New best cup dice {best_cup_dice:.4f}, saved: {best_cup_path}")

        if avg_dice > best_combined_dice:
            best_combined_dice = avg_dice
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_glaucoma_v2_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best combined dice {best_combined_dice:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
