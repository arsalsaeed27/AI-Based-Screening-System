import argparse
import os
import time

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import ConcatDataset, DataLoader

from training.glaucoma_dataset import GlaucomaSegDataset

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/glaucoma-deeplab-models")
IMAGE_SIZE = 640

bce = nn.BCELoss()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3+ (ResNet50) for optic disc/cup segmentation"
    )
    parser.add_argument("--refuge_train_images", type=str, required=True)
    parser.add_argument("--refuge_train_masks", type=str, required=True)
    parser.add_argument("--refuge_val_images", type=str, required=True)
    parser.add_argument("--refuge_val_masks", type=str, required=True)
    parser.add_argument("--refuge_test_images", type=str, required=True)
    parser.add_argument("--refuge_test_masks", type=str, required=True)
    parser.add_argument("--origa_images", type=str, required=True)
    parser.add_argument("--origa_masks", type=str, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--best_val_loss", type=float, default=float("inf"))
    return parser.parse_args()


def dice_loss(preds, targets, smooth=1.0):
    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()


def dice_score(preds, targets, smooth=1.0):
    preds = (preds > 0.5).float()
    preds = preds.contiguous().view(preds.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    return ((2 * intersection + smooth) / (union + smooth)).mean().item()


def combined_loss(outputs, masks):
    disc_loss = bce(outputs[:, 0], masks[:, 0]) + dice_loss(outputs[:, 0], masks[:, 0])
    cup_loss = bce(outputs[:, 1], masks[:, 1]) + dice_loss(outputs[:, 1], masks[:, 1])
    return disc_loss + 2.0 * cup_loss


def build_optimizer(model):
    param_groups = [
        {"params": model.encoder.parameters(), "lr": 0.00005, "weight_decay": 1e-4},
        {
            "params": list(model.decoder.parameters()) + list(model.segmentation_head.parameters()),
            "lr": 0.0005,
            "weight_decay": 1e-4,
        },
    ]
    return AdamW(param_groups)


def run_train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}", flush=True)

    return total_loss / total


def run_val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    disc_dice_sum = 0.0
    cup_dice_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = combined_loss(outputs, masks)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            disc_dice_sum += dice_score(outputs[:, 0], masks[:, 0])
            cup_dice_sum += dice_score(outputs[:, 1], masks[:, 1])
            num_batches += 1

    return total_loss / total, disc_dice_sum / num_batches, cup_dice_sum / num_batches


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    refuge_train = GlaucomaSegDataset(
        args.refuge_train_images, args.refuge_train_masks, image_size=IMAGE_SIZE, augment=True
    )
    refuge_val = GlaucomaSegDataset(
        args.refuge_val_images, args.refuge_val_masks, image_size=IMAGE_SIZE, augment=True
    )
    refuge_test = GlaucomaSegDataset(
        args.refuge_test_images, args.refuge_test_masks, image_size=IMAGE_SIZE, augment=True
    )
    origa_val = GlaucomaSegDataset(
        args.origa_images, args.origa_masks, image_size=IMAGE_SIZE, augment=False
    )

    print(f"REFUGE train: {len(refuge_train)} images")
    print(f"REFUGE val: {len(refuge_val)} images")
    print(f"REFUGE test: {len(refuge_test)} images")
    print(f"ORIGA val: {len(origa_val)} images")

    train_dataset = ConcatDataset([refuge_train, refuge_val, refuge_test])
    val_dataset = origa_val

    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
    )

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation="sigmoid",
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from: {args.resume}")

    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-7)

    best_val_loss = args.best_val_loss
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device)
        val_loss, disc_dice, cup_dice = run_val_epoch(model, val_loader, device)

        scheduler.step(epoch)

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Dice Disc: {disc_dice:.4f} | Dice Cup: {cup_dice:.4f}"
        )

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_glaucoma_deeplab_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print("New best saved")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
