import argparse
import os
import time

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from training.glaucoma_dataset import GlaucomaSegDataset

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/kaggle/working/glaucoma-smp-models")
IMAGE_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an SMP UNet (EfficientNet-B4 encoder) for optic disc/cup segmentation"
    )
    parser.add_argument("--train_images", type=str, required=True, help="path to REFUGE train Images")
    parser.add_argument("--train_masks", type=str, required=True, help="path to REFUGE train Masks")
    parser.add_argument("--val_images", type=str, required=True, help="path to REFUGE val Images")
    parser.add_argument("--val_masks", type=str, required=True, help="path to REFUGE val Masks")
    parser.add_argument("--batch", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="epoch to start from")
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


def combined_loss(bce_criterion, outputs, masks):
    disc_out, cup_out = outputs[:, 0], outputs[:, 1]
    disc_mask, cup_mask = masks[:, 0], masks[:, 1]

    disc_loss = bce_criterion(disc_out, disc_mask) + dice_loss(disc_out, disc_mask)
    cup_loss = bce_criterion(cup_out, cup_mask) + dice_loss(cup_out, cup_mask)

    return disc_loss + 2.0 * cup_loss


def build_optimizer(model):
    param_groups = [
        {"params": model.encoder.parameters(), "lr": 0.0001, "weight_decay": 1e-4},
        {
            "params": [
                p for n, p in model.named_parameters() if not n.startswith("encoder.")
            ],
            "lr": 0.001,
            "weight_decay": 1e-4,
        },
    ]
    return AdamW(param_groups)


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
        loss = combined_loss(bce_criterion, outputs, masks)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        if not train:
            disc_dice_sum += dice_score(outputs[:, 0], masks[:, 0])
            cup_dice_sum += dice_score(outputs[:, 1], masks[:, 1])
            num_batches += 1

        if train and batch_idx % 10 == 0:
            print(
                f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}",
                flush=True,
            )

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

    train_dataset = GlaucomaSegDataset(
        args.train_images, args.train_masks, image_size=IMAGE_SIZE, augment=True
    )
    val_dataset = GlaucomaSegDataset(
        args.val_images, args.val_masks, image_size=IMAGE_SIZE, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
    )

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation="sigmoid",
    ).to(device)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from checkpoint: {args.resume}")

    bce_criterion = nn.BCELoss()
    optimizer = build_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

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

        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Dice Disc: {disc_dice:.4f} | Dice Cup: {cup_dice:.4f}"
        )

        if epoch % 10 == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "best_glaucoma_smp_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val loss {best_val_loss:.4f}, saved: {best_model_path}")

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed / 60:.2f} minutes ({elapsed:.2f} seconds)")


if __name__ == "__main__":
    main()
