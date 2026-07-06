import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from training.dataset import apply_clahe

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


class GlaucomaSegDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size=512, augment=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.augment = augment

        self.filenames = sorted(
            f for f in os.listdir(images_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
            and os.path.exists(os.path.join(masks_path, os.path.splitext(f)[0] + ".png"))
        )

    def __len__(self):
        return len(self.filenames)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        angle = random.uniform(-30, 30)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(
            mask, matrix, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return image, mask

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        stem = os.path.splitext(filename)[0]

        image_path = os.path.join(self.images_path, filename)
        mask_path = os.path.join(self.masks_path, stem + ".png")

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = apply_clahe(image)

        if self.augment:
            image, mask = self._augment(image, mask)

        disc = (mask >= 1).astype(np.float32)
        cup = (mask == 2).astype(np.float32)
        mask_tensor = torch.from_numpy(np.stack([disc, cup], axis=0)).float()

        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return image_tensor, mask_tensor


def get_dataloaders(images_path, masks_path, batch_size=4):
    full_train = GlaucomaSegDataset(images_path, masks_path, augment=True)
    full_val = GlaucomaSegDataset(images_path, masks_path, augment=False)

    n = len(full_train)
    n_train = int(0.8 * n)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(n), [n_train, n_val], generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_train, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_val, val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
