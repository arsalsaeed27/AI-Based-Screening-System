import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from training.dataset import apply_clahe, crop_to_circle

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def build_glaucoma_pairs(folder_list):
    all_pairs = []

    for folder in folder_list:
        images_dir = folder["images"]
        masks_dir = folder["masks"]

        folder_pairs = []
        for filename in sorted(os.listdir(images_dir)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            stem = os.path.splitext(filename)[0]
            mask_path = os.path.join(masks_dir, stem + ".png")
            if os.path.isfile(mask_path):
                image_path = os.path.join(images_dir, filename)
                folder_pairs.append((image_path, mask_path))

        print(f"{images_dir}: {len(folder_pairs)} pairs found")
        all_pairs.extend(folder_pairs)

    print(f"Total pairs: {len(all_pairs)}")
    return all_pairs


class GlaucomaSegDatasetV2(Dataset):
    def __init__(self, pairs, image_size=512, augment=False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

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
        image_path, mask_path = self.pairs[idx]

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = crop_to_circle(image)

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


def get_glaucoma_v2_dataloaders(folder_list, batch_size=4, val_split=0.15):
    pairs = build_glaucoma_pairs(folder_list)

    n = len(pairs)
    n_val = int(val_split * n)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(n), [n_train, n_val], generator=generator)

    train_pairs = [pairs[i] for i in train_indices.indices]
    val_pairs = [pairs[i] for i in val_indices.indices]

    train_dataset = GlaucomaSegDatasetV2(train_pairs, image_size=512, augment=True)
    val_dataset = GlaucomaSegDatasetV2(val_pairs, image_size=512, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
