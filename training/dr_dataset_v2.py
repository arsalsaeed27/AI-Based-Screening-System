import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.dataset import apply_clahe, crop_to_circle

CLASS_FOLDERS = ["0", "1", "2", "3", "4"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


class FolderDRDataset(Dataset):
    def __init__(self, root_path, image_size=224, augment=False):
        self.image_size = image_size
        self.augment = augment
        self.samples = []

        for label_folder in CLASS_FOLDERS:
            folder_path = os.path.join(root_path, label_folder)
            if not os.path.isdir(folder_path):
                continue

            label = int(label_folder)
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(folder_path, filename), label))

    def __len__(self):
        return len(self.samples)

    def _augment(self, image):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        if random.random() < 0.5:
            image = cv2.flip(image, 0)

        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        brightness = random.uniform(-30, 30)
        contrast = random.uniform(0.8, 1.2)
        image = image.astype(np.float32) * contrast + brightness
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = crop_to_circle(image)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = apply_clahe(image)

        if self.augment:
            image = self._augment(image)

        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return tensor, label


def get_folder_dataloaders(train_path, val_path, batch_size=32):
    train_dataset = FolderDRDataset(train_path, augment=True)
    val_dataset = FolderDRDataset(val_path, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
