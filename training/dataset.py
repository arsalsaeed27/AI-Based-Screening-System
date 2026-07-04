import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def crop_to_circle(image):
    """Crop the black border by finding the brightest circular (fundus) region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w == 0 or h == 0:
        return image

    return image[y:y + h, x:x + w]


def apply_clahe(image):
    """Apply CLAHE contrast enhancement on the luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class AptosDataset(Dataset):
    def __init__(self, csv_path, images_path=None, image_size=224, augment=False):
        self.df = pd.read_csv(csv_path)
        self.images_path = images_path
        self.image_size = image_size
        self.augment = augment
        self.has_image_path = "image_path" in self.df.columns

        if not self.has_image_path and images_path is None:
            raise ValueError(
                "images_path is required when the CSV has no 'image_path' column"
            )

    def __len__(self):
        return len(self.df)

    def _augment(self, image):
        # Horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        # Random rotation up to 15 degrees
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Brightness and contrast jitter
        brightness = random.uniform(-30, 30)
        contrast = random.uniform(0.8, 1.2)
        image = image.astype(np.float32) * contrast + brightness
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["diagnosis"])

        if self.has_image_path:
            image_path = row["image_path"]
        else:
            image_path = os.path.join(self.images_path, row["id_code"] + ".png")

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


def get_dataloaders(csv_path, images_path=None, batch_size=16):
    full_train = AptosDataset(csv_path, images_path, augment=True)
    full_val = AptosDataset(csv_path, images_path, augment=False)

    n = len(full_train)
    n_train = int(0.8 * n)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(n), [n_train, n_val], generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_train, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_val, val_indices.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
