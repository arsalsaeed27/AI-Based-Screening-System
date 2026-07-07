import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from training.dataset import apply_clahe, crop_to_circle

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def _resolve_image_path(images_dir, filename):
    path = os.path.join(images_dir, filename)
    if os.path.isfile(path):
        return path

    stem = os.path.splitext(filename)[0]
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate

    return path


def _load_zoya(hr_path, normal_path):
    hr_df = pd.DataFrame({
        "image_path": [os.path.abspath(os.path.join(hr_path, f)) for f in _list_images(hr_path)],
        "label": 1,
    })
    normal_df = pd.DataFrame({
        "image_path": [os.path.abspath(os.path.join(normal_path, f)) for f in _list_images(normal_path)],
        "label": 0,
    })
    return pd.concat([hr_df, normal_df], ignore_index=True)


def _load_hrdc_csv(csv_path, images_dir, label_column):
    df = pd.read_csv(csv_path)
    df["image_path"] = df["Image"].apply(
        lambda x: os.path.abspath(_resolve_image_path(images_dir, str(x)))
    )
    df["label"] = df[label_column].astype(int)
    return df[["image_path", "label"]]


def _load_eyepacs_healthy(csv_path, images_dir, n):
    df = pd.read_csv(csv_path)
    df = df[df["level"] == 0]
    df = df.sample(n=min(n, len(df)), random_state=42)

    df["image_path"] = df["image"].apply(
        lambda x: os.path.abspath(os.path.join(images_dir, x + ".png"))
    )
    df = df[df["image_path"].apply(os.path.isfile)]
    df["label"] = 0

    return df[["image_path", "label"]]


def build_merged_dataframe(zoya_hr_path, zoya_normal_path, hrdc_hr_csv, hrdc_hr_images,
                            hrdc_hyp_csv, hrdc_hyp_images,
                            eyepacs_csv=None, eyepacs_images=None, eyepacs_n=600):
    zoya_df = _load_zoya(zoya_hr_path, zoya_normal_path)
    hrdc_hr_df = _load_hrdc_csv(hrdc_hr_csv, hrdc_hr_images, "Hypertensive Retinopathy")
    hrdc_hyp_df = _load_hrdc_csv(hrdc_hyp_csv, hrdc_hyp_images, "Hypertensive")

    dfs = [zoya_df, hrdc_hr_df, hrdc_hyp_df]

    if eyepacs_csv is not None and eyepacs_images is not None:
        dfs.append(_load_eyepacs_healthy(eyepacs_csv, eyepacs_images, eyepacs_n))

    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df["filename"] = merged_df["image_path"].apply(os.path.basename)
    merged_df = merged_df.drop_duplicates(subset="filename").drop(columns="filename")
    merged_df = merged_df.reset_index(drop=True)

    print("Class distribution:")
    print(merged_df["label"].value_counts().sort_index())

    return merged_df


class HRDataset(Dataset):
    def __init__(self, df, image_size=224, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _augment(self, image):
        # Horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)

        # Vertical flip
        if random.random() < 0.5:
            image = cv2.flip(image, 0)

        # Random rotation up to 30 degrees
        angle = random.uniform(-30, 30)
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
        label = int(row["label"])
        image = cv2.imread(row["image_path"])

        # skip corrupted images silently, return a black image
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            return tensor, label

        image = crop_to_circle(image)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = apply_clahe(image)

        if self.augment:
            image = self._augment(image)

        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return tensor, label


def get_dataloaders(df, batch_size=16):
    n = len(df)
    n_train = int(0.8 * n)
    n_val = n - n_train

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(n), [n_train, n_val], generator=generator)

    train_df = df.iloc[train_indices.indices].reset_index(drop=True)
    val_df = df.iloc[val_indices.indices].reset_index(drop=True)

    train_dataset = HRDataset(train_df, augment=True)
    val_dataset = HRDataset(val_df, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
