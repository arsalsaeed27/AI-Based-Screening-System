import argparse
import os
import random

import pandas as pd

CLASS_FOLDERS = ["0", "1", "2", "3", "4"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a class-balanced DR dataset from Sehastrajits, APTOS, and EyePACS"
    )
    parser.add_argument("--sehastrajits_train", required=True, help="path to sehastrajits train folder")
    parser.add_argument("--aptos_csv", required=True, help="path to APTOS csv")
    parser.add_argument("--aptos_images", required=True, help="path to APTOS images folder")
    parser.add_argument("--eyepacs_csv", required=True, help="path to EyePACS csv")
    parser.add_argument("--eyepacs_images", required=True, help="path to EyePACS images folder")
    parser.add_argument("--output", required=True, help="path to save merged CSV")
    parser.add_argument("--samples_per_class", type=int, default=8116, help="target samples per class")
    return parser.parse_args()


def load_sehastrajits(train_path):
    samples = []
    for label_folder in CLASS_FOLDERS:
        folder_path = os.path.join(train_path, label_folder)
        if not os.path.isdir(folder_path):
            continue

        label = int(label_folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                samples.append((os.path.join(folder_path, filename), label))

    return samples


def load_aptos(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, f"{row['id_code']}.png")
        samples.append((image_path, int(row["diagnosis"])))

    return samples


def load_eyepacs(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_dir, f"{row['image']}.png")
        samples.append((image_path, int(row["level"])))

    return samples


def filter_existing(samples):
    return [(path, label) for path, label in samples if os.path.isfile(path)]


def group_by_label(samples):
    grouped = {label: [] for label in range(5)}
    for path, label in samples:
        if label in grouped:
            grouped[label].append(path)

    return grouped


def sample_class(paths, samples_per_class):
    if len(paths) >= samples_per_class:
        return random.sample(paths, samples_per_class)

    print(
        f"  Warning: only {len(paths)} images available, "
        f"sampling with replacement to reach {samples_per_class}"
    )
    return [random.choice(paths) for _ in range(samples_per_class)]


def main():
    args = parse_args()
    random.seed(42)

    all_samples = []
    all_samples.extend(load_sehastrajits(args.sehastrajits_train))
    all_samples.extend(load_aptos(args.aptos_csv, args.aptos_images))
    all_samples.extend(load_eyepacs(args.eyepacs_csv, args.eyepacs_images))

    total_before = len(all_samples)
    all_samples = filter_existing(all_samples)
    print(f"Filtered {total_before - len(all_samples)} missing images out of {total_before}")

    grouped = group_by_label(all_samples)

    balanced_rows = []
    for label in range(5):
        paths = grouped[label]
        print(f"Class {label}: {len(paths)} images available")
        sampled = sample_class(paths, args.samples_per_class)
        for path in sampled:
            balanced_rows.append({"image_path": path, "label": label})

    merged_df = pd.DataFrame(balanced_rows)
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nFinal class distribution:")
    print(merged_df["label"].value_counts().sort_index())
    print(f"\nTotal samples: {len(merged_df)}")

    merged_df.to_csv(args.output, index=False)
    print(f"Saved merged dataset to {args.output}")


if __name__ == "__main__":
    main()
