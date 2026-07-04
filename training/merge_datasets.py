import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge APTOS 2019 and EyePACS datasets into one unified CSV"
    )
    parser.add_argument("--aptos_csv", required=True)
    parser.add_argument("--aptos_images", required=True)
    parser.add_argument("--eyepacs_csv", required=True)
    parser.add_argument("--eyepacs_images", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_aptos(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"id_code": "id_code", "diagnosis": "diagnosis"})
    df["source"] = "aptos"
    df["image_path"] = df["id_code"].apply(
        lambda x: os.path.abspath(os.path.join(images_dir, f"{x}.png"))
    )
    return df[["id_code", "diagnosis", "source", "image_path"]]


def load_eyepacs(csv_path, images_dir):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"image": "id_code", "level": "diagnosis"})
    df["source"] = "eyepacs"
    df["image_path"] = df["id_code"].apply(
        lambda x: os.path.abspath(os.path.join(images_dir, f"{x}.png"))
    )
    return df[["id_code", "diagnosis", "source", "image_path"]]


def filter_existing(df):
    exists_mask = df["image_path"].apply(os.path.isfile)
    skipped = int((~exists_mask).sum())
    return df[exists_mask].reset_index(drop=True), skipped


def print_distribution(name, df):
    print(f"\nLabel distribution for {name}:")
    print(df["diagnosis"].value_counts().sort_index())


def main():
    args = parse_args()

    aptos_df = load_aptos(args.aptos_csv, args.aptos_images)
    eyepacs_df = load_eyepacs(args.eyepacs_csv, args.eyepacs_images)

    aptos_df, aptos_skipped = filter_existing(aptos_df)
    eyepacs_df, eyepacs_skipped = filter_existing(eyepacs_df)

    print(f"Skipped {aptos_skipped} missing images from APTOS")
    print(f"Skipped {eyepacs_skipped} missing images from EyePACS")

    print_distribution("APTOS", aptos_df)
    print_distribution("EyePACS", eyepacs_df)

    merged_df = pd.concat([aptos_df, eyepacs_df], ignore_index=True)
    print_distribution("Combined", merged_df)

    merged_df.to_csv(args.output, index=False)
    print(f"\nMerged dataset saved to {args.output}")
    print(f"Total records: {len(merged_df)}")


if __name__ == "__main__":
    main()
