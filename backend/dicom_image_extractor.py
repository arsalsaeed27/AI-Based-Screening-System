import sys

import numpy as np
import pydicom
from PIL import Image


def extract_image(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        arr -= arr.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val * 255.0
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L").convert("RGB")
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        img = Image.fromarray(arr, mode="RGB")
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        img = Image.fromarray(arr, mode="RGBA").convert("RGB")
    else:
        img = Image.fromarray(arr.squeeze()).convert("RGB")

    img.save(output_path, format="JPEG", quality=90)


if __name__ == '__main__':
    dicom_path = sys.argv[1]
    output_path = sys.argv[2]
    extract_image(dicom_path, output_path)
