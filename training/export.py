import argparse
import os
import shutil

import torch

from training.model import RetinalCNN

SMOKE_TEST_PATH = "/content/drive/MyDrive/retinal-models/smoke_test.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Export RetinalCNN to ONNX")
    parser.add_argument("--weights", required=True, help="Path to the .pth weights file")
    parser.add_argument("--output", required=True, help="Path to save the .onnx file")
    return parser.parse_args()


def main():
    args = parse_args()

    model = RetinalCNN()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )

    os.makedirs(os.path.dirname(SMOKE_TEST_PATH), exist_ok=True)
    shutil.copy(args.output, SMOKE_TEST_PATH)

    print("Export complete")


if __name__ == "__main__":
    main()
