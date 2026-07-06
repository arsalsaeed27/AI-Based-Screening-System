import argparse

import torch

from training.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Export UNet to ONNX")
    parser.add_argument("--weights", required=True, help="Path to the .pth weights file")
    parser.add_argument("--output", required=True, help="Path to save the .onnx file")
    return parser.parse_args()


def main():
    args = parse_args()

    model = UNet(in_channels=3, out_channels=2)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )

    print("Export complete")


if __name__ == "__main__":
    main()
