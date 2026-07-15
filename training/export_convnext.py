import argparse
import os

import onnx
import timm
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export ConvNeXt-Base DR model to ONNX")
    parser.add_argument("--weights", required=True, help="Path to the .pth weights file")
    parser.add_argument("--output", required=True, help="Path to save the .onnx file")
    return parser.parse_args()


def main():
    args = parse_args()

    model = timm.create_model("convnext_base", pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
    )

    model_onnx = onnx.load(args.output)
    onnx.save_model(model_onnx, args.output, save_as_external_data=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    print("Export complete")


if __name__ == "__main__":
    main()
