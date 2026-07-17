import argparse
import os

import onnx
import segmentation_models_pytorch as smp
import torch


def main():
    parser = argparse.ArgumentParser(description="Export DeepLabV3+ glaucoma model to ONNX")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation="sigmoid",
    )
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 640, 640)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    model_onnx = onnx.load(args.output)
    onnx.save_model(model_onnx, args.output, save_as_external_data=False)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    print("Export complete")


if __name__ == "__main__":
    main()
