import argparse

import onnx
import segmentation_models_pytorch as smp
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export SMP glaucoma UNet to ONNX")
    parser.add_argument("--weights", required=True, help="Path to the .pth weights file")
    parser.add_argument("--output", required=True, help="Path to save the .onnx file")
    return parser.parse_args()


def main():
    args = parse_args()

    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=2,
        activation="sigmoid",
    )
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

    model_onnx = onnx.load(args.output)
    onnx.save_model(model_onnx, args.output, save_as_external_data=False)

    print("Export complete")


if __name__ == "__main__":
    main()
