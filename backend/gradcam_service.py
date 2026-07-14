import base64
import os
import sys

import cv2
import numpy as np
import timm
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.dataset import apply_clahe, crop_to_circle

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "best_efficientnet_model.pth"
)
IMAGE_SIZE = 300
OVERLAY_OPACITY = 0.4

app = Flask(__name__)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

        score = output[0, predicted_class]
        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam).numpy()
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # only show top 70% of attention
        cam_threshold = cam.copy()
        cam_threshold[cam_threshold < 0.3] = 0
        cam = cam_threshold

        return cam, predicted_class


def preprocess_image(image_bgr):
    cropped = crop_to_circle(image_bgr)
    resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    enhanced = apply_clahe(rgb)

    normalized = enhanced.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()

    return tensor, resized


def apply_circular_mask(heatmap, image):
    h, w = heatmap.shape[:2]
    center_x, center_y = w // 2, h // 2

    # radius is 46% of the smaller dimension
    radius = int(min(h, w) * 0.46)

    # create distance map from center
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # create soft mask — 1.0 inside, gradual fade at edge
    # transition zone is 8% of radius width
    transition = radius * 0.08
    mask = np.clip((radius - dist_from_center) / transition, 0, 1).astype(np.float32)

    # apply soft mask to each channel of heatmap
    if len(heatmap.shape) == 3:
        mask = mask[:, :, np.newaxis]

    heatmap_masked = heatmap * mask
    return heatmap_masked.astype(heatmap.dtype)


def overlay_heatmap(cam, original_bgr):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        heatmap, OVERLAY_OPACITY, original_bgr, 1 - OVERLAY_OPACITY, 0
    )
    return overlay


def encode_image_base64(image_bgr):
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Failed to encode overlay image")
    return base64.b64encode(buffer).decode("utf-8")


model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

grad_cam = GradCAM(model, model.blocks[-1])


@app.route("/gradcam", methods=["POST"])
def gradcam():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file_bytes = np.frombuffer(request.files["image"].read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    input_tensor, resized_original = preprocess_image(image_bgr)
    cam, predicted_class = grad_cam.generate(input_tensor)
    cam = apply_circular_mask(cam, resized_original)
    overlay = overlay_heatmap(cam, resized_original)
    heatmap_b64 = encode_image_base64(overlay)

    return jsonify({"heatmap": heatmap_b64, "predicted_class": predicted_class})


if __name__ == "__main__":
    app.run(port=5000)
