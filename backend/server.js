const path = require("path");
const express = require("express");
const multer = require("multer");
const sharp = require("sharp");
const ort = require("onnxruntime-node");

const PORT = 3000;
const MODEL_PATH = path.join(__dirname, "..", "models", "smoke_test.onnx");

const CLASS_LABELS = {
  0: "No DR",
  1: "Mild DR",
  2: "Moderate DR",
  3: "Severe DR",
  4: "Proliferative DR",
};

const REFERRAL_GUIDANCE = {
  0: "Re-screen in 12 months",
  1: "Re-screen in 6 months",
  2: "Refer to ophthalmologist within 6 months",
  3: "Urgent referral within 2-4 weeks",
  4: "Emergency referral - high risk of vision loss",
};

const upload = multer({ storage: multer.memoryStorage() });
const app = express();

let session;

async function preprocessImage(buffer) {
  const { data } = await sharp(buffer)
    .resize(224, 224)
    .removeAlpha()
    .toColorspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32Data = new Float32Array(3 * 224 * 224);
  const pixelCount = 224 * 224;

  for (let i = 0; i < pixelCount; i++) {
    float32Data[i] = data[i * 3] / 255;
    float32Data[pixelCount + i] = data[i * 3 + 1] / 255;
    float32Data[2 * pixelCount + i] = data[i * 3 + 2] / 255;
  }

  return new ort.Tensor("float32", float32Data, [1, 3, 224, 224]);
}

function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded" });
  }

  try {
    const inputTensor = await preprocessImage(req.file.buffer);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    const scores = softmax(Array.from(outputTensor.data));
    const predictedClass = scores.indexOf(Math.max(...scores));

    res.json({
      predicted_class: predictedClass,
      severity_label: CLASS_LABELS[predictedClass],
      confidence: Number(scores[predictedClass].toFixed(2)),
      scores: scores.map((s) => Number(s.toFixed(2))),
      referral: REFERRAL_GUIDANCE[predictedClass],
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Inference failed" });
  }
});

async function start() {
  session = await ort.InferenceSession.create(MODEL_PATH);
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
  });
}

start();
