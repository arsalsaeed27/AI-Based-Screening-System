const path = require("path");
const express = require("express");
const multer = require("multer");
const sharp = require("sharp");
const ort = require("onnxruntime-node");

const PORT = 3000;
const MODEL_PATH = path.join(__dirname, "..", "models", "smoke_test.onnx");
const GLAUCOMA_MODEL_PATH = path.join(__dirname, "..", "models", "glaucoma_model.onnx");
const HR_MODEL_PATH = path.join(__dirname, "..", "models", "hr_model.onnx");

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
app.use(express.static(path.join(__dirname, "public")));

let session;
let glaucomaSession;
let hrSession;

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

function sigmoid(logit) {
  return 1 / (1 + Math.exp(-logit));
}

async function preprocessGlaucomaImage(buffer) {
  const { data } = await sharp(buffer)
    .resize(512, 512)
    .removeAlpha()
    .toColorspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32Data = new Float32Array(3 * 512 * 512);
  const pixelCount = 512 * 512;

  for (let i = 0; i < pixelCount; i++) {
    float32Data[i] = data[i * 3] / 255;
    float32Data[pixelCount + i] = data[i * 3 + 1] / 255;
    float32Data[2 * pixelCount + i] = data[i * 3 + 2] / 255;
  }

  return new ort.Tensor("float32", float32Data, [1, 3, 512, 512]);
}

async function validateFundusImage(imageBuffer) {
  const { data } = await sharp(imageBuffer)
    .resize(224, 224)
    .removeAlpha()
    .toColorspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });

  const totalValues = data.length;
  let sum = 0;
  for (let i = 0; i < totalValues; i++) sum += data[i];
  const mean = sum / totalValues;

  if (mean < 20) {
    return {
      valid: false,
      error:
        "Image appears to be blank or too dark. Please upload a clear fundus photograph.",
    };
  }

  let varianceSum = 0;
  for (let i = 0; i < totalValues; i++) {
    const diff = data[i] - mean;
    varianceSum += diff * diff;
  }
  const variance = varianceSum / totalValues;

  if (variance < 100) {
    return {
      valid: false,
      error: "Image quality too low. Please upload a clear fundus photograph.",
    };
  }

  const pixelCount = 224 * 224;
  const DARK_THRESHOLD = 20;
  const BRIGHT_THRESHOLD = 235;
  let extremeCount = 0;

  for (let i = 0; i < pixelCount; i++) {
    const intensity = (data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2]) / 3;
    if (intensity < DARK_THRESHOLD || intensity > BRIGHT_THRESHOLD) {
      extremeCount++;
    }
  }

  const extremeRatio = extremeCount / pixelCount;

  if (extremeRatio > 0.95) {
    return {
      valid: false,
      error:
        "This does not appear to be a fundus photograph. Please upload a retinal image.",
    };
  }

  return { valid: true };
}

function getRiskLevel(cdr) {
  if (cdr < 0.3) {
    return { risk_level: "Normal", risk_detail: "CDR within normal range" };
  }
  if (cdr < 0.5) {
    return { risk_level: "Monitor", risk_detail: "CDR slightly elevated, monitor over time" };
  }
  if (cdr < 0.7) {
    return { risk_level: "Suspicious", risk_detail: "Suspicious — refer for IOP testing" };
  }
  return { risk_level: "High risk", risk_detail: "High risk — urgent referral" };
}

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded" });
  }

  const validation = await validateFundusImage(req.file.buffer);
  if (!validation.valid) {
    return res.status(400).json({ error: validation.error });
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

app.post("/predict-glaucoma", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded" });
  }

  const validation = await validateFundusImage(req.file.buffer);
  if (!validation.valid) {
    return res.status(400).json({ error: validation.error });
  }

  try {
    const inputTensor = await preprocessGlaucomaImage(req.file.buffer);
    const feeds = { [glaucomaSession.inputNames[0]]: inputTensor };
    const results = await glaucomaSession.run(feeds);
    const outputTensor = results[glaucomaSession.outputNames[0]];

    const data = outputTensor.data;
    const pixelCount = 512 * 512;

    let discPixels = 0;
    let cupPixels = 0;

    for (let i = 0; i < pixelCount; i++) {
      if (data[i] >= 0.5) discPixels++;
      if (data[pixelCount + i] >= 0.5) cupPixels++;
    }

    const cdr = discPixels === 0 ? 0 : cupPixels / discPixels;
    const { risk_level, risk_detail } = getRiskLevel(cdr);

    res.json({
      cdr: Number(cdr.toFixed(2)),
      risk_level,
      risk_detail,
      disc_pixels: discPixels,
      cup_pixels: cupPixels,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Inference failed" });
  }
});

app.post("/predict-hr", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded" });
  }

  const validation = await validateFundusImage(req.file.buffer);
  if (!validation.valid) {
    return res.status(400).json({ error: validation.error });
  }

  try {
    const inputTensor = await preprocessImage(req.file.buffer);
    const feeds = { [hrSession.inputNames[0]]: inputTensor };
    const results = await hrSession.run(feeds);
    const outputTensor = results[hrSession.outputNames[0]];

    const logit = outputTensor.data[0];
    const probability = sigmoid(logit);
    const hrDetected = probability > 0.3;

    res.json({
      hr_detected: hrDetected,
      probability: Number(probability.toFixed(2)),
      risk_level: hrDetected ? "HR Detected" : "No HR Detected",
      recommendation: hrDetected
        ? "Refer to ophthalmologist — signs of hypertensive retinopathy detected"
        : "No signs of hypertensive retinopathy. Monitor blood pressure regularly.",
      note: "This is a preliminary screening result. Confirmation requires blood pressure measurement and specialist review.",
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Inference failed" });
  }
});

async function start() {
  session = await ort.InferenceSession.create(MODEL_PATH);
  glaucomaSession = await ort.InferenceSession.create(GLAUCOMA_MODEL_PATH);
  hrSession = await ort.InferenceSession.create(HR_MODEL_PATH);
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
  });
}

start();
