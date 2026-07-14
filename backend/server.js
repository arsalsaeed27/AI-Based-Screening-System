const path = require("path");
const express = require("express");
const mongoose = require("mongoose");
const multer = require("multer");
const sharp = require("sharp");
const ort = require("onnxruntime-node");
const FormData = require("form-data");
const fetch = require("node-fetch");
const { spawn } = require("child_process");

const PORT = 3000;
const MODEL_PATH = path.join(__dirname, "..", "models", "smoke_test.onnx");
const GLAUCOMA_MODEL_PATH = path.join(
  __dirname,
  "..",
  "models",
  "glaucoma_model.onnx",
);
const HR_MODEL_PATH = path.join(
  __dirname,
  "..",
  "models",
  "hr_efficientnet_model.onnx",
);
const GRADCAM_SERVICE_URL = "http://localhost:5000/gradcam";
const MONGODB_URI =
  process.env.MONGODB_URI ||
  "mongodb+srv://ai_retinal_screening:D8jaYBNFn0kURWcg@cluster0.hzqnb4s.mongodb.net/?appName=Cluster0";

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

const scanSchema = new mongoose.Schema(
  {
    // Scan metadata
    scanId: { type: String, required: true, unique: true },
    timestamp: { type: Date, default: Date.now },

    // Patient demographics
    patientId: String,
    patientName: String,
    patientAge: Number,
    patientSex: String,
    patientDob: String,
    patientEye: String, // OD / OS / OU

    // Clinical context
    diabeticStatus: String, // Type 1 / Type 2 / No
    hba1c: Number, // HbA1c percentage
    referringClinician: String,
    institution: String,

    // Image quality
    imageQuality: {
      passed: Boolean,
      blurIndex: Number,
    },

    // Conditions screened
    conditionsScreened: [String], // DR, Glaucoma, HR

    // DR result
    drResult: {
      performed: Boolean,
      grade: Number, // 0-4
      severityLabel: String, // No DR / Mild / Moderate / Severe / Proliferative
      confidence: Number,
      scores: [Number], // all 5 class probabilities
      referral: String,
      lowConfidence: Boolean,
      gradDescription: String, // clinical description of the grade
      icdrGrade: String, // ICDR grade text
    },

    // Glaucoma result
    glaucomaResult: {
      performed: Boolean,
      cdr: Number, // Cup-to-Disc Ratio
      riskLevel: String, // Normal / Monitor / Suspicious / High risk
      riskDetail: String,
      discPixels: Number,
      cupPixels: Number,
      glaucomaSuspected: Boolean,
    },

    // HR result
    hrResult: {
      performed: Boolean,
      detected: Boolean,
      probability: Number,
      riskLevel: String,
      recommendation: String,
    },

    // Overall triage
    triage: {
      level: String, // ROUTINE / MONITORING / NON-URGENT / URGENT / EMERGENCY
      mainMessage: String,
      description: String,
      referralRequired: Boolean,
      referralUrgency: String, // Within 12 months / 6 months / 4 weeks / immediate
    },

    // Follow up
    followUpDate: Date,
    notes: String,
    status: { type: String, default: "pending" }, // pending / reviewed / referred
    reviewedBy: String,
  },
  { timestamps: true },
);

const Scan = mongoose.model("Scan", scanSchema);

const upload = multer({ storage: multer.memoryStorage() });
const app = express();
app.use(express.static(path.join(__dirname, "public")));
app.use(express.json());

let session;
let glaucomaSession;
let hrSession;
let gradcamProcess;

async function preprocessImageDR(buffer) {
  const { data } = await sharp(buffer)
    .resize(300, 300)
    .removeAlpha()
    .toColorspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32Data = new Float32Array(3 * 300 * 300);
  const pixelCount = 300 * 300;

  for (let i = 0; i < pixelCount; i++) {
    float32Data[i] = data[i * 3] / 255;
    float32Data[pixelCount + i] = data[i * 3 + 1] / 255;
    float32Data[2 * pixelCount + i] = data[i * 3 + 2] / 255;
  }

  return new ort.Tensor("float32", float32Data, [1, 3, 300, 300]);
}

async function preprocessImageHREfficientNet(buffer) {
  const { data } = await sharp(buffer)
    .resize(300, 300)
    .removeAlpha()
    .toColorspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });

  const float32Data = new Float32Array(3 * 300 * 300);
  const pixelCount = 300 * 300;

  for (let i = 0; i < pixelCount; i++) {
    float32Data[i] = data[i * 3] / 255;
    float32Data[pixelCount + i] = data[i * 3 + 1] / 255;
    float32Data[2 * pixelCount + i] = data[i * 3 + 2] / 255;
  }

  return new ort.Tensor("float32", float32Data, [1, 3, 300, 300]);
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

async function preprocessImageGlaucoma(buffer) {
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

function regionBrightness(data, imageSize, startX, startY, regionSize) {
  let sum = 0;
  let count = 0;

  for (let y = startY; y < startY + regionSize; y++) {
    for (let x = startX; x < startX + regionSize; x++) {
      const idx = (y * imageSize + x) * 3;
      sum += (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
      count++;
    }
  }

  return sum / count;
}

function regionVariance(data, imageSize, startX, startY, regionSize) {
  const brightnessValues = [];

  for (let y = startY; y < startY + regionSize; y++) {
    for (let x = startX; x < startX + regionSize; x++) {
      const idx = (y * imageSize + x) * 3;
      brightnessValues.push((data[idx] + data[idx + 1] + data[idx + 2]) / 3);
    }
  }

  const mean =
    brightnessValues.reduce((a, b) => a + b, 0) / brightnessValues.length;
  const variance =
    brightnessValues.reduce((sum, v) => sum + (v - mean) ** 2, 0) /
    brightnessValues.length;

  return variance;
}

function regionAverageColor(data, imageSize, startX, startY, regionSize) {
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  let count = 0;

  for (let y = startY; y < startY + regionSize; y++) {
    for (let x = startX; x < startX + regionSize; x++) {
      const idx = (y * imageSize + x) * 3;
      sumR += data[idx];
      sumG += data[idx + 1];
      sumB += data[idx + 2];
      count++;
    }
  }

  return { r: sumR / count, g: sumG / count, b: sumB / count };
}

async function validateFundusImage(imageBuffer) {
  const IMAGE_SIZE = 224;

  const { data } = await sharp(imageBuffer)
    .resize(IMAGE_SIZE, IMAGE_SIZE)
    .removeAlpha()
    .toColorspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });

  const CORNER_SIZE = 10;
  const corners = [
    regionBrightness(data, IMAGE_SIZE, 0, 0, CORNER_SIZE),
    regionBrightness(
      data,
      IMAGE_SIZE,
      IMAGE_SIZE - CORNER_SIZE,
      0,
      CORNER_SIZE,
    ),
    regionBrightness(
      data,
      IMAGE_SIZE,
      0,
      IMAGE_SIZE - CORNER_SIZE,
      CORNER_SIZE,
    ),
    regionBrightness(
      data,
      IMAGE_SIZE,
      IMAGE_SIZE - CORNER_SIZE,
      IMAGE_SIZE - CORNER_SIZE,
      CORNER_SIZE,
    ),
  ];
  const cornerBrightness = corners.reduce((a, b) => a + b, 0) / corners.length;

  if (cornerBrightness > 60) {
    return {
      valid: false,
      error:
        "This does not appear to be a fundus photograph. Fundus images have a dark circular border.",
    };
  }

  const CENTER_SIZE = 80;
  const centerStart = (IMAGE_SIZE - CENTER_SIZE) / 2;
  const centerBrightness = regionBrightness(
    data,
    IMAGE_SIZE,
    centerStart,
    centerStart,
    CENTER_SIZE,
  );

  if (centerBrightness < 40) {
    return {
      valid: false,
      error: "Image is too dark. Please upload a clear fundus photograph.",
    };
  }

  if (centerBrightness - cornerBrightness < 30) {
    return {
      valid: false,
      error:
        "This does not appear to be a fundus photograph. Please upload a retinal fundus image.",
    };
  }

  const { r: rAvg, b: bAvg } = regionAverageColor(
    data,
    IMAGE_SIZE,
    centerStart,
    centerStart,
    CENTER_SIZE,
  );

  if (rAvg < 80) {
    return {
      valid: false,
      error:
        "Image does not appear to be a retinal fundus photograph. Fundus images have a characteristic red/orange tone.",
    };
  }

  if (rAvg - bAvg < 15) {
    return {
      valid: false,
      error:
        "Image does not appear to be a retinal fundus photograph. Fundus images have a characteristic red/orange tone.",
    };
  }

  const sharpnessVariance = regionVariance(
    data,
    IMAGE_SIZE,
    centerStart,
    centerStart,
    CENTER_SIZE,
  );

  if (sharpnessVariance < 50) {
    return {
      valid: false,
      error:
        "Image appears blurry. Please upload a sharper fundus photograph for accurate results.",
    };
  }

  return { valid: true };
}

async function fetchGradCam(imageBuffer, filename) {
  try {
    const form = new FormData();
    form.append("image", imageBuffer, filename || "image.png");

    const res = await fetch(GRADCAM_SERVICE_URL, {
      method: "POST",
      body: form,
      headers: form.getHeaders(),
    });

    if (!res.ok) return null;

    const data = await res.json();
    return data.heatmap || null;
  } catch (err) {
    console.error("Grad-CAM service unavailable:", err.message);
    return null;
  }
}

function getRiskLevel(cdr) {
  if (cdr < 0.3) {
    return { risk_level: "Normal", risk_detail: "CDR within normal range" };
  }
  if (cdr < 0.5) {
    return {
      risk_level: "Monitor",
      risk_detail: "CDR slightly elevated, monitor over time",
    };
  }
  if (cdr < 0.7) {
    return {
      risk_level: "Suspicious",
      risk_detail: "Suspicious — refer for IOP testing",
    };
  }
  return {
    risk_level: "High risk",
    risk_detail: "High risk — urgent referral",
  };
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
    const inputTensor = await preprocessImageDR(req.file.buffer);
    const feeds = { [session.inputNames[0]]: inputTensor };
    const results = await session.run(feeds);
    const outputTensor = results[session.outputNames[0]];

    const scores = softmax(Array.from(outputTensor.data));
    const predictedClass = scores.indexOf(Math.max(...scores));

    const topScore = scores[predictedClass];
    const secondScore = Math.max(
      ...scores.filter((_, i) => i !== predictedClass),
    );
    const isLowConfidence = topScore < 0.5 || topScore - secondScore < 0.15;

    const heatmap = await fetchGradCam(req.file.buffer, req.file.originalname);

    res.json({
      predicted_class: predictedClass,
      severity_label: CLASS_LABELS[predictedClass],
      confidence: Number(scores[predictedClass].toFixed(2)),
      scores: scores.map((s) => Number(s.toFixed(2))),
      referral: REFERRAL_GUIDANCE[predictedClass],
      heatmap,
      low_confidence: isLowConfidence,
      confidence_warning: isLowConfidence
        ? "Low confidence prediction — consider re-imaging or specialist review"
        : null,
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
    const inputTensor = await preprocessImageGlaucoma(req.file.buffer);
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
    const inputTensor = await preprocessImageHREfficientNet(req.file.buffer);
    const feeds = { [hrSession.inputNames[0]]: inputTensor };
    const results = await hrSession.run(feeds);
    const outputTensor = results[hrSession.outputNames[0]];

    const logit = outputTensor.data[0];
    const probability = sigmoid(logit);
    const hrDetected = probability > 0.2;

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

app.post("/save-scan", async (req, res) => {
  try {
    const scan = new Scan(req.body);
    await scan.save();
    res.json({ success: true, scanId: scan.scanId });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to save scan" });
  }
});

app.get("/scans", async (req, res) => {
  try {
    const scans = await Scan.find({}, { heatmap: 0 })
      .sort({ timestamp: -1 })
      .limit(50);
    res.json(scans);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch scans" });
  }
});

app.get("/scans/:scanId", async (req, res) => {
  try {
    const scan = await Scan.findOne({ scanId: req.params.scanId });
    if (!scan) {
      return res.status(404).json({ error: "Scan not found" });
    }
    res.json(scan);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch scan" });
  }
});

app.patch("/scans/:scanId", async (req, res) => {
  try {
    const { status, reviewedBy, notes, followUpDate } = req.body;
    const update = {};
    if (status !== undefined) update.status = status;
    if (reviewedBy !== undefined) update.reviewedBy = reviewedBy;
    if (notes !== undefined) update.notes = notes;
    if (followUpDate !== undefined) update.followUpDate = followUpDate;

    const scan = await Scan.findOneAndUpdate(
      { scanId: req.params.scanId },
      update,
      { new: true },
    );
    if (!scan) {
      return res.status(404).json({ error: "Scan not found" });
    }
    res.json(scan);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to update scan" });
  }
});

function startGradCam() {
  const gradcam = spawn("python", [path.join(__dirname, "gradcam_service.py")], {
    detached: false,
    stdio: ["ignore", "pipe", "pipe"],
  });

  gradcam.stdout.on("data", (data) => {
    console.log("[GradCAM]", data.toString().trim());
  });

  gradcam.stderr.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) console.log("[GradCAM]", msg);
  });

  gradcam.on("exit", (code) => {
    console.log("[GradCAM] Process exited with code", code);
  });

  console.log("[GradCAM] Starting service...");
  return gradcam;
}

async function start() {
  try {
    await mongoose.connect(MONGODB_URI);
    console.log("MongoDB connected");
  } catch (err) {
    console.error("MongoDB connection failed, continuing without it:", err.message);
    console.error("Scan history will not be saved until MongoDB is reachable.");
  }

  gradcamProcess = startGradCam();

  await new Promise((resolve) => setTimeout(resolve, 3000));

  session = await ort.InferenceSession.create(MODEL_PATH);
  glaucomaSession = await ort.InferenceSession.create(GLAUCOMA_MODEL_PATH);
  hrSession = await ort.InferenceSession.create(HR_MODEL_PATH);
  app.listen(PORT, () => {
    console.log(`Server listening on port ${PORT}`);
  });
}

process.on("exit", () => {
  if (gradcamProcess) gradcamProcess.kill();
});
process.on("SIGINT", () => {
  process.exit();
});

start();
