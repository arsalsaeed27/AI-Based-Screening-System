const path = require("path");
const fs = require("fs");
const chokidar = require("chokidar");
const { execFile, spawnSync } = require("child_process");
const mongoose = require("mongoose");

const WATCH_FOLDER = process.env.WATCH_FOLDER || path.join(__dirname, "shared_folder");

const WATCHED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".pdf"];

// server.js exports nothing, so redefine a minimal schema here just for the watcher.
const scanSchema = new mongoose.Schema({
  scanId:        { type: String, required: true, unique: true },
  timestamp:     { type: Date, default: Date.now },
  source:        { type: String, default: "shared_folder" },
  filename:      String,
  machineType:   String,
  patientName:   String,
  patientId:     String,
  patientAge:    String,
  eye:           String,
  date:          String,
  extractedData: Object,
  rawLines:      [String],
  status:        { type: String, default: "ocr_complete" },
}, { timestamps: true });

const FolderScan = mongoose.models.FolderScan ||
  mongoose.model("FolderScan", scanSchema);

function generateScanId() {
  return "OCR-" + Date.now().toString(36).toUpperCase();
}

function getPython() {
  return process.env.GRADCAM_PYTHON || "C:/gradcam-venv/Scripts/python.exe";
}

function handleNewFile(filePath) {
  const filename = path.basename(filePath);
  const ext = path.extname(filename).toLowerCase();

  if (!WATCHED_EXTENSIONS.includes(ext)) {
    return;
  }

  console.log(`[Watcher] New file detected: ${filename}`);

  const PYTHON = getPython();
  const OCR_SCRIPT = path.join(__dirname, "ocr_report.py");
  const PARSER_SCRIPT = path.join(__dirname, "ocr_parser.py");

  execFile(PYTHON, [OCR_SCRIPT, filePath], { timeout: 120000 }, (err, stdout, stderr) => {
    if (err) {
      console.error(`[Watcher] OCR failed for ${filename}:`, err.message);
      return;
    }

    let lines;
    try {
      lines = JSON.parse(stdout);
    } catch (e) {
      console.error(`[Watcher] Failed to parse OCR output for ${filename}:`, e.message);
      return;
    }

    const parserProc = spawnSync(PYTHON, [PARSER_SCRIPT], {
      input: JSON.stringify(lines),
      encoding: "utf8",
      timeout: 10000,
    });

    let structured = {};
    try {
      structured = JSON.parse(parserProc.stdout);
    } catch (e) {}

    console.log(`[Watcher] Extracted structured data for ${filename}:`, structured);

    const record = new FolderScan({
      scanId:        generateScanId(),
      filename:      filename,
      machineType:   structured.machine_type || "Unknown",
      patientName:   structured.patient_name || "Unknown",
      patientId:     structured.patient_id || "OCR-" + Date.now(),
      patientAge:    structured.patient_age || "",
      eye:           structured.eye || "",
      date:          structured.date || new Date().toISOString(),
      extractedData: structured,
      rawLines:      lines,
    });

    record.save()
      .then((saved) => {
        console.log(`[Watcher] Saved to MongoDB: ${saved.scanId}`);
        if (global.io) {
          global.io.emit("new_scan_from_folder", {
            filename,
            structured,
            scanId: saved.scanId,
            timestamp: saved.timestamp,
          });
        }
      })
      .catch((err) => {
        console.error("[Watcher] MongoDB save failed:", err.message);
      });
  });
}

function startWatcher() {
  if (!fs.existsSync(WATCH_FOLDER)) {
    fs.mkdirSync(WATCH_FOLDER, { recursive: true });
  }

  const watcher = chokidar.watch(WATCH_FOLDER, {
    ignoreInitial: true,
    awaitWriteFinish: {
      stabilityThreshold: 2000,
      pollInterval: 500,
    },
  });

  watcher.on("add", handleNewFile);

  console.log(`[Watcher] Watching folder: ${WATCH_FOLDER}`);

  return watcher;
}

module.exports = { startWatcher, WATCH_FOLDER };
