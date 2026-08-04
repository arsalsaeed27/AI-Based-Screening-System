const path = require("path");
const fs = require("fs");
const chokidar = require("chokidar");
const { execFile, spawnSync } = require("child_process");

const WATCH_FOLDER = process.env.WATCH_FOLDER || path.join(__dirname, "shared_folder");

const WATCHED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".pdf"];

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

    if (global.io) {
      global.io.emit("new_scan_from_folder", {
        filename,
        structured,
        lines,
        timestamp: new Date().toISOString(),
      });
    }
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
