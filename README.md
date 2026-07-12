# AI-Based Eye Screening

AI-assisted screening for diabetic retinopathy, glaucoma, and hypertensive
retinopathy from fundus photographs. The app is a Node/Express API that runs
inference, plus a static HTML/JS clinical UI. `server.js` automatically
launches the Python/Flask Grad-CAM service (`gradcam_service.py`) as a child
process on startup and shuts it down when the server stops — you don't need
to run it separately.

## Prerequisites

- Node.js (18+) and npm
- Python 3.9+ and pip, available as `python` on your PATH (this is what
  `server.js` shells out to when it launches the Grad-CAM service)
- The trained model files (see below — **not included in git**)

## 1. Get the model files

The `models/` folder is git-ignored (model weights are large and change
often), so it will be empty after cloning. Ask whoever shared this repo for
the model files and place them directly in `models/`:

```
models/
├── smoke_test.onnx           (+ smoke_test.onnx.data)      — DR model
├── glaucoma_model.onnx       (+ glaucoma_model.onnx.data)  — glaucoma model
├── hr_efficientnet_model.onnx                              — hypertensive retinopathy model
└── best_efficientnet_model.pth                             — DR model weights, used for Grad-CAM heatmaps
```

The server won't start without the three `.onnx` files. The Grad-CAM service
won't start without `best_efficientnet_model.pth`.

## 2. Install dependencies

**Backend (Node):**

```bash
cd backend
npm install
```

**Grad-CAM service (Python) — from the repo root:**

```bash
pip install -r requirements.txt
```

Both need to be installed even though you only run one command to start
everything (see below) — Node runs the API, and it spawns `python` to run
the Grad-CAM service alongside it.

## 3. Run the app

One command, from `backend/`:

```bash
cd backend
node server.js
```

This starts the Grad-CAM service on port 5000 (as a background child
process) and the main API + web UI on port 3000. If the Grad-CAM service
fails to start or crashes, `/predict` still works but returns
`heatmap: null`. Stopping `server.js` (Ctrl+C) also stops the Grad-CAM
process.

## 4. Open the app

Go to [http://127.0.0.1:3000](http://127.0.0.1:3000) in your browser.

If your terminal reports a different port for `server.js`, use that port
instead (default is 3000).

## Project layout

- `backend/` — Express API (`server.js`, which also launches the Flask
  Grad-CAM service `gradcam_service.py` as a subprocess) and the static UI
  (`public/index.html`)
- `training/` — dataset classes, model definitions, training scripts, and
  ONNX export scripts for all three models
- `models/` — model weights (git-ignored, provided separately)
- `data/` — datasets used for training (git-ignored)
