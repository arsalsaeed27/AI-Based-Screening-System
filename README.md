# AFIO ScanAI — AI-Powered Ophthalmic Screening Platform

> An end-to-end clinical AI system for retinal disease screening, built for the Armed Forces Institute of Ophthalmology (AFIO), Rawalpindi, Pakistan.

**Military College of Signals, NUST · Neudym AI Internship 2026**  
Built by Arsal Saeed · Mashal Areej

---

## What It Does

AFIO ScanAI screens for three major retinal diseases from a single fundus photograph in under 3 seconds:

| Disease                  | Model               | Accuracy                                              |
| ------------------------ | ------------------- | ----------------------------------------------------- |
| Diabetic Retinopathy     | EfficientNet-B3     | 83.86% val accuracy · AUC 0.9694 · Sensitivity 94.36% |
| Glaucoma                 | DeepLabV3+ ResNet50 | Disc Dice 96.58% · Cup Dice 93.69% · CDR Error 0.0196 |
| Hypertensive Retinopathy | EfficientNet-B3     | 94.16% accuracy · AUC 0.9895                          |

Beyond AI screening, the platform provides:

- **Digital memory bank** — permanent cloud database replacing paper reports
- **Shared folder pipeline** — auto-processes exports from any of AFIO's 10 machines
- **OCR report parsing** — extracts clinical values from machine-generated report images
- **DICOM support** — exports and imports standard medical imaging format
- **AI clinical reports** — Groq LLaMA 3.3 70B generates clinical narratives
- **Hospital-grade dashboard** — role-based access for technicians, doctors, admins

---

## Prerequisites

| Tool          | Version                  |
| ------------- | ------------------------ |
| Node.js       | 18+                      |
| Python        | 3.11 or 3.12 (not 3.13+) |
| MongoDB Atlas | Free tier or above       |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/arsalsaeed27/AI-Based-Screening-System
cd AI-Based-Screening-System
```

### 2. Install Node dependencies

```bash
cd backend
npm install
```

### 3. Set up Python virtual environment

```bash
# Windows
python -m venv C:/gradcam-venv
C:/gradcam-venv/Scripts/pip install torch torchvision timm flask \
  onnxruntime numpy pillow opencv-python easyocr pydicom pynetdicom

# Linux / Mac
python3.11 -m venv gradcam-venv
source gradcam-venv/bin/activate
pip install torch torchvision timm flask onnxruntime numpy \
  pillow opencv-python easyocr pydicom pynetdicom
```

### 4. Add ONNX model files

Download or train the three models and place them in the `models/` folder:

```
models/smoke_test.onnx              # DR model (input: 1×3×224×224)
models/glaucoma_model.onnx          # Glaucoma model (input: 1×3×640×640)
models/hr_efficientnet_model.onnx   # HR model (input: 1×3×300×300)
```

> Models are not included in the repository due to size. Contact the team for access.

### 5. Create `.env` file

Create `backend/.env`:

```env
MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/retinal_system
GROQ_API_KEY=your_groq_api_key_here
GRADCAM_PYTHON=C:/gradcam-venv/Scripts/python.exe
PORT=3000
```

**Getting API keys:**

- MongoDB Atlas: [mongodb.com/atlas](https://mongodb.com/atlas) → free tier → get connection string
- Groq API: [console.groq.com](https://console.groq.com) → free tier → create API key

### 6. Run the server

```bash
cd backend
node server.js
```

Expected output:

```
MongoDB connected
[GradCAM] * Running on http://127.0.0.1:5000
DR model: input size OK (224×224)
Glaucoma model: input size OK (640×640)
HR model: input size OK (300×300)
[Watcher] Watching folder: .../backend/shared_folder
Server listening on port 3000
```

Open: **http://localhost:3000/app.html**

---

## API Endpoints

### Screening

| Method | Endpoint            | Description                            |
| ------ | ------------------- | -------------------------------------- |
| POST   | `/predict`          | DR screening + Grad-CAM (image upload) |
| POST   | `/predict-glaucoma` | Glaucoma CDR segmentation              |
| POST   | `/predict-hr`       | Hypertensive retinopathy detection     |

### Patient Records

| Method | Endpoint         | Description                       |
| ------ | ---------------- | --------------------------------- |
| POST   | `/save-scan`     | Save scan to MongoDB              |
| GET    | `/scans`         | Get last 100 scans                |
| GET    | `/scans/:scanId` | Get one scan with full detail     |
| PATCH  | `/scans/:scanId` | Update scan status / doctor notes |
| DELETE | `/scans/:scanId` | Delete scan                       |

### AI Reports

| Method | Endpoint           | Description                       |
| ------ | ------------------ | --------------------------------- |
| POST   | `/generate-report` | Groq NLP clinical narrative       |
| POST   | `/ocr-report`      | EasyOCR on machine report image   |
| POST   | `/ocr-summary`     | Groq interpretation of OCR values |

### DICOM

| Method | Endpoint                      | Description                           |
| ------ | ----------------------------- | ------------------------------------- |
| POST   | `/dicom-upload`               | Upload DICOM file, parse + AI analyze |
| GET    | `/dicom-scans`                | All DICOM scans from MongoDB          |
| GET    | `/dicom-scans/:scanId`        | One DICOM scan with full detail       |
| GET    | `/scans/:scanId/export-dicom` | Export existing scan as .dcm file     |

### Shared Folder

| Method | Endpoint        | Description                        |
| ------ | --------------- | ---------------------------------- |
| GET    | `/folder-scans` | All folder-watcher processed files |

### System

| Method | Endpoint  | Description                  |
| ------ | --------- | ---------------------------- |
| GET    | `/health` | MongoDB + server status ping |

---

## Shared Folder Pipeline

The folder watcher automatically processes any image file dropped into `backend/shared_folder/`:

```
Technician exports from machine software
        ↓
Saves file to backend/shared_folder/
        ↓
chokidar detects new file (within 2 seconds)
        ↓
EasyOCR extracts text from report image
        ↓
Parser extracts clinical values
        ↓
Saves to MongoDB FolderScan collection
        ↓
Socket.IO notifies dashboard
```

---

## DICOM Support

**Export existing scan as DICOM:**

```
Dashboard → Scan History → Select scan → Export DICOM
```

**Upload DICOM for AI analysis:**

```
Dashboard → DICOM Viewer → Upload .dcm file → Analyze
```

The AI analyzer uses Groq LLaMA to identify abnormalities from DICOM-extracted values without hardcoded rules — it applies clinical knowledge from its training to flag what is abnormal and why.

---

## Dashboard Pages

| Page           | Description                                                   |
| -------------- | ------------------------------------------------------------- |
| Dashboard      | Live stats, recent activity, 7-day summary                    |
| New Screening  | 4-step workflow — patient info, conditions, upload, results   |
| Patients       | Searchable patient list with individual timeline view         |
| Scan History   | Full scan detail with DR/Glaucoma/HR results                  |
| Analytics      | Population-level charts — DR distribution, age groups, trends |
| AI Performance | Model accuracy metrics and validation results                 |
| Doctor Review  | Approve/modify AI findings, add clinical notes                |
| OCR Report     | Upload machine report image, extract values, get AI summary   |
| DICOM Viewer   | Upload/export DICOM files, AI abnormality analysis            |
| Admin          | Data management, CSV export, audit logs                       |
| Settings       | Theme, display, report preferences                            |

---

## Datasets Used

| Model    | Dataset                                    | Images |
| -------- | ------------------------------------------ | ------ |
| DR       | APTOS 2019 + EyePACS (balanced)            | 40,580 |
| Glaucoma | REFUGE train + val + test                  | 1,200  |
| HR       | Zoya77 + HRDC + EyePACS normals (balanced) | 772    |

---

## Environment Variables Reference

| Variable         | Required | Description                                                |
| ---------------- | -------- | ---------------------------------------------------------- |
| `MONGODB_URI`    | Yes      | MongoDB Atlas connection string                            |
| `GROQ_API_KEY`   | Yes      | Groq API key for NLP reports                               |
| `GRADCAM_PYTHON` | Yes      | Path to Python 3.11/3.12 executable with packages          |
| `PORT`           | No       | Server port (default: 3000)                                |
| `WATCH_FOLDER`   | No       | Custom shared folder path (default: backend/shared_folder) |
| `CHECKPOINT_DIR` | No       | Training checkpoint directory                              |

---

## Tech Stack

**Backend:** Node.js · Express · MongoDB Atlas · Mongoose · Socket.IO · Multer · chokidar  
**AI Models:** PyTorch · ONNX Runtime · EfficientNet-B3 · DeepLabV3+ ResNet50  
**Grad-CAM:** Flask · PyTorch hooks · timm  
**OCR:** EasyOCR · regex parsers  
**DICOM:** pydicom · pynetdicom  
**NLP:** Groq API · LLaMA 3.3 70B  
**Frontend:** Vanilla HTML/CSS/JS · Chart.js · jsPDF · Socket.IO client

---

## Known Limitations

- Models validated on research datasets — not yet validated on local Pakistani patient population
- OCR accuracy depends on image/scan quality
- Grad-CAM available for DR only (not Glaucoma or HR)
- DICOM private tag extraction requires real machine files for calibration
- Blue-channel fundus images not supported

---

## License

This project was developed as part of an internship at Neudym AI under the supervision of Military College of Signals, NUST. All rights reserved.

---

_AFIO ScanAI · Military College of Signals, NUST · 2026_
