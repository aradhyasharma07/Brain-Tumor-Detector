# NeuroScan AI: Deep Learning Brain Tumor MRI Classification System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.3-black.svg)](https://nextjs.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-38bdf8.svg)](https://tailwindcss.com/)
[![Verified Test Accuracy](https://img.shields.io/badge/Verified%20Test%20Accuracy-95.06%25-brightgreen.svg)]()

NeuroScan AI is a full-stack medical intelligence system designed for automated, reliable multi-class classification of brain tumors from Magnetic Resonance Imaging (MRI) scans. Leveraging transfer learning with a fine-tuned **ResNet-50** deep convolutional neural network, the system classifies brain scans into four distinct diagnostic categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor (Healthy)**.

---

## Table of Contents

- [Problem Statement & Objectives](#problem-statement--objectives)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Architecture & Directory Structure](#system-architecture--directory-structure)
- [Dataset Organization](#dataset-organization)
- [Model Architecture & Transfer Learning](#model-architecture--transfer-learning)
- [Training & Evaluation Pipeline](#training--evaluation-pipeline)
- [Verified Evaluation Metrics](#verified-evaluation-metrics)
- [Confusion Matrix & Per-Class Performance](#confusion-matrix--per-class-performance)
- [Model Checkpoint & Handover Notice](#model-checkpoint--handover-notice)
- [Local Setup & Installation](#local-setup--installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup (FastAPI & PyTorch)](#backend-setup-fastapi--pytorch)
  - [Frontend Setup (Next.js & Tailwind CSS)](#frontend-setup-nextjs--tailwind-css)
- [Running the Application](#running-the-application)
- [API Endpoints & Documentation](#api-endpoints--documentation)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Verification & Testing Scripts](#verification--testing-scripts)
- [Known Limitations & Future Scope](#known-limitations--future-scope)
- [Medical & Academic Disclaimer](#medical--academic-disclaimer)

---

## Problem Statement & Objectives

Brain tumors represent one of the most critical oncological challenges worldwide. Prompt, accurate diagnosis is vital for clinical intervention planning, patient prognosis, and surgical navigation. Manual radiologic evaluation of MRI slices requires specialized expertise and can be time-consuming.

**Core Objectives:**
1. **Automated Multi-Class Identification:** Rapidly distinguish between primary intracranial tumor types and healthy brain scans.
2. **Transfer Learning Efficiency:** Fine-tune deep feature representations from a pre-trained ResNet-50 backbone to achieve superior diagnostic accuracy without training from scratch.
3. **Calibrated Confidence Scoring:** Deliver class probability distributions across all four diagnostic possibilities to give healthcare practitioners full visibility into model certainty.
4. **Intuitive Clinical Interface:** Provide a responsive, modern radiology workstation UI with drag-and-drop upload, live API heartbeat monitoring, and instant inference feedback.

---

## Key Features

- **End-to-End Diagnostic Pipeline:** Accepts MRI scan uploads (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`), performs automated pre-processing, runs inference on a fine-tuned ResNet-50 model, and returns full class probabilities.
- **Four Diagnostic Classes:** Differentiates between:
  - **Glioma:** Tumors arising from glial tissue.
  - **Meningioma:** Slow-growing tumors originating from the meninges.
  - **Pituitary Tumor:** Abnormal growths inside the sella turcica / pituitary gland.
  - **No Tumor:** Normal, healthy brain tissue scans.
- **Validated 95.06% Test Accuracy:** Evaluated and confirmed on an independent balanced benchmark test split of 1,600 MRI images (400 per class).
- **FastAPI Inference Server:** High-performance asynchronous REST API supporting multipart file uploads, robust schema validation via Pydantic v2, and integrated CORS handling.
- **Next.js Radiology Workstation Frontend:** Modern dark-mode interface built with Next.js 16 (Turbopack) and Tailwind CSS v4, featuring:
  - Interactive file drag-and-drop dock station with real-time scanline animations.
  - Health ping HUD indicator that automatically monitors backend API availability.
  - Visual probability distribution progress bars across all four tumor categories.
  - Diagnostic result cards with color-coded severity indicators and confidence scoring.

---

## Technology Stack

### Machine Learning & Backend
- **Language:** Python 3.12
- **Framework:** PyTorch 2.13.0 & Torchvision 0.28.0
- **Model Architecture:** ResNet-50 (Deep Residual Network with Bottleneck Residual Blocks)
- **API Framework:** FastAPI 0.141.1 with Starlette and Pydantic v2
- **Image Processing:** Pillow (PIL) 12.3.0
- **ASGI Server:** Uvicorn 0.52.1

### Frontend
- **Framework:** Next.js 16.3.0 (React 19.2.8, App Router, Turbopack)
- **Styling:** Tailwind CSS v4 with custom keyframe animations
- **Iconography:** Lucide React 1.33.0
- **Language:** TypeScript 5 (Strict Mode enabled)

---

## System Architecture & Directory Structure

```
Brain Tumor Detector/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application entry point, CORS, and health endpoints
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── prediction_router.py    # POST /api/v1/predict upload and inference handler
│   ├── models/
│   │   ├── __init__.py
│   │   └── resnet_model.py             # ResNet-50 architecture builder and parameter utilities
│   ├── saved_models/
│   │   ├── resnet50_brain_tumor.pth    # Best trained model weights (Epoch 19, 97.77% Val Acc)
│   │   └── resnet50_brain_tumor_v1.pth # Initial baseline checkpoint
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction_schema.py        # Pydantic v2 schemas for API requests and responses
│   ├── services/
│   │   ├── __init__.py
│   │   ├── inference_service.py        # Checkpoint loader, transform pipeline, and predict logic
│   │   └── training_service.py         # Full training loop, early stopping, and scheduler
│   ├── utils/
│   │   ├── __init__.py
│   │   └── dataset_utils.py            # PyTorch Dataset, augmentation transforms, and DataLoader splits
│   ├── evaluate.py                     # Batch PyTorch DataLoader test dataset evaluation script
│   ├── evaluate_model.py               # Single-image inference evaluation script with CSV export
│   ├── requirements.txt                # Python backend dependencies
│   ├── test_results.csv                # Detailed evaluation records per test image
│   ├── train.py                        # CLI entry point to trigger training
│   ├── verify_api.py                   # Automated test script for FastAPI endpoints
│   └── verify_inference.py             # Smoke test script for model checkpoint and inference pipeline
│
├── frontend/
│   ├── app/
│   │   ├── favicon.ico
│   │   ├── globals.css                 # Global styles and Tailwind v4 themes
│   │   ├── layout.tsx                  # Root layout wrapper
│   │   └── page.tsx                    # Main workstation page with diagnostic dashboard
│   ├── components/
│   │   ├── ApiStatus.tsx               # Real-time backend connectivity heartbeat indicator
│   │   ├── Header.tsx                  # Application top navigation bar
│   │   ├── PredictionResult.tsx        # Diagnosis result card and 4-class probability breakdown
│   │   └── UploadZone.tsx              # MRI file upload box with animated scan lines
│   ├── lib/
│   │   └── api.ts                      # Client-side API fetch client and error handler
│   ├── types/
│   │   └── api.ts                      # TypeScript definitions for backend payloads
│   ├── eslint.config.mjs
│   ├── next.config.ts
│   ├── package.json
│   ├── postcss.config.mjs
│   └── tsconfig.json
│
├── docs/                               # Project documentation assets
├── LICENSE                             # MIT License
└── README.md                           # Master project documentation
```

---

## Dataset Organization

The system is trained and evaluated on an open-access Brain Tumor MRI dataset organized into stratified training and testing splits:

```
Brain Tumor Dataset/
├── Training/
│   ├── glioma/          # ~1,321 images
│   ├── meningioma/      # ~1,339 images
│   ├── notumor/         # ~1,595 images
│   └── pituitary/       # ~1,457 images
└── Testing/
    ├── glioma/          # 400 images
    ├── meningioma/      # 400 images
    ├── notumor/         # 400 images
    └── pituitary/       # 400 images
```

- **Stratified Training Split:** The Training folder is divided into an 80% training set and a 20% validation set with deterministic random seeding (`seed=42`).
- **Independent Test Split:** Exactly 1,600 unseen MRI scans (balanced: 400 per class) held out exclusively for final model benchmarking.

---

## Model Architecture & Transfer Learning

```
Input Image (224 x 224 x 3)
         │
         ▼
[ ResNet-50 Pretrained Feature Extractor ]
   ├── Conv1 (7x7, 64, stride 2) + MaxPool
   ├── Layer 1 (3 Residual Blocks) -> Frozen
   ├── Layer 2 (4 Residual Blocks) -> Frozen
   ├── Layer 3 (6 Residual Blocks) -> Frozen
   └── Layer 4 (3 Residual Blocks) -> Unfrozen (Fine-Tuned)
         │
         ▼
[ Adaptive Average Pooling (7x7 -> 1x1) ]
         │  (2048 Features)
         ▼
[ Custom Classification Head ]
   ├── Linear (2048 -> 512)
   ├── ReLU Activation
   ├── Dropout (p = 0.30)
   └── Linear (512 -> 4 Classes)
         │
         ▼
[ Softmax Layer ] -> [ Glioma, Meningioma, Pituitary, No Tumor ]
```

### Parameter Breakdown
- **Total Parameters:** 24,561,476
- **Trainable Parameters:** 15,992,324 (~65.1%)
- **Frozen Parameters:** 8,569,152 (~34.9%)

Fine-tuning `layer4` alongside the new dense classification head allows the network to adapt high-level visual features (tumor contours, tissue contrast, edema margins) while preserving fundamental low-level edge and texture detectors learned from ImageNet.

---

## Training & Evaluation Pipeline

The training pipeline is implemented in `backend/services/training_service.py`:
- **Image Preprocessing:**
  - Training: Random horizontal flip ($p=0.5$), random rotation ($\pm 10^\circ$), random affine translation ($\pm 5\%$), scaling ($0.95 \times - 1.05 \times$), resize to $224 \times 224$, ImageNet normalization.
  - Validation/Testing: Deterministic resize to $224 \times 224$ and ImageNet normalization.
- **Loss Function:** `torch.nn.CrossEntropyLoss()`
- **Optimizer:** `torch.optim.Adam` (Initial Learning Rate: $1 \times 10^{-4}$)
- **Learning Rate Scheduler:** `ReduceLROnPlateau(mode="max", factor=0.5, patience=3)`
- **Epochs:** 20 epochs with automatic checkpointing of the best validation accuracy.

---

## Verified Evaluation Metrics

> **Note on Metric Verification:** Metrics were independently verified on the full 1,600-image test set using `backend/evaluate.py` with checkpoint `resnet50_brain_tumor.pth`.

| Metric | Result | Context / Dataset |
|---|:---:|---|
| **Overall Test Accuracy** | **95.06%** | Evaluated on 1,600 unseen test MRI images (1,521 correct / 79 incorrect) |
| **Peak Validation Accuracy** | **97.77%** | Best validation epoch performance achieved during training (Epoch 19) |
| **Training Accuracy** | **99.55%** | Training set accuracy at best checkpoint epoch |
| **Overall Test Loss** | **0.8465** | Cross-Entropy Loss over 1,600 test images |

---

## Confusion Matrix & Per-Class Performance

### Per-Class Metrics Table

| Class | Total Images | Correct Predictions | Per-Class Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Glioma** | 400 | 340 | **85.00%** | 0.9884 | 0.8500 | 0.9140 |
| **Meningioma** | 400 | 390 | **97.50%** | 0.8966 | 0.9750 | 0.9341 |
| **Pituitary Tumor** | 400 | 399 | **99.75%** | 0.9901 | 0.9975 | 0.9938 |
| **No Tumor (Healthy)** | 400 | 392 | **98.00%** | 0.9378 | 0.9800 | 0.9584 |
| **Macro Average** | **1,600** | **1,521** | — | **0.9532** | **0.9506** | **0.9501** |
| **Weighted Average** | **1,600** | **1,521** | — | **0.9532** | **0.9506** | **0.9501** |

### Confusion Matrix

```
                 PREDICTED
Actual Class    │   Glioma   │ Meningioma │ Pituitary  │  No Tumor  │ Total
────────────────┼────────────┼────────────┼────────────┼────────────┼───────
Glioma          │    340     │     36     │     1      │     23     │  400
Meningioma      │      4     │    390     │     3      │      3     │  400
Pituitary       │      0     │      1     │   399      │      0     │  400
No Tumor        │      0     │      8     │     0      │    392     │  400
────────────────┴────────────┴────────────┴────────────┴────────────┴───────
Total Predicted │    344     │    435     │   403      │    418     │ 1,600
```

---

## Model Checkpoint & Handover Notice

> [!IMPORTANT]
> **Required Checkpoint File:** `backend/saved_models/resnet50_brain_tumor.pth` (~226 MB)
> - **Inference Dependency:** This PyTorch checkpoint contains the fine-tuned weights for the ResNet-50 feature extractor (Layer 4) and custom 4-class classification head. It is strictly required by `InferenceService` to run predictions, verify endpoints, and evaluate test datasets.
> - **Exclusion from Git:** Because the trained weights file exceeds standard Git file size limits (226 MB), `backend/saved_models/` and all `*.pth` binary files are excluded from version control via `.gitignore`.
> - **Academic Submission & Repository Handover:**
>   - **Submission ZIP:** When submitting this project as a ZIP archive for university grading, college evaluation, or project viva, the `backend/saved_models/resnet50_brain_tumor.pth` file **must be explicitly included** inside the archive under the `backend/saved_models/` folder.
>   - **Cloning from Git:** Anyone cloning this repository will not receive the weights file automatically via `git clone` and must obtain `resnet50_brain_tumor.pth` separately and place it into `backend/saved_models/` before starting the FastAPI backend.

---

## Local Setup & Installation

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** Version 3.10 to 3.12 (Tested on Python 3.12.x)
- **Node.js:** Version 18.x or 20.x (Tested on Node.js 20+)
- **Git**

---

### Backend Setup (FastAPI & PyTorch)

1. Open **Windows PowerShell** and navigate to the project directory:
   ```powershell
   cd "d:\SEM 5 all projects\Brain Tumor Detector\backend"
   ```

2. Create a virtual environment (if not already present):
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

4. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

---

### Frontend Setup (Next.js & Tailwind CSS)

1. Open a new PowerShell terminal and navigate to the frontend directory:
   ```powershell
   cd "d:\SEM 5 all projects\Brain Tumor Detector\frontend"
   ```

2. Install npm dependencies:
   ```powershell
   npm install
   ```

---

## Running the Application

### 1. Start the FastAPI Backend
From the activated `backend` terminal:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API root: `http://127.0.0.1:8000`
- Interactive Swagger UI: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### 2. Start the Next.js Frontend
From the `frontend` terminal:
```powershell
npm run dev
```
- Web Application: `http://localhost:3000`

---

## API Endpoints & Documentation

| Method | Endpoint | Description | Request / Payload | Success Response |
|---|---|---|---|---|
| `GET` | `/` | API Information | None | `{"message": "Brain Tumor Detection API is active", ...}` |
| `GET` | `/health` | Server Health Status | None | `{"status": "healthy", "service": "..."}` |
| `POST` | `/api/v1/predict` | MRI Image Classification | `multipart/form-data`<br>Field: `file` (Image) | See JSON below |

### Sample Prediction Response (`200 OK`)
```json
{
  "status": "success",
  "predicted_class": "glioma",
  "confidence": 0.9942,
  "class_index": 0,
  "probabilities": {
    "glioma": 0.994278,
    "meningioma": 0.003512,
    "pituitary": 0.001205,
    "notumor": 0.001005
  }
}
```

### Error Responses
- `400 Bad Request`: If file format is not allowed (`.txt`, `.pdf`, etc.), file is empty, or bytes cannot be decoded as an image.
- `422 Unprocessable Entity`: If `file` form parameter is missing.

---

## Configuration & Environment Variables

### Frontend Configuration
The frontend connects to `http://127.0.0.1:8000` by default. A template is provided in `frontend/.env.example`. If your backend is hosted on a different port or server, create a `.env.local` file inside the `frontend/` directory (or copy `.env.example`):

```env
# frontend/.env.local (template available at frontend/.env.example)
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

> **Security Note:** Never commit sensitive credentials or secret API tokens to version control. The Next.js public URL prefix (`NEXT_PUBLIC_`) is exposed to the client browser and is strictly for non-sensitive public endpoints.

---

## Verification & Testing Scripts

The repository includes standalone validation scripts located in the `backend/` directory:

### 1. Verify Model Inference Pipeline
Performs a smoke test on checkpoint loading, tensor shapes, and single-image prediction:
```powershell
python backend/verify_inference.py
```
*Optional:* Specify a custom image using `--image-path <path_to_image>`.

### 2. Verify FastAPI Endpoints
Tests the `/health` endpoint, valid image upload, invalid extension rejection, and corrupted byte detection:
```powershell
python backend/verify_api.py
```

### 3. Evaluate on Benchmark Dataset
Runs complete batch evaluation across the 1,600-image test set:
```powershell
python backend/evaluate.py --data-dir "D:\SEM 5 all projects\Brain Tumor Dataset\Testing"
```

### 4. Single-Image Evaluation with CSV Export
Generates a detailed per-image record (`test_results.csv`) and prints the classification report:
```powershell
python backend/evaluate_model.py
```

---

## Known Limitations & Future Scope

### Current Limitations
1. **2D Axial Slices:** The current model processes single 2D axial MRI slices; multi-planar volumetric MRI sequences (coronal/sagittal) are evaluated independently per image.
2. **Binary Contrast Dependency:** Performance depends on standardized MRI contrast; severe motion artifacts or heavily compressed images can impact prediction confidence.
3. **Absence of Localization Maps:** While classification accuracy is high (95.06%), the model does not currently generate pixel-level segmentation boundaries or bounding boxes.

### Future Scope (Proposed Enhancements)
- **Grad-CAM Visual Saliency:** Integrating Gradient-weighted Class Activation Mapping (Grad-CAM) onto `layer4` feature maps to produce visual heatmaps of suspicious regions.
- **DICOM File Ingestion:** Supporting native `.dcm` medical image standard formats directly alongside standard raster images.
- **Docker Containerization:** Bundling backend and frontend into multi-stage `Dockerfile` and `docker-compose.yml` for reproducible cloud deployment on AWS/GCP/Render.

---

## Medical & Academic Disclaimer

> **IMPORTANT MEDICAL NOTICE:**
> This software is an **academic research and educational project** developed as part of undergraduate coursework. It is **not** an FDA-approved or CE-certified medical device, and it is **not** intended to replace the clinical judgment, diagnosis, or advice of a qualified healthcare professional, radiologist, or neurosurgeon. All predictions must be validated by licensed medical specialists.

---

## License

This project is licensed under the [MIT License](LICENSE) — free for academic, educational, and research use.
