# NeuroScan AI: Deep Learning Brain Tumor MRI Classification & Explainability System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16.4-black.svg)](https://nextjs.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-38bdf8.svg)](https://tailwindcss.com/)
[![ReportLab](https://img.shields.io/badge/ReportLab-5.0+-red.svg)](https://www.reportlab.com/)
[![Verified Test Accuracy](https://img.shields.io/badge/Verified%20Test%20Accuracy-95.06%25-brightgreen.svg)]()

NeuroScan AI is a full-stack deep learning intelligence system designed for automated multi-class classification and visual explainability of brain tumors from Magnetic Resonance Imaging (MRI) scans for academic and educational research. Leveraging transfer learning with a fine-tuned **ResNet-50** deep convolutional neural network, the system classifies axial brain scans into four diagnostic categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor (Healthy)**.

The platform integrates a **4-layer input validation and out-of-distribution (OOD) safeguard engine**, **Grad-CAM visual saliency heatmaps**, in-memory **automated PDF research report generation via ReportLab**, and a **tabbed radiology workstation dashboard** built with Next.js 16 and Tailwind CSS v4.

---

## Table of Contents

- [Problem Statement & Objectives](#problem-statement--objectives)
- [Key Features & Implemented Capabilities](#key-features--implemented-capabilities)
  - [1. Four-Class Brain Tumor Classification](#1-four-class-brain-tumor-classification)
  - [2. Multi-Stage MRI Input Validation & OOD Safeguards](#2-multi-stage-mri-input-validation--ood-safeguards)
  - [3. Grad-CAM Explainability & Heatmap Visualization](#3-grad-cam-explainability--heatmap-visualization)
  - [4. Automated PDF Research Report Generation (ReportLab)](#4-automated-pdf-research-report-generation-reportlab)
  - [5. Interactive 3-Tab Diagnostic Dashboard](#5-interactive-3-tab-diagnostic-dashboard)
  - [6. Calibrated Confidence Scoring & Uncertainty Advisory](#6-calibrated-confidence-scoring--uncertainty-advisory)
- [Technology Stack](#technology-stack)
- [System Architecture & Data Pipeline](#system-architecture--data-pipeline)
- [Project Directory Structure](#project-directory-structure)
- [Dataset Organization](#dataset-organization)
- [Model Architecture & Transfer Learning](#model-architecture--transfer-learning)
- [Training & Optimization Pipeline](#training--optimization-pipeline)
- [Verified Evaluation Metrics & Confusion Matrix](#verified-evaluation-metrics--confusion-matrix)
- [Pre-Trained Model Download & Checkpoint Notice](#pre-trained-model-download--checkpoint-notice)
- [Local Setup & Installation](#local-setup--installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup (FastAPI & PyTorch)](#backend-setup-fastapi--pytorch)
  - [Frontend Setup (Next.js & Tailwind CSS)](#frontend-setup-nextjs--tailwind-css)
  - [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Application Usage & Walkthrough](#application-usage--walkthrough)
- [API Endpoints & Documentation](#api-endpoints--documentation)
- [Confidence Scores Analysis & Limitations](#confidence-scores-analysis--limitations)
- [Verification & Testing Suites](#verification--testing-suites)
- [Known Limitations & Future Scope](#known-limitations--future-scope)
- [Medical & Academic Disclaimer](#medical--academic-disclaimer)
- [License](#license)

---

## Problem Statement & Objectives

Intracranial tumors represent one of the most critical oncological challenges worldwide. Prompt, accurate diagnosis is vital for clinical intervention planning, patient prognosis, and surgical navigation. Manual radiologic evaluation of MRI slices requires specialized neuro-radiological expertise and can be subjective under heavy clinical workloads.

**Core Objectives:**
1. **Automated Multi-Class Identification:** Rapidly distinguish between three primary intracranial tumor types and healthy neuroanatomy from 2D axial MRI slices.
2. **Defensive Input Validation:** Prevent adversarial or out-of-domain inputs (e.g., natural photos, screenshots, inverted scans, corrupt files) from reaching the neural network by implementing a multi-stage validation filter.
3. **Model Interpretability (Explainable AI):** Render Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps overlaid on the original MRI scan to highlight regions guiding classification.
4. **Automated Research Documentation:** Deliver downloadable, in-memory PDF research reports containing diagnostic metrics, dual-image visual evidence, and regulatory notices.
5. **Calibrated Confidence Scoring:** Deliver class probability distributions across all four diagnostic possibilities to give practitioners full visibility into model certainty, including elevated uncertainty alerts when confidence drops below 70%.
6. **Intuitive Clinical Interface:** Provide a dark-mode radiology workstation UI with drag-and-drop ingestion, live API heartbeat monitoring, and a 3-tab results dashboard.

---

## Key Features & Implemented Capabilities

### 1. Four-Class Brain Tumor Classification
The model evaluates axial brain MRI slices and categorizes them into four distinct diagnostic classes:
- **Glioma (Intra-axial Neoplasm):** Tumors arising from glial cells (astrocytes, oligodendrocytes, ependymal cells) infiltrating brain parenchyma.
- **Meningioma (Extra-axial Neoplasm):** Typically slow-growing, benign tumors originating from arachnoid cap cells of the meninges covering the brain and spinal cord.
- **Pituitary Tumor (Sellar / Neuroendocrine Mass):** Abnormal growths within the sella turcica / pituitary gland region affecting endocrine regulation.
- **No Tumor (Normal Neuroanatomy):** Healthy brain MRI slices showing standard ventricular symmetry, gray-white matter differentiation, and absence of detectable intracranial masses.

### 2. Multi-Stage MRI Input Validation & OOD Safeguards
To prevent arbitrary images (e.g., photos of pets, documents, synthetic noise, corrupt byte streams) from producing false diagnoses, the backend implements a 4-layer defense-in-depth validation engine (`MriValidationService`) before inference:
- **Layer 1: Payload & Safe Decoding Checks:** Verifies file size ($\le 10$ MB), allowed extensions (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`), PIL format headers, single-frame static image constraints (rejects animated GIFs/APNGs), minimum resolution ($64 \times 64$ px), maximum resolution ($4096 \times 4096$ px), and aspect ratio ($0.33 \le \text{AR} \le 3.0$).
- **Layer 2: Visual & Photometric Domain Checks:**
  - *Color Saturation:* Ensures mean saturation $\le 0.12$ (genuine brain MRIs are monochromatic grayscale acquisitions).
  - *Peripheral Dark Perimeter:* Confirms that $\ge 60\%$ of perimeter pixels represent the dark scanner framing ($\text{intensity} < 0.28$) with mean border intensity $\le 0.30$.
  - *Central Anatomical Contrast:* Validates that the central cranial region has mean intensity between $0.05$ and $0.85$ and standard deviation $\ge 0.04$ (rejects solid black, blank white, or flat images).
- **Layer 3: Morphological Cranial Structure Verification:**
  - *High-Frequency Noise:* Laplacian variance check ($\le 4000.0$) rejects synthetic uncorrelated noise.
  - *Contour & Area Segmentation:* Detects cranial foreground contour using Otsu thresholding; requires cranial area ratio $\ge 10\%$ of total frame.
  - *Centroid Centering:* Checks that the cranial mass centroid $(c_x, c_y)$ is centrally positioned within $25\% - 75\%$ of frame dimensions.
- **Layer 4: Deep Feature & Logit OOD Screening:** Extracts intermediate 512-dimensional feature norms ($\le 3500.0$) and raw logit activations ($\le 500.0$) from ResNet-50 to reject adversarial or anomalous inputs.

Any validation failure immediately aborts execution and returns an informative **HTTP 400 Bad Request** error with a clear clinical explanation.

### 3. Grad-CAM Explainability & Heatmap Visualization
To eliminate the "black box" nature of deep neural networks, NeuroScan AI integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)** implemented in `GradCamService`:
- **Target Layer:** Connects PyTorch forward and backward hooks into the final Bottleneck residual block of ResNet-50 (`model.layer4[-1]`).
- **Feature Computation:** Captures $2048 \times 7 \times 7$ feature activation maps and backpropagates gradients with respect to the target predicted class score.
- **Global Average Pooling (GAP):** Computes importance weights $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$.
- **ReLU Rectification:** Applies ReLU to isolate features that have a positive correlation with the target class.
- **Bilinear Upsampling & Color Blending:** Upsamples the normalized activation map to $224 \times 224$ px, applies OpenCV `COLORMAP_JET`, and alpha-blends ($\alpha = 0.45$) the heatmap over the original grayscale MRI.
- **Output:** Encodes the blended overlay into a Base64 JPEG data URI returned in the prediction response and rendered side-by-side with the original scan.
- **Thread Safety:** Manages hooks with threading locks and ensures hook cleanup via `finally` blocks to prevent memory leaks across concurrent requests.

### 4. Automated PDF Research Report Generation (ReportLab)
The backend features an authoritative clinical research report generation service (`PdfReportService`) using **ReportLab Standard**:
- **In-Memory Buffer (`io.BytesIO`):** Generates binary PDF documents entirely in memory without writing temporary files to the server disk, ensuring zero file-system clutter and protecting system privacy.
- **Structured Report Sections:**
  - *Document Header:* System branding, unique Report ID (`REP-<HEX>`), UTC generation timestamp, model name, and input resolution ($224 \times 224$ px).
  - *Primary Verdict Banner:* Highlighted classification badge with color-coded severity (Emerald for healthy, Crimson for neoplasms) and confidence percentage.
  - *Diagnostic Probability Table:* Complete 5-column breakdown showing Class Name, Clinical Category, Raw Float Score, Percentage, and Selection Status.
  - *Dual-Image Visual Evidence:* High-resolution side-by-side rendering of the original input MRI scan and the Grad-CAM activation overlay ($180 \times 180$ pt each).
  - *Prominent Clinical Disclaimer:* Standardized framed callout reiterating educational and research boundaries.
- **Download Headers:** Transmitted via `POST /api/v1/report` with `Content-Type: application/pdf` and `Content-Disposition: attachment; filename="neuroscan_report_<REPORT_ID>.pdf"`.

### 5. Interactive 3-Tab Diagnostic Dashboard
The frontend (`PredictionResult.tsx`) organizes the diagnostic evaluation into three structured views:
- **Top Summary Dock:** Includes a quick return button ("New Analysis"), unique Prediction ID badge, primary class verdict pill, confidence score, and a one-click "PDF Report" download button with live loading spinner and success state.
- **Tab 1: Overview:**
  - *Primary Verdict Card:* Displays category badge, tumor type, clinical summary, and confidence score.
  - *Original Scan Preview:* Shows the uploaded MRI slice with resolution tags and file size.
  - *Multi-Class Confidence Distribution:* Four responsive progress bars representing probability percentages for Glioma, Meningioma, Pituitary, and No Tumor with an active "TOP SCORE" indicator.
  - *Low Confidence Advisory:* Displays an amber alert banner if the primary confidence score is below 70%, warning of elevated radiological ambiguity.
- **Tab 2: Explainability (Grad-CAM):**
  - Side-by-side visual comparison between the original axial MRI scan and the colorized Grad-CAM overlay.
  - Colormap gradient legend (Baseline Blue $\rightarrow$ Salient Red).
  - Annotation specifying target layer (`ResNet-50 layer4[-1]`).
  - Explainability notice clarifying that heatmaps highlight computational feature focus rather than confirmed histological tumor margins.
- **Tab 3: Clinical Report & PDF:**
  - Executive metadata summary with Report ID, local browser-formatted timestamp, primary verdict, and confidence score.
  - Detailed probability breakdown table.
  - Dedicated PDF Download hub card.
  - Medical and regulatory notice.

### 6. Calibrated Confidence Scoring & Uncertainty Advisory
- The model computes normalized class probabilities via the Softmax function over the 4 output logits:
  $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{4} e^{z_j}}$$
- **Uncertainty Advisory:** If the highest probability falls below $0.70$ (70.0%), the dashboard renders an explicit amber advisory recommending inspection of alternative differential classes and review of adjacent axial slices.
- **Notice on Limitations:** Confidence scores represent statistical correlation within the training distribution and must never be interpreted as an empirical clinical probability or standalone diagnosis.

---

## Technology Stack

### Machine Learning & Backend
| Technology | Version | Purpose |
|---|:---:|---|
| **Python** | 3.12 | Core backend programming runtime |
| **PyTorch** | 2.13.0 | Deep learning framework, tensor execution, and autograd hooks |
| **Torchvision** | 0.28.0 | Pre-trained ResNet-50 weights and image transformations |
| **FastAPI** | 0.141.1 | High-performance asynchronous REST API framework |
| **Pydantic** | 2.13.4 (v2) | Strict data validation, request parsing, and schema generation |
| **Starlette** | 1.6.0 | ASGI routing, HTTP exceptions, and streaming responses |
| **Pillow (PIL)** | 12.3.0 | Image decoding, formatting, and buffer manipulations |
| **OpenCV (`opencv-python`)** | 5.0.0.93 | Morphological contour analysis, Laplacian variance, and JET colormap blending |
| **ReportLab** | 5.0+ | In-memory clinical PDF document compilation |
| **Uvicorn** | 0.52.1 | Asynchronous ASGI production server |
| **NumPy** | 2.5.1 | High-performance array operations for pixel and feature computations |

### Frontend Workstation
| Technology | Version | Purpose |
|---|:---:|---|
| **Next.js** | 16.4.2 | React framework with App Router and Turbopack bundler |
| **React** | 19.2.8 | Declarative component UI library |
| **Tailwind CSS** | v4 | Modern utility-first CSS styling and custom scanline animations |
| **Lucide React** | 1.33.0 | Medical and radiological iconography |
| **TypeScript** | 5.x | Strict type safety across client payloads and state models |

---

## System Architecture & Data Pipeline

```
                                 [ User / Web Browser ]
                                           │
                        1. Upload MRI Scan (.jpg, .png, etc.)
                                           │
                                           ▼
                                 [ Next.js Frontend ]
                        (Drag-and-Drop Dock / Validation Client)
                                           │
                        2. Multipart POST /api/v1/predict
                                           │
                                           ▼
                             [ FastAPI Inference Server ]
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 │                                                   │
     [ Layer 1: Payload Checks ]                         [ Layer 2: Domain Checks ]
     - File size <= 10 MB                                - Saturation <= 0.12 (Grayscale)
     - Allowed format & single-frame                     - Dark perimeter fraction >= 60%
     - Resolution [64..4096], AR [0.33..3.0]             - Contrast std >= 0.04
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 │                                                   │
   [ Layer 3: Morphological Checks ]                   [ Layer 4: Deep OOD Checks ]
   - Laplacian variance <= 4000                        - Feature norm <= 3500.0
   - Cranial contour area >= 10%                       - Max logit <= 500.0
   - Centroid centering [0.25..0.75]                                 │
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           │ (Validation Passed)
                                           ▼
                         [ ResNet-50 Transfer Learning Model ]
                         ├── Frozen Layers 1-3 (ImageNet Features)
                         ├── Fine-Tuned Layer 4 (High-Level Pathology)
                         └── Custom Head (AdaptivePool -> Linear -> ReLU -> Dropout -> Linear)
                                           │
                 ┌─────────────────────────┴─────────────────────────┐
                 │                                                   │
                 ▼                                                   ▼
      [ Softmax Probabilities ]                           [ Grad-CAM Service ]
      - Class probability distribution                    - Hooks into layer4[-1]
      - Top predicted class                               - Global Average Pooling
      - Confidence score                                  - JET colormap alpha blend
                 │                                                   │
                 └─────────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                         [ API JSON Response / PDF Service ]
                         ├── JSON Payload (Scores, Class, Grad-CAM URI)
                         └── On Request: ReportLab In-Memory PDF Build
                                           │
                                           ▼
                             [ Next.js Results Dashboard ]
                             ├── Overview Tab (Verdict & Distribution)
                             ├── Explainability Tab (Grad-CAM Overlay)
                             └── Report Tab (Downloadable PDF)
```

---

## Project Directory Structure

```
Brain Tumor Detector/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI application entry point, CORS, root & health endpoints
│   │   └── routers/
│   │       ├── __init__.py
│   │       └── prediction_router.py    # POST /predict, POST /gradcam, and POST /report endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   └── resnet_model.py             # ResNet-50 architecture builder and parameter utilities
│   ├── saved_models/
│   │   ├── resnet50_brain_tumor.pth    # Best trained model weights (Epoch 19, 97.77% Val Acc, ~226 MB)
│   │   └── resnet50_brain_tumor_v1.pth # Initial baseline checkpoint
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction_schema.py        # Pydantic v2 schemas for API requests, class scores, and responses
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gradcam_service.py          # Grad-CAM heatmap generator targeting ResNet-50 layer4[-1]
│   │   ├── inference_service.py        # Checkpoint loader, transform pipeline, OOD check, and predict logic
│   │   ├── pdf_service.py              # ReportLab in-memory PDF clinical research report generator
│   │   ├── training_service.py         # Full training loop, early stopping, and scheduler
│   │   └── validation_service.py       # Multi-layer MRI validation & OOD detection engine
│   ├── utils/
│   │   ├── __init__.py
│   │   └── dataset_utils.py            # PyTorch Dataset, augmentation transforms, and DataLoader splits
│   ├── evaluate.py                     # Batch PyTorch DataLoader test dataset evaluation script
│   ├── evaluate_model.py               # Single-image inference evaluation script with CSV export
│   ├── requirements.txt                # Python backend dependencies
│   ├── test_features.py                # Unit & integration test suite for confidence, Grad-CAM & PDF reports
│   ├── test_results.csv                # Detailed evaluation records per test image
│   ├── test_validation.py              # 23-test validation suite for MRI domain & OOD safeguards
│   ├── train.py                        # CLI entry point to trigger model training
│   ├── verify_api.py                   # Automated smoke test script for FastAPI endpoints
│   └── verify_inference.py             # Smoke test script for model checkpoint and inference pipeline
│
├── frontend/
│   ├── app/
│   │   ├── favicon.ico
│   │   ├── globals.css                 # Global styles and Tailwind v4 themes
│   │   ├── layout.tsx                  # Root layout wrapper with metadata
│   │   └── page.tsx                    # Main workstation page with diagnostic dashboard & state management
│   ├── components/
│   │   ├── ApiStatus.tsx               # Real-time backend connectivity heartbeat indicator
│   │   ├── Header.tsx                  # Application top navigation bar
│   │   ├── PredictionResult.tsx        # 3-tab results dashboard (Overview, Explainability, Clinical Report)
│   │   └── UploadZone.tsx              # Interactive drag-and-drop file upload box with animated scan lines
│   ├── lib/
│   │   └── api.ts                      # Client API fetch client (predict, health check, PDF download)
│   ├── types/
│   │   └── api.ts                      # TypeScript interfaces for API payloads and class scores
│   ├── .env.example                    # Environment template for frontend configuration
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

The system is trained and evaluated on an open-access Brain Tumor MRI dataset organized into stratified training and testing splits totaling 7,200 images:

```
Brain Tumor Dataset/
├── Training/                            # 5,600 images total (used for 80/20 train/val split)
│   ├── glioma/          # 1,400 images
│   ├── meningioma/      # 1,400 images
│   ├── notumor/         # 1,400 images
│   └── pituitary/       # 1,400 images
└── Testing/                             # Exactly 1,600 images (held-out independent benchmark)
    ├── glioma/          # 400 images
    ├── meningioma/      # 400 images
    ├── notumor/         # 400 images
    └── pituitary/       # 400 images
```

- **Stratified Training Split:** The Training directory contains 5,600 balanced images (1,400 per class), partitioned into an 80% training set (4,480 images) and a 20% validation set (1,120 images) using deterministic random seeding (`seed=42`).
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
   └── Layer 4 (3 Residual Blocks) -> Unfrozen (Fine-Tuned)  <── [Grad-CAM Hook Point: layer4[-1]]
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
- **Total Parameters:** 24,559,172
- **Trainable Parameters:** 16,015,876 (~65.21%)
- **Frozen Parameters:** 8,543,296 (~34.79%)

Fine-tuning `layer4` alongside the new dense classification head allows the network to adapt high-level visual features (tumor contours, tissue contrast, edema margins) while preserving fundamental low-level edge and texture detectors learned from ImageNet.

---

## Training & Optimization Pipeline

The training pipeline is implemented in `backend/services/training_service.py`:
- **Image Preprocessing:**
  - *Training:* Random horizontal flip ($p=0.5$), random rotation ($\pm 10^\circ$), random affine translation ($\pm 5\%$), scaling ($0.95 \times - 1.05 \times$), resize to $224 \times 224$, ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
  - *Validation / Testing:* Deterministic resize to $224 \times 224$ and ImageNet normalization.
- **Loss Function:** `torch.nn.CrossEntropyLoss()`
- **Optimizer:** `torch.optim.Adam` (Initial Learning Rate: $1 \times 10^{-4}$)
- **Learning Rate Scheduler:** `ReduceLROnPlateau(mode="max", factor=0.5, patience=3)`
- **Epochs:** 20 epochs with automatic checkpointing of the best validation accuracy epoch.

---

## Verified Evaluation Metrics & Confusion Matrix

> **Note on Metric Verification:** Metrics were independently verified on the full 1,600-image test set using `backend/evaluate.py` with checkpoint `resnet50_brain_tumor.pth`.

### Overall Benchmark Metrics

| Metric | Result | Context / Dataset |
|---|:---:|---|
| **Overall Test Accuracy** | **95.06%** | Evaluated on 1,600 unseen test MRI images (1,521 correct / 79 incorrect) |
| **Peak Validation Accuracy** | **97.77%** | Best validation epoch performance achieved during training (Epoch 19) |
| **Training Accuracy** | **99.55%** | Training set accuracy at best checkpoint epoch |
| **Overall Test Loss** | **0.8465** | Cross-Entropy Loss over 1,600 test images |

### Per-Class Performance Table

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

## Pre-Trained Model Download & Checkpoint Notice

The fine-tuned PyTorch checkpoint is distributed separately from the source code repository due to its large binary file size (~226 MB).

**Direct Download:** [resnet50_brain_tumor.pth — Model v1.0 Release](https://github.com/aradhyasharma07/Brain-Tumor-Detector/releases/download/v1.0/resnet50_brain_tumor.pth)

### Checkpoint Placement
Download `resnet50_brain_tumor.pth` and place it in the `backend/saved_models/` directory:

```text
backend/
└── saved_models/
    └── resnet50_brain_tumor.pth
```

> [!IMPORTANT]
> **Academic Submission & Repository Handover Notice:**
> - Because binary `.pth` weights exceed standard Git size thresholds, `backend/saved_models/` is excluded via `.gitignore`.
> - **University Viva / Grading Handover:** When submitting this project as an offline ZIP archive for academic evaluation, ensure that `resnet50_brain_tumor.pth` is explicitly copied into `backend/saved_models/` before archiving.
> - **Git Clones:** Anyone cloning this repository must manually download the weights file and place it in `backend/saved_models/` before launching the backend server.

---

## Local Setup & Installation

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** Version 3.10 to 3.12 (Tested on Python 3.12.x)
- **Node.js:** Version 18.x or 20.x (Tested on Node.js 20+)
- **Git**

---

### Backend Setup (FastAPI & PyTorch)

1. Open a terminal and navigate to the `backend` directory:
   ```powershell
   cd backend
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
     *(If script execution is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
   - **macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```

4. Install the required Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

5. Confirm the trained checkpoint `resnet50_brain_tumor.pth` is located at `backend/saved_models/resnet50_brain_tumor.pth`.

---

### Frontend Setup (Next.js & Tailwind CSS)

1. Open a **second terminal** and navigate to the `frontend` directory:
   ```powershell
   cd frontend
   ```

2. Install npm dependencies:
   ```powershell
   npm install
   ```

---

### Environment Variables

The frontend communicates with `http://127.0.0.1:8000` by default. If your backend is hosted on a different port or host, create a `.env.local` file inside `frontend/` (a template is available at `frontend/.env.example`):

```env
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

> **Security Notice:** Never commit secrets to version control. The `NEXT_PUBLIC_` prefix is bundled into client-side JavaScript and is strictly for non-sensitive public endpoints.

---

## Running the Application

### 1. Start the FastAPI Backend
From the activated `backend` terminal:
```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **API Root:** `http://127.0.0.1:8000`
- **Interactive Swagger Documentation:** `http://127.0.0.1:8000/docs`
- **Health Check Endpoint:** `http://127.0.0.1:8000/health`

### 2. Start the Next.js Frontend
From the `frontend` terminal:
```powershell
npm run dev
```
- **Web Application:** `http://localhost:3000`

---

## Application Usage & Walkthrough

1. **Access the Radiology Workstation:**
   Navigate to `http://localhost:3000` in your web browser. Verify the green API connectivity indicator in the top header (`API STATUS: ONLINE`).

2. **Select or Drag an MRI Scan:**
   Drag an axial brain MRI scan (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) onto the upload zone, or click to browse your local filesystem. An immediate client-side preview will display with scanline animations.

3. **Trigger Deep Learning Analysis:**
   Click **"Analyze Scan"**. The client uploads the image to `/api/v1/predict`, where it passes through the 4-layer validation pipeline before running ResNet-50 inference and Grad-CAM generation.

4. **Explore the Results Dashboard:**
   - **Overview Tab:** Review the primary classification verdict, confidence percentage, input scan slice, and multi-class distribution progress bars across all four categories.
   - **Explainability Tab:** Inspect the side-by-side comparison between the raw scan and the Grad-CAM heatmap overlay to understand which anatomical regions guided the prediction.
   - **Report Tab:** Review the diagnostic metrics table and click **"Download PDF Report"** to save the formatted ReportLab clinical research report directly to your downloads folder.

5. **Start a New Analysis:**
   Click the **"New Analysis"** button in the dashboard top bar to return to the ingestion screen for subsequent MRI scans.

---

## API Endpoints & Documentation

| Method | Endpoint | Description | Request Format | Success Response |
|---|---|---|---|---|
| `GET` | `/` | API Root Information | None | `{"message": "Brain Tumor Detection API is active", ...}` |
| `GET` | `/health` | Server Health Status | None | `{"status": "healthy", "service": "..."}` |
| `POST` | `/api/v1/predict` | Full MRI classification with Grad-CAM & scores | `multipart/form-data`<br>`file`: Image<br>`include_gradcam`: bool (default: true) | `PredictionResponse` JSON |
| `POST` | `/api/v1/gradcam` | Dedicated Grad-CAM heatmap overlay | `multipart/form-data`<br>`file`: Image | `{"status": "success", "gradcam_image": "data:image/jpeg;base64,..."}` |
| `POST` | `/api/v1/report` | Downloadable ReportLab PDF Research Report | `multipart/form-data`<br>`file`: Image | Binary `application/pdf` with attachment header |
| `POST` | `/api/v1/generate-report`| Alias for PDF research report generation | `multipart/form-data`<br>`file`: Image | Binary `application/pdf` with attachment header |

### Sample Prediction Response (`POST /api/v1/predict` — `200 OK`)

```json
{
  "status": "success",
  "predicted_class": "glioma",
  "confidence": 0.9214,
  "class_index": 0,
  "probabilities": {
    "glioma": 0.9214,
    "meningioma": 0.0574,
    "pituitary": 0.0117,
    "notumor": 0.0095
  },
  "class_scores": [
    {
      "class_name": "glioma",
      "display_name": "Glioma",
      "score": 0.9214,
      "percentage": 92.14,
      "is_highest": true
    },
    {
      "class_name": "meningioma",
      "display_name": "Meningioma",
      "score": 0.0574,
      "percentage": 5.74,
      "is_highest": false
    },
    {
      "class_name": "pituitary",
      "display_name": "Pituitary Tumor",
      "score": 0.0117,
      "percentage": 1.17,
      "is_highest": false
    },
    {
      "class_name": "notumor",
      "display_name": "No Tumor Detected",
      "score": 0.0095,
      "percentage": 0.95,
      "is_highest": false
    }
  ],
  "gradcam_image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "prediction_id": "PRED-8A7B2C10",
  "timestamp": "2026-10-23T16:20:00.123456+00:00"
}
```

### Error Responses
- **`400 Bad Request`:** Returned when the uploaded file fails any validation check (e.g., non-MRI photo, missing dark scanner perimeter, excessive color saturation, corrupt byte stream, empty upload, or file size $> 10$ MB).
- **`422 Unprocessable Entity`:** Returned if the required `file` multipart field is missing from the request.
- **`500 Internal Server Error`:** Returned if an unexpected runtime failure occurs during tensor inference or PDF generation.

---

## Confidence Scores Analysis & Limitations

### Understanding Softmax Confidence
The confidence metric returned by the model represents the highest normalized probability value computed across the four classes by the Softmax activation function. While useful for ranking diagnostic candidates:
1. **Mathematical Representation:** Softmax exponentiation forces probabilities across all classes to sum strictly to $1.0$. It reflects relative activation strength rather than an absolute biological probability of malignancy.
2. **Elevated Uncertainty Warning (<70%):** When model confidence falls below $70.0\%$, the user interface surfaces an amber advisory alerting the user to heightened radiological ambiguity.
3. **Out-of-Distribution Vulnerability:** Standard neural networks can assign high confidence to out-of-distribution inputs that visually resemble the training set. NeuroScan AI mitigates this through its 4-layer validation engine, but users must recognize that model confidence should never replace histopathological tissue biopsy or specialist neuro-radiological review.

---

## Verification & Testing Suites

The repository includes a comprehensive set of automated unit, integration, and evaluation suites located in `backend/`:

### 1. New Features Test Suite (`test_features.py`)
Validates the newly implemented capabilities including class scores canonical ordering, Grad-CAM generation and hook cleanup, in-memory PDF report generation, and router endpoints:
```powershell
python backend/test_features.py
```
*Result: 8 tests passing.*

### 2. Multi-Stage MRI Validation & OOD Safeguards Suite (`test_validation.py`)
Runs 23 automated tests across 7 rigorous input categories to verify that non-MRI inputs are rejected with HTTP 400:
- Valid MRI scans across all 4 classes (Glioma, Meningioma, Pituitary, No Tumor).
- Random natural color and grayscale landscape photographs.
- White document screenshots and dark-mode IDE screenshots.
- Corrupted byte streams and truncated JPEG files.
- Empty 0-byte uploads and disallowed extensions (`.txt`, `.pdf`).
- Oversized files (>10 MB), tiny icons ($16 \times 16$), extreme aspect ratios, solid black/white images, and full-frame noise.
- Non-brain synthetic medical-like images and inverted scans.
```powershell
python backend/test_validation.py
```
*Result: 23 / 23 tests passing (100.0%).*

### 3. FastAPI Endpoint Verification (`verify_api.py`)
Tests health endpoints, valid image classification, extension rejection, and corrupt file detection:
```powershell
python backend/verify_api.py
```

### 4. Single-Image Inference Smoke Test (`verify_inference.py`)
Performs a smoke test on checkpoint loading, tensor shapes, and single-image prediction:
```powershell
python backend/verify_inference.py
```

### 5. Benchmark Test Dataset Evaluation (`evaluate.py`)
Executes full PyTorch DataLoader batch evaluation across the 1,600-image test set:
```powershell
python backend/evaluate.py --data-dir "D:\SEM 5 all projects\Brain Tumor Dataset\Testing"
```

### 6. Per-Image CSV Record Generation (`evaluate_model.py`)
Generates a comprehensive per-image classification report saved to `backend/test_results.csv`:
```powershell
python backend/evaluate_model.py
```

---

## Known Limitations & Future Scope

### Current Limitations
1. **2D Axial Slices:** The current pipeline evaluates single 2D axial MRI slices; multi-planar volumetric sequences (coronal, sagittal, 3D volumetric T1-CE) are analyzed independently per image rather than as a unified 3D volume.
2. **Standardized Contrast Sensitivity:** Model performance relies on standardized MRI acquisition contrast; non-standard acquisition protocols or heavy compression artifacts may degrade classification certainty.
3. **Heatmap Resolution vs. Pixel Segmentation:** While Grad-CAM provides visual saliency heatmaps by interpolating from the $7 \times 7$ activation maps of `layer4[-1]`, it does not generate pixel-level semantic segmentation masks or explicit tumor boundary contours.

### Future Scope (Proposed Enhancements)
- **3D Volumetric Segmentation (nnU-Net):** Integrating a 3D U-Net backbone for volumetric tumor segmentation, edema delineation, and necrotic core volumetry.
- **DICOM (.dcm) Medical Standard Ingestion:** Direct support for native clinical DICOM files including metadata extraction (repetition time $TR$, echo time $TE$, magnetic field strength).
- **Multi-Sequence MRI Fusion:** Multi-modal feature fusion accepting co-registered T1, T1-contrast enhanced (T1-CE), T2, and FLAIR sequences simultaneously.
- **Containerized Cloud Deployment:** Multi-stage `Dockerfile` and `docker-compose.yml` configurations for deployment to Kubernetes or cloud providers.

---

## Medical & Academic Disclaimer

> [!CAUTION]
> **IMPORTANT REGULATORY AND CLINICAL DISCLAIMER:**
>
> NeuroScan AI is an **academic research and educational software project** developed for computer science coursework and portfolio demonstration.
>
> - **Not a Medical Device:** This software is **NOT** cleared, approved, or evaluated by the United States Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any equivalent national or international medical regulatory body.
> - **No Clinical Diagnostic Utility:** Predictions, probability scores, Grad-CAM heatmaps, and downloadable PDF reports produced by this software do **NOT** constitute medical advice, a clinical diagnosis, surgical planning guidance, or patient prognosis.
> - **Mandatory Professional Consultation:** Never make clinical decisions, alter medical treatment, or delay seeking professional medical advice based on outputs from this application. All brain MRI evaluations must be performed by a board-certified radiologist, neuro-oncologist, or licensed physician.

---

## License

This project is licensed under the [MIT License](LICENSE) — free for educational, academic, and research use.
