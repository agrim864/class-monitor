# 🎓 Classroom Monitor

An AI-powered classroom monitoring and analytics system that processes video recordings to generate comprehensive reports on attendance, student activity, attention levels, seating patterns, and speech analysis.

## 📸 Overview

| Component | Description |
|-----------|-------------|
| **Backend** | FastAPI server orchestrating multiple detector pipelines |
| **Frontend** | Flutter web app with real-time dashboards and data exploration |
| **Detectors** | Face identity, attention/attendance, activity tracking, speech analysis |
| **Processing** | Runs at **1 FPS** for efficient GPU utilization on long classroom videos |

## 🏗️ Architecture

```
classroom-monitor/
├── app/                      # FastAPI backend
│   ├── api/main.py           # REST API endpoints
│   ├── models/               # Pose analysis, attention engine
│   └── utils/                # Helpers, env loading
├── configs/
│   └── config.yaml           # Central pipeline configuration
├── detectors/
│   ├── face_detector/        # Face identity & tracking (InsightFace + AdaFace)
│   ├── attention_detector/   # Pose-based attention + attendance monitoring
│   ├── activity_detector/    # Device use, hand-raise, note-taking detection
│   └── vlp_detector/         # Speech topic segmentation
├── frontend/                 # Flutter web application
│   └── lib/
│       ├── screens/          # Dashboard, Analytics, Embeddings, Data Explorer
│       ├── services/         # API client, data services
│       └── widgets/          # Reusable UI components
├── scripts/                  # Utilities (model download, augmentation, harvesting)
├── student-details/          # Student photo registry (Photo-*/ID-* per student)
├── third_party/              # TalkNet-ASD, VTP submodules
├── weights/                  # Model weights (not in git, see Setup)
├── media/                    # Input videos (not in git)
├── outputs/                  # Generated analysis results (not in git)
├── start_all.ps1             # ⚡ One-click launcher (backend + frontend)
├── start_backend.ps1         # Backend-only launcher
├── start_frontend.ps1        # Frontend-only launcher
└── run_all.py                # CLI pipeline runner (no frontend needed)
```

## 🚀 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11+ | Tested on 3.11, 3.12 |
| **CUDA** | 12.1+ | Required for GPU inference |
| **Flutter** | 3.x | For the web frontend |
| **Chrome** | Latest | Frontend runs in Chrome debug mode |
| **FFmpeg** | Any | Required for video processing |
| **Git** | 2.x | With submodule support |

## 📦 Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/agrim864/class-monitor.git
cd class-monitor
```

> If you already cloned without `--recurse-submodules`:
> ```bash
> git submodule update --init --recursive
> ```

### 2. Create Python virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA 12.1 is specified in requirements.txt. If you need a different CUDA version, install PyTorch separately first:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Download model weights

```bash
python scripts/download_models.py
```

This downloads ~3.2 GB of model weights into the `weights/` directory:

| Model | Size | Purpose |
|-------|------|---------|
| `yolov8x-worldv2.pt` | 140 MB | Open-vocabulary object detection |
| `yolo11x-pose.pt` | 113 MB | Pose estimation |
| `yolov8n.pt` | 6 MB | Lightweight detection |
| `yolo26s.pt` | 20 MB | Person detection backbone |
| `adaface/adaface_ir101.onnx*` | 249 MB | Face recognition embeddings |
| `talknet/pretrain_*.model` | 120 MB | Active speaker detection |
| `vtp/public_train_data/*.pth` | 2.5 GB | Visual speech processing |

> **Dry run** to see what would be downloaded:
> ```bash
> python scripts/download_models.py --dry-run
> ```

### 5. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your values. The most important variable:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | For augmentations | HuggingFace API token for AI face augmentations |
| `CHROME_PATH` | Optional | Path to Chrome executable |

### 6. Set up Flutter frontend

```bash
cd frontend
flutter pub get
cd ..
```

### 7. Add student photos

Place student reference photos in `student-details/` following this structure:

```
student-details/
├── John-12345/
│   ├── Photo-John-12345.jpg      # Primary reference photo
│   └── ID-John-12345.jpg         # ID card photo (optional)
├── Jane-67890/
│   ├── Photo-Jane-67890.jpg
│   └── ID-Jane-67890.jpg
└── ...
```

### 8. Add classroom videos

Place input videos in the `media/` directory:

```
media/
├── classroom1.mp4
├── classroom2.MOV
└── ...
```

## ▶️ Running the System

### Option A: One-click launch (recommended)

```powershell
.\start_all.ps1
```

This opens two PowerShell windows:
- **Backend**: FastAPI server on `http://localhost:8000`
- **Frontend**: Flutter web app on Chrome

### Option B: Run components separately

**Terminal 1 — Backend:**
```powershell
.\start_backend.ps1
```

**Terminal 2 — Frontend:**
```powershell
.\start_frontend.ps1
```

### Option C: CLI-only pipeline (no frontend)

```bash
python run_all.py --source media/classroom1.mp4 --sample-fps 1.0
```

Key CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | (required) | Input video file path |
| `--sample-fps` | `1.0` | Processing FPS (1.0 = 1 frame/sec) |
| `--skip-face` | false | Skip face identity detection |
| `--skip-speech` | false | Skip speech analysis |
| `--max-frames` | -1 | Limit frames for debugging |

## 🖥️ Frontend Features

| Screen | Navigation | Description |
|--------|-----------|-------------|
| **Dashboard** | Home icon | Subject overview, upload videos, monitor jobs |
| **Analytics** | Chart icon | Visual attendance & attention analytics |
| **Data Explorer** | Table icon | Browse all generated CSV reports |
| **Embeddings** | Face icon | Generate & manage face recognition embeddings |
| **Study Assistant** | Chat icon | AI-powered study helper |

## 📊 Generated Output Files

After processing a video, the following artifacts are generated in `outputs/<run_id>/`:

| File | Description |
|------|-------------|
| `face_identity.mp4` | Video with face identity overlays |
| `face_identity.csv` | Per-frame face detection log |
| `output.avi` | Attention/attendance annotated video |
| `attendance_report.csv` | Student attendance summary |
| `attendance_events_log.csv` | Entry/exit/late events |
| `seat_shifts_timeline.csv` | Seat occupancy changes |
| `activity_tracking.mp4` | Activity-annotated video |
| `person_activity_summary.csv` | Per-person activity breakdown |
| `speech_topic_segments.csv` | Speech topic timestamps |
| `final_student_summary.csv` | Comprehensive per-student report |

## ⚙️ Configuration

The main configuration file is `configs/config.yaml`. Key sections:

- **detection**: YOLO person detection thresholds and image sizes
- **object_detection**: Open-vocabulary object detection settings
- **pose**: Pose estimation parameters
- **tracking**: DeepSORT tracker settings
- **student_backbone**: Face detection and recognition pipeline
- **attention**: Attention scoring weights and thresholds
- **seating**: Seat calibration and assignment timing
- **attendance_events**: Late/early/absent thresholds

## 🔧 Generating Face Augmentations

To improve face recognition accuracy, generate augmented versions of student photos:

```bash
# Generate all augmentation types
python scripts/run_all_face_augmentations.py --continue-on-error

# Or from the Flutter frontend
# Navigate to the Embeddings screen and click "Generate All"
```

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

## 📝 API Documentation

Once the backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 License

This project is developed as an academic research tool.
