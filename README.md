 🩺 AI-Enhanced Multi-Planar DICOM Viewer

An interactive DICOM/NIfTI viewer that merges classic radiology visualization with **AI assistance**.

- 🧠 **AI orientation detection** (axial / coronal / sagittal)
- 🫀 **Multi-organ segmentation** (plug any model; ships with hooks)
- 🩻 **Multi-planar & oblique slicing** with synchronized views
- ⚙️ **GPU (CUDA 12.1) or CPU** execution
- 💾 **Caching & fast mode** for large volumes
- 🖥️ Built with **PySide6 + PyTorch + SimpleITK**

---

## ⚡ Quick Start

```bash
# 1) Create and activate a virtual environment (Windows)
python -m venv .venv310
.\.venv310\Scripts\activate

# 2) Install deps (CPU build by default)
pip install -r requirements.txt

# 3) Run the app
python run_viewer.py
Optional: enable GPU (faster)
bash
Copy code
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
🧰 Tech Stack
Category	Tools
GUI	PySide6 (Qt for Python)
AI	PyTorch, TorchVision
Medical Imaging	SimpleITK, Nibabel
Data Types	DICOM, NIfTI (.nii, .nii.gz)
Hardware	CPU / GPU (CUDA 12.1)

🧩 Project Structure
bash
Copy code
│
├── GUI.py                     # Base GUI (multi-planar & oblique viewer)   ← CHANGED
├── run_viewer.py              # Launcher that wires GUI + AI modules
├── orientation_detector.py    # Orientation AI (PyTorch)
├── organ_segmentor.py         # Organ segmentation pipeline (plug-in)
├── best_orientation_classifier.pth   # Example orientation model (load via UI)
├── README.md
└── requirements.txt
🎮 How to Use
1) Launch
bash
Copy code
python run_viewer.py
2) Load Data
DICOM: open a study/series from your local folder

NIfTI: .nii or .nii.gz

3) Orientation (AI)
Go to Orientation tab

Click Load AI Model → select best_orientation_classifier.pth

Click Detect Orientation Now

The viewer shows predicted plane + confidence & per-class scores

4) Organ Segmentation (AI)
Go to Organs tab

Click Run Segmentation

Detected organs appear in the list; per-slice labels show as you browse

Re-runs on the same file load from cache (instant)

Note: organ_segmentor.py is a drop-in module; you can plug in MONAI/TotalSegmentator or your own model by loading weights locally (no internet required).

🧪 Requirements
requirements.txt (typical):

nginx
Copy code
numpy
PySide6
SimpleITK
nibabel
torch
torchvision
torchaudio
matplotlib
scipy
For GPU: install the CUDA 12.1 torch wheel (see Quick Start).

📂 Example Datasets
TCIA – NSCLC-Radiomics: CT DICOM + DICOM-SEG/RTSTRUCT masks (lung, heart, esophagus, GTV)

MMWHS 2017: whole-heart segmentation (CT & MRI)

🧭 Roadmap
3D volume rendering & MIP

In-GUI mask overlay & color legend

Multi-model manager (swap checkpoints live)

DICOM-SEG export & JSON reports