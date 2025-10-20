# -*- coding: utf-8 -*-
# run_viewer.py — launcher that extends your GUI.py viewer with AI tabs
# Works offline, supports CPU or GPU (auto-detect), and adds Orientation + Organs tabs.

import os, sys
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QProgressBar, QTabWidget, QTextEdit, QFileDialog, QApplication
)

# --- Resolve local imports reliably
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

# --- Block surprise downloads (keep models local)
os.environ.setdefault("TORCH_HOME", os.path.join(HERE, "torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --- Import your base GUI (CHANGED: was hoor_new; now GUI)
import GUI as _orig_gui

# --- Import AI modules (your files)
from orientation_detector import OrientationDetector
from organ_segmentor import OrganSegmentor


# -------------------- background thread for organ segmentation --------------------
class SegmentationThread(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, segmentor, nifti_path):
        super().__init__()
        self.segmentor = segmentor
        self.nifti_path = nifti_path

    def run(self):
        try:
            def cb(msg): self.progress.emit(msg)
            _mask, _is_new = self.segmentor.segment_nifti(
                self.nifti_path, use_cache=True, fast=True, callback=cb
            )
            self.finished.emit(True, "Success")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.finished.emit(False, str(e))


# ------------------------------ safe extension helpers ------------------------------
def _ensure_ai_fields(self):
    # Choose device: GPU if available, else CPU (orientation + organs use the same device)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"

    if not hasattr(self, 'orientation_detector'):
        # OrientationDetector should accept device="cpu"/"cuda"; if not, it still works
        self.orientation_detector = OrientationDetector(device=dev) if 'device' in OrientationDetector.__init__.__code__.co_varnames else OrientationDetector()
    if not hasattr(self, 'organ_segmentor'):
        self.organ_segmentor = OrganSegmentor()
    if not hasattr(self, 'detected_orientation'):
        self.detected_orientation = None
    if not hasattr(self, 'detection_confidence'):
        self.detection_confidence = 0.0
    if not hasattr(self, 'current_nifti_path'):
        self.current_nifti_path = None
    if not hasattr(self, 'segmentation_thread'):
        self.segmentation_thread = None


# ------------------------------ pick the viewer class from GUI.py ------------------------------
try:
    Viewer = _orig_gui.DICOMMultiPlanarViewer
except AttributeError:
    # fallback: find a class that looks like the viewer
    Viewer = None
    for _name in dir(_orig_gui):
        obj = getattr(_orig_gui, _name)
        if isinstance(obj, type) and all(hasattr(obj, m) for m in ("create_control_panel", "load_nifti_file", "initialize_after_load")):
            Viewer = obj
            break
    if Viewer is None:
        raise ImportError("Could not locate the viewer class in GUI.py (expecting DICOMMultiPlanarViewer or similar).")


# ---------------------------------- ORIENTATION TAB ----------------------------------
def create_orientation_tab(self):
    _ensure_ai_fields(self)
    tab = QWidget(); lay = QVBoxLayout(tab)

    lay.addWidget(QLabel('<b>AI Orientation Detection</b>'))

    self.load_model_btn = QPushButton('Load AI Model (.pth)')
    self.load_model_btn.clicked.connect(self.load_orientation_model)
    lay.addWidget(self.load_model_btn)

    self.model_status_label = QLabel('Model not loaded')
    self.model_status_label.setStyleSheet("background:#ffeb3b;color:#000;padding:6px;border-radius:4px;font-weight:bold;")
    lay.addWidget(self.model_status_label)

    self.auto_detect_checkbox = QCheckBox('Auto-detect on load')
    self.auto_detect_checkbox.setChecked(True)
    self.auto_detect_checkbox.setEnabled(False)
    lay.addWidget(self.auto_detect_checkbox)

    self.detect_btn = QPushButton('Detect Orientation Now')
    self.detect_btn.setEnabled(False)
    self.detect_btn.clicked.connect(self.detect_orientation)
    lay.addWidget(self.detect_btn)

    self.orientation_result_label = QLabel('Orientation: Not detected')
    self.orientation_result_label.setWordWrap(True)
    self.orientation_result_label.setStyleSheet("background:#f5f5f5;padding:10px;border-radius:4px;border:2px solid #ddd;")
    lay.addWidget(self.orientation_result_label)

    lay.addWidget(QLabel('<b>Confidence Scores</b>'))
    self.confidence_labels = {}
    for o in ['axial','sagittal','coronal']:
        row = QHBoxLayout()
        row.addWidget(QLabel(o.capitalize()+":"))
        lab = QLabel('0.0%'); lab.setMinimumWidth(60)
        self.confidence_labels[o] = lab
        row.addWidget(lab); lay.addLayout(row)

    lay.addStretch()
    return tab


# ----------------------------------- ORGANS TAB -----------------------------------
def create_organ_segmentation_tab(self):
    _ensure_ai_fields(self)
    tab = QWidget(); lay = QVBoxLayout(tab)

    lay.addWidget(QLabel('<b>Organ Segmentation</b>'))

    self.segment_btn = QPushButton('Run Segmentation')
    self.segment_btn.setEnabled(False)
    self.segment_btn.clicked.connect(self.run_segmentation)
    lay.addWidget(self.segment_btn)

    self.seg_progress = QProgressBar(); self.seg_progress.setVisible(False)
    lay.addWidget(self.seg_progress)

    self.seg_status_label = QLabel('Status: Not started')
    self.seg_status_label.setStyleSheet("background:#f5f5f5;padding:8px;border-radius:4px;")
    lay.addWidget(self.seg_status_label)

    lay.addWidget(QLabel('<b>Detected Organs</b>'))
    self.organ_list = QTextEdit(); self.organ_list.setReadOnly(True); self.organ_list.setMaximumHeight(200)
    lay.addWidget(self.organ_list)

    lay.addWidget(QLabel('<b>Current Slice Organs</b>'))
    self.slice_organs_label = QLabel('Move sliders to see organs')
    self.slice_organs_label.setWordWrap(True)
    self.slice_organs_label.setStyleSheet("background:#e8f5e9;padding:8px;border-radius:4px;")
    lay.addWidget(self.slice_organs_label)

    lay.addStretch()
    return tab


# ---------------------------------- ORIENTATION ACTIONS ----------------------------------
def load_orientation_model(self):
    _ensure_ai_fields(self)
    pth, _ = QFileDialog.getOpenFileName(
        self, "Select Orientation Model", HERE, "PyTorch Model (*.pth);;All files (*)"
    )
    if not pth: return
    ok = self.orientation_detector.load_model(pth)
    if ok:
        self.model_status_label.setText('Model loaded')
        self.model_status_label.setStyleSheet("background:#4caf50;color:#fff;padding:6px;border-radius:4px;font-weight:bold;")
        self.auto_detect_checkbox.setEnabled(True)
        self.detect_btn.setEnabled(self.dicom_data is not None)
        if hasattr(self, 'status_bar'): self.status_bar.showMessage("Orientation model loaded")
    else:
        self.model_status_label.setText('Failed to load model')
        self.model_status_label.setStyleSheet("background:#f44336;color:#fff;padding:6px;border-radius:4px;font-weight:bold;")

def detect_orientation(self):
    _ensure_ai_fields(self)
    if getattr(self, 'dicom_data', None) is None:
        if hasattr(self, 'status_bar'): self.status_bar.showMessage("Load data first"); return
    if not self.orientation_detector.is_loaded():
        if hasattr(self, 'status_bar'): self.status_bar.showMessage("Load AI model first"); return

    try:
        if hasattr(self, 'status_bar'): self.status_bar.showMessage("Detecting orientation...")
        QApplication.processEvents()
        orient, conf, probs = self.orientation_detector.predict(self.dicom_data)
        if orient is None:
            if hasattr(self, 'status_bar'): self.status_bar.showMessage("Detection failed"); return
        self.detected_orientation, self.detection_confidence = orient, conf
        self.orientation_result_label.setText(
            f"<b>Detected Orientation:</b> {orient.upper()}<br><b>Confidence:</b> {conf*100:.1f}%")
        for k,p in probs.items():
            lab = self.confidence_labels.get(k)
            if lab: lab.setText(f"{p*100:.1f}%"); lab.setStyleSheet("font-weight:bold;")
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Detected: {orient.upper()} ({conf*100:.1f}%)")
    except Exception as e:
        if hasattr(self, 'status_bar'): self.status_bar.showMessage(f"Detection error: {e}")


# ---------------------------------- SEGMENTATION ACTIONS ----------------------------------
def run_segmentation(self):
    _ensure_ai_fields(self)
    if self.current_nifti_path is None:
        _orig_gui.QMessageBox.warning(self, "NIfTI Required", "Please load a .nii or .nii.gz file first.")
        return
    if self.segmentation_thread and self.segmentation_thread.isRunning():
        if hasattr(self,'status_bar'): self.status_bar.showMessage("Segmentation already running"); return

    self.segment_btn.setEnabled(False)
    self.seg_progress.setVisible(True)
    self.seg_progress.setRange(0, 0)
    self.seg_status_label.setText('Status: Running...')

    self.segmentation_thread = SegmentationThread(self.organ_segmentor, self.current_nifti_path)
    self.segmentation_thread.progress.connect(lambda m: (
        self.seg_status_label.setText(f"Status: {m}"),
        hasattr(self, 'status_bar') and self.status_bar.showMessage(m)
    ))

    def _done(ok, msg):
        self.seg_progress.setVisible(False)
        self.segment_btn.setEnabled(True)
        if ok:
            self.seg_status_label.setText('Status: Segmentation complete!')
            labels = self.organ_segmentor.get_organ_labels() or {}
            if labels:
                lines = ["Detected Organs:", ""]
                for _, name in sorted(labels.items()):
                    lines.append(f"• {name.replace('_',' ').title()}")
                self.organ_list.setText("\n".join(lines))
            if hasattr(self.organ_segmentor, 'get_mask'):
                self.mask_data = self.organ_segmentor.get_mask()
            if hasattr(self, 'mask_checkbox'): self.mask_checkbox.setEnabled(True)
            if hasattr(self, 'status_bar'):
                self.status_bar.showMessage(f"Segmentation complete! Found {len(labels)} organs")
            self.update_all_views()
        else:
            self.seg_status_label.setText(f"Status: Error: {msg}")

    self.segmentation_thread.finished.connect(_done)
    self.segmentation_thread.start()


# ---------------------------------- patch viewer (non-invasive) ----------------------------------
Viewer._orig_initialize_after_load = Viewer.initialize_after_load
def _init_after_load(self, *a, **k):
    res = Viewer._orig_initialize_after_load(self, *a, **k)
    _ensure_ai_fields(self)
    try:
        if self.orientation_detector.is_loaded() and hasattr(self, 'detect_btn'):
            self.detect_btn.setEnabled(True)
        if self.current_nifti_path and hasattr(self, 'segment_btn'):
            self.segment_btn.setEnabled(True)
    except Exception:
        pass
    return res
Viewer.initialize_after_load = _init_after_load

Viewer._orig_load_nifti_file = Viewer.load_nifti_file
def _load_nifti_file_override(self):
    fp, _ = QFileDialog.getOpenFileName(
        self, "Open NIfTI File", HERE, "NIfTI files (*.nii *.nii.gz);;All files (*)"
    )
    if not fp: return
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(fp)
        arr = sitk.GetArrayFromImage(img)  # (z,y,x)
        self.dicom_data = arr.astype(_orig_gui.np.float32)
        sx, sy, sz = img.GetSpacing()
        self._voxel_sizes = (float(sz), float(sy), float(sx))
        _ensure_ai_fields(self)
        self.current_nifti_path = fp
        self.initialize_after_load()
    except Exception as e:
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Error loading NIfTI: {e}")
Viewer.load_nifti_file = _load_nifti_file_override

Viewer._orig_create_control_panel = Viewer.create_control_panel
def _create_control_panel_wrapper(self, *a, **k):
    panel = Viewer._orig_create_control_panel(self, *a, **k)
    _ensure_ai_fields(self)
    tabs = panel.findChild(QTabWidget)
    if tabs:
        tabs.addTab(self.create_orientation_tab(), "Orientation")
        tabs.addTab(self.create_organ_segmentation_tab(), "Organs")
    return panel
Viewer.create_control_panel = _create_control_panel_wrapper

# add the new tab methods
Viewer.create_orientation_tab = create_orientation_tab
Viewer.create_organ_segmentation_tab = create_organ_segmentation_tab
Viewer.load_orientation_model = load_orientation_model
Viewer.detect_orientation = detect_orientation
Viewer.run_segmentation = run_segmentation

# ---------------------------------- main ----------------------------------
if __name__ == '__main__':
    print(">>> USING PYTHON:", sys.executable)
    try:
        import torch; print(">>> TORCH:", torch.__version__, "| CUDA:", getattr(torch.cuda, "is_available", lambda: False)())
    except Exception as e:
        print(">>> TORCH IMPORT FAILED:", e)
    app = _orig_gui.QApplication(sys.argv)
    viewer = Viewer()
    viewer.show()
    sys.exit(app.exec())
