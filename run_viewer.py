# run_viewer.py — launcher that extends your GUI.py viewer with AI tabs
# ROI-aware organ segmentation

from organ_segmentor import OrganSegmentor
from orientation_detector import OrientationDetector
import GUI as _orig_gui
import os
import sys
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
    QProgressBar, QTabWidget, QTextEdit, QFileDialog, QApplication, QMessageBox
)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

os.environ.setdefault("TORCH_HOME", os.path.join(HERE, "torch_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# -------------------- Background thread for organ segmentation --------------------
class SegmentationThreadROI(QThread):
    progress = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, segmentor, nifti_path, roi_bounds=None):
        super().__init__()
        self.segmentor = segmentor
        self.nifti_path = nifti_path
        self.roi_bounds = roi_bounds

    def run(self):
        try:
            def cb(msg):
                self.progress.emit(msg)

            # Segment full volume (not just ROI)
            _mask, _is_new = self.segmentor.segment_nifti(
                self.nifti_path,
                use_cache=True,
                fast=True,
                callback=cb
            )
            self.finished.emit(True, "Success")
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.finished.emit(False, str(e))


# ------------------------------ Safe extension helpers ------------------------------
def _ensure_ai_fields(self):
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"

    if not hasattr(self, 'orientation_detector'):
        try:
            self.orientation_detector = OrientationDetector(device=dev)
        except:
            self.orientation_detector = OrientationDetector()

    if not hasattr(self, 'organ_segmentor'):
        self.organ_segmentor = OrganSegmentor()

    if not hasattr(self, 'detected_orientation'):
        self.detected_orientation = None
        self.detection_confidence = 0.0
        self.current_nifti_path = None
        self.segmentation_thread = None


# ------------------------------ Get viewer class from GUI.py ------------------------------
try:
    Viewer = _orig_gui.DICOMMultiPlanarViewer
except AttributeError:
    Viewer = None
    for _name in dir(_orig_gui):
        obj = getattr(_orig_gui, _name)
        if isinstance(obj, type) and all(hasattr(obj, m) for m in ("create_control_panel", "load_nifti_file")):
            Viewer = obj
            break
    if Viewer is None:
        raise ImportError("Could not locate viewer class in GUI.py")


# ---------------------------------- ORIENTATION TAB ----------------------------------
def create_orientation_tab(self):
    _ensure_ai_fields(self)
    tab = QWidget()
    lay = QVBoxLayout(tab)

    lay.addWidget(QLabel('<b>AI Orientation Detection</b>'))

    self.load_model_btn = QPushButton('Load AI Model (.pth)')
    self.load_model_btn.clicked.connect(self.load_orientation_model)
    lay.addWidget(self.load_model_btn)

    self.model_status_label = QLabel('Model not loaded')
    self.model_status_label.setStyleSheet(
        "background:#ffeb3b;color:#000;padding:6px;border-radius:4px;font-weight:bold;")
    lay.addWidget(self.model_status_label)

    self.detect_btn = QPushButton('Detect Orientation Now')
    self.detect_btn.setEnabled(False)
    self.detect_btn.clicked.connect(self.detect_orientation)
    lay.addWidget(self.detect_btn)

    self.orientation_result_label = QLabel('Orientation: Not detected')
    self.orientation_result_label.setWordWrap(True)
    self.orientation_result_label.setStyleSheet(
        "background:#f5f5f5;padding:10px;border-radius:4px;border:2px solid #ddd;")
    lay.addWidget(self.orientation_result_label)

    lay.addStretch()
    return tab


# ----------------------------------- ORGANS TAB -----------------------------------
def create_organ_segmentation_tab(self):
    _ensure_ai_fields(self)
    tab = QWidget()
    lay = QVBoxLayout(tab)

    lay.addWidget(QLabel('<b>Organ Segmentation</b>'))

    # ROI-only checkbox
    self.roi_only_checkbox = QCheckBox(
        '🎯 Segment ROI Only (display organs in ROI)')
    self.roi_only_checkbox.setChecked(False)
    self.roi_only_checkbox.setToolTip(
        'If checked, only displays organs within the selected ROI bounds')
    lay.addWidget(self.roi_only_checkbox)

    self.roi_status_label = QLabel('ROI: Not selected')
    self.roi_status_label.setStyleSheet(
        "background:#fff3cd;padding:6px;border-radius:3px;font-size:9pt;")
    lay.addWidget(self.roi_status_label)

    self.segment_btn = QPushButton('Run Segmentation')
    self.segment_btn.setEnabled(False)
    self.segment_btn.clicked.connect(self.run_segmentation)
    lay.addWidget(self.segment_btn)

    self.seg_progress = QProgressBar()
    self.seg_progress.setVisible(False)
    lay.addWidget(self.seg_progress)

    self.seg_status_label = QLabel('Status: Not started')
    self.seg_status_label.setStyleSheet(
        "background:#f5f5f5;padding:8px;border-radius:4px;")
    lay.addWidget(self.seg_status_label)

    lay.addWidget(QLabel('<b>Detected Organs</b>'))
    self.organ_list = QTextEdit()
    self.organ_list.setReadOnly(True)
    self.organ_list.setMaximumHeight(200)
    lay.addWidget(self.organ_list)

    lay.addWidget(QLabel('<b>Current Slice Organs</b>'))
    self.slice_organs_label = QLabel('Move sliders to see organs')
    self.slice_organs_label.setWordWrap(True)
    self.slice_organs_label.setStyleSheet(
        "background:#e8f5e9;padding:8px;border-radius:4px;")
    lay.addWidget(self.slice_organs_label)

    lay.addStretch()
    return tab


# ---------------------------------- ORIENTATION ACTIONS ----------------------------------
def load_orientation_model(self):
    _ensure_ai_fields(self)
    pth, _ = QFileDialog.getOpenFileName(
        self, "Select Orientation Model", HERE, "PyTorch Model (*.pth);;All files (*)"
    )
    if not pth:
        return

    ok = self.orientation_detector.load_model(pth)
    if ok:
        self.model_status_label.setText('Model loaded')
        self.model_status_label.setStyleSheet(
            "background:#4caf50;color:#fff;padding:6px;border-radius:4px;font-weight:bold;")
        self.detect_btn.setEnabled(self.dicom_data is not None)
    else:
        self.model_status_label.setText('Failed to load model')
        self.model_status_label.setStyleSheet(
            "background:#f44336;color:#fff;padding:6px;border-radius:4px;font-weight:bold;")


def detect_orientation(self):
    _ensure_ai_fields(self)
    if getattr(self, 'dicom_data', None) is None:
        QMessageBox.warning(self, "Error", "Load data first")
        return

    if not self.orientation_detector.is_loaded():
        QMessageBox.warning(self, "Error", "Load AI model first")
        return

    try:
        orient, conf, probs = self.orientation_detector.predict(
            self.dicom_data)
        if orient is None:
            QMessageBox.warning(self, "Error", "Detection failed")
            return

        self.detected_orientation = orient
        self.detection_confidence = conf
        self.orientation_result_label.setText(
            f"<b>Detected Orientation:</b> {orient.upper()}<br><b>Confidence:</b> {conf*100:.1f}%")
        self.status_bar.showMessage(
            f"Detected: {orient.upper()} ({conf*100:.1f}%)")
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Detection error: {e}")


# ---------------------------------- SEGMENTATION ACTIONS ----------------------------------
def run_segmentation(self):
    _ensure_ai_fields(self)

    if self.current_nifti_path is None:
        QMessageBox.warning(self, "NIfTI Required",
                            "Please load a .nii or .nii.gz file first.")
        return

    if self.segmentation_thread and self.segmentation_thread.isRunning():
        QMessageBox.warning(self, "Already Running",
                            "Segmentation already running")
        return

    # Check if ROI-only mode is enabled
    roi_enabled = self.roi_only_checkbox.isChecked()

    if roi_enabled:
        # Check if ROI is defined
        if not hasattr(self, 'roi_slices') or self.roi_slices is None:
            QMessageBox.warning(
                self, "ROI Required",
                "ROI-only mode is enabled but no ROI is selected.\n\n"
                "Please either:\n"
                "1. Select an ROI first (Data & ROI tab), or\n"
                "2. Uncheck 'Segment ROI Only'"
            )
            return

        # Get ROI bounds
        roi_bounds = {
            'axial': self.roi_slices.get('axial', (0, self.dicom_data.shape[0]-1)),
            'sagittal': self.roi_slices.get('sagittal', (0, self.dicom_data.shape[2]-1)),
            'coronal': self.roi_slices.get('coronal', (0, self.dicom_data.shape[1]-1))
        }
    else:
        roi_bounds = None

    self.segment_btn.setEnabled(False)
    self.seg_progress.setVisible(True)
    self.seg_progress.setRange(0, 0)
    self.seg_status_label.setText('Status: Running segmentation...')

    # Create thread
    self.segmentation_thread = SegmentationThreadROI(
        self.organ_segmentor,
        self.current_nifti_path,
        roi_bounds  # Pass ROI bounds
    )

    self.segmentation_thread.progress.connect(
        lambda m: self.seg_status_label.setText(f"Status: {m}")
    )

    def _done(ok, msg):
        self.seg_progress.setVisible(False)
        self.segment_btn.setEnabled(True)

        if ok:
            self.seg_status_label.setText('Status: Segmentation complete!')
            labels = self.organ_segmentor.get_organ_labels() or {}

            if labels:
                lines = ["Detected Organs (Full Volume):", ""]
                for _, name in sorted(labels.items()):
                    lines.append(f"• {name.replace('_', ' ').title()}")
                self.organ_list.setText("\n".join(lines))

            # Store ROI bounds for slice updates
            self._current_roi_bounds = roi_bounds

            self.status_bar.showMessage(
                f"Segmentation complete! Found {len(labels)} organs")
            self.update_all_views()
        else:
            self.seg_status_label.setText(f"Status: Error: {msg}")

    self.segmentation_thread.finished.connect(_done)
    self.segmentation_thread.start()


# ---------------------------------- Patch viewer (non-invasive) ----------------------------------
Viewer._orig_initialize_after_load = Viewer.initialize_after_load


def _init_after_load(self, *a, **k):
    res = Viewer._orig_initialize_after_load(self, *a, **k)
    _ensure_ai_fields(self)
    self._current_roi_bounds = None  # Initialize ROI bounds
    try:
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
    if not fp:
        return
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(fp)
        arr = sitk.GetArrayFromImage(img)
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

# Patch update_all_views to pass ROI bounds to organ display
Viewer._orig_update_all_views = Viewer.update_all_views


def _update_all_views_with_roi(self):
    res = Viewer._orig_update_all_views(self)

    # Update ROI status if checkbox exists
    if hasattr(self, 'roi_only_checkbox') and hasattr(self, 'roi_status_label'):
        if self.roi_only_checkbox.isChecked() and hasattr(self, 'roi_slices') and self.roi_slices:
            roi_text = (
                f"ROI: Z[{self.roi_slices['axial'][0]}-{self.roi_slices['axial'][1]}] "
                f"X[{self.roi_slices['sagittal'][0]}-{self.roi_slices['sagittal'][1]}] "
                f"Y[{self.roi_slices['coronal'][0]}-{self.roi_slices['coronal'][1]}]"
            )
            self.roi_status_label.setText(roi_text)
            self.roi_status_label.setStyleSheet(
                "background:#d4edda;padding:6px;border-radius:3px;color:#155724;font-size:9pt;")
        else:
            self.roi_status_label.setText("ROI: Not selected")
            self.roi_status_label.setStyleSheet(
                "background:#fff3cd;padding:6px;border-radius:3px;font-size:9pt;")

    return res


Viewer.update_all_views = _update_all_views_with_roi

# Add the new tab methods and action methods
Viewer.create_orientation_tab = create_orientation_tab
Viewer.create_organ_segmentation_tab = create_organ_segmentation_tab
Viewer.load_orientation_model = load_orientation_model
Viewer.detect_orientation = detect_orientation
Viewer.run_segmentation = run_segmentation

# Patch create_control_panel to add AI tabs
Viewer._orig_create_control_panel = Viewer.create_control_panel


def _create_control_panel_with_ai(self, *a, **k):
    panel = Viewer._orig_create_control_panel(self, *a, **k)
    _ensure_ai_fields(self)

    # Find the tab widget
    tabs = panel.findChild(QTabWidget)
    if tabs:
        tabs.addTab(self.create_orientation_tab(), "Orientation")
        tabs.addTab(self.create_organ_segmentation_tab(), "Organs")

    return panel


Viewer.create_control_panel = _create_control_panel_with_ai

# ---------------------------------- main ----------------------------------
if __name__ == '__main__':
    print(">>> Starting MPR Viewer with AI Integration")
    try:
        import torch
        print(
            f">>> TORCH: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    except Exception as e:
        print(f">>> TORCH IMPORT FAILED: {e}")

    app = _orig_gui.QApplication(sys.argv)
    viewer = Viewer()
    viewer.show()
    sys.exit(app.exec())
