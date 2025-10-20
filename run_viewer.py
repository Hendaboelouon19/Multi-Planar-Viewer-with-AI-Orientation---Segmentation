# run_viewer.py — launcher that extends your GUI.py viewer with AI tabs

from organ_segmentor import OrganSegmentor
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


# Background thread for organ segmentation
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

            # Run TotalSegmentator
            image_data, success = self.segmentor.segment_nifti(
                self.nifti_path,
                use_cache=True,
                fast=True,
                callback=cb
            )

            if success:
                self.finished.emit(True, "TotalSegmentator Successful")
            else:
                self.finished.emit(
                    False, "TotalSegmentator completed with warnings")

        except Exception as e:
            self.finished.emit(False, f"TotalSegmentator Error: {str(e)}")


# Safe extension helpers
def _ensure_ai_fields(self):
    if not hasattr(self, 'organ_segmentor'):
        self.organ_segmentor = OrganSegmentor()

    if not hasattr(self, 'current_nifti_path'):
        self.current_nifti_path = None
        self.segmentation_thread = None


# Get viewer class from GUI.py
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


# Organ Segmentation Tab
def create_organ_segmentation_tab(self):
    _ensure_ai_fields(self)
    tab = QWidget()
    lay = QVBoxLayout(tab)

    lay.addWidget(QLabel('<b>TotalSegmentator AI</b>'))

    info_label = QLabel(
        "Uses TotalSegmentator AI model to detect 104+ anatomical structures in CT/MRI scans.\n"
        "ROI filtering shows only organs within selected region."
    )
    info_label.setWordWrap(True)
    info_label.setStyleSheet(
        "background:#e3f2fd;padding:8px;border-radius:4px;color:#1565c0;")
    lay.addWidget(info_label)

    # ROI-only checkbox
    self.roi_only_checkbox = QCheckBox(
        '🎯 Show Only Organs in ROI')
    self.roi_only_checkbox.setChecked(False)
    self.roi_only_checkbox.setToolTip(
        'If checked, only displays organs within the selected ROI bounds')
    lay.addWidget(self.roi_only_checkbox)

    self.roi_status_label = QLabel('ROI: Not selected')
    self.roi_status_label.setStyleSheet(
        "background:#fff3cd;padding:6px;border-radius:3px;font-size:9pt;")
    lay.addWidget(self.roi_status_label)

    self.segment_btn = QPushButton('Run TotalSegmentator')
    self.segment_btn.setEnabled(False)
    self.segment_btn.clicked.connect(self.run_segmentation)
    lay.addWidget(self.segment_btn)

    self.seg_progress = QProgressBar()
    self.seg_progress.setVisible(False)
    lay.addWidget(self.seg_progress)

    self.seg_status_label = QLabel('Status: Load a NIfTI file to start')
    self.seg_status_label.setStyleSheet(
        "background:#f5f5f5;padding:8px;border-radius:4px;")
    lay.addWidget(self.seg_status_label)

    lay.addWidget(QLabel('<b>AI-Detected Organs</b>'))
    self.organ_list = QTextEdit()
    self.organ_list.setReadOnly(True)
    self.organ_list.setMaximumHeight(150)
    self.organ_list.setPlaceholderText(
        "Run TotalSegmentator to see detected organs...")
    lay.addWidget(self.organ_list)

    lay.addWidget(QLabel('<b>Current Slice Organs</b>'))
    self.slice_organs_label = QLabel(
        'Run TotalSegmentator and move sliders to see organs in current slice')
    self.slice_organs_label.setWordWrap(True)
    self.slice_organs_label.setStyleSheet(
        "background:#e8f5e9;padding:8px;border-radius:4px;")
    lay.addWidget(self.slice_organs_label)

    lay.addStretch()
    return tab


# Segmentation Actions
def run_segmentation(self):
    _ensure_ai_fields(self)

    if self.current_nifti_path is None:
        QMessageBox.warning(self, "NIfTI Required",
                            "Please load a .nii or .nii.gz file first.")
        return

    if self.segmentation_thread and self.segmentation_thread.isRunning():
        QMessageBox.warning(self, "Already Running",
                            "TotalSegmentator already running")
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
                "2. Uncheck 'Show Only Organs in ROI'"
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
    self.seg_status_label.setText('Status: TotalSegmentator processing...')

    # Create thread
    self.segmentation_thread = SegmentationThreadROI(
        self.organ_segmentor,
        self.current_nifti_path,
        roi_bounds
    )

    self.segmentation_thread.progress.connect(
        lambda m: self.seg_status_label.setText(f"Status: {m}")
    )

    def _done(ok, msg):
        self.seg_progress.setVisible(False)
        self.segment_btn.setEnabled(True)

        if ok:
            self.seg_status_label.setText('Status: TotalSegmentator complete!')
            labels = self.organ_segmentor.get_organ_labels() or {}

            if labels:
                lines = ["TotalSegmentator Detected:", ""]
                for _, name in sorted(labels.items()):
                    lines.append(f"• {name.replace('_', ' ').title()}")
                self.organ_list.setText("\n".join(lines))
            else:
                self.organ_list.setText(
                    "No organs detected by TotalSegmentator")

            # Store ROI bounds
            self._current_roi_bounds = roi_bounds

            # Apply ROI mask if needed
            if roi_bounds is not None:
                self._apply_roi_mask_permanently(roi_bounds)

            self.status_bar.showMessage(
                f"TotalSegmentator complete! Found {len(labels)} organs")
            self.update_all_views()
        else:
            self.seg_status_label.setText(f"Status: {msg}")
            QMessageBox.warning(self, "TotalSegmentator", msg)

    self.segmentation_thread.finished.connect(_done)
    self.segmentation_thread.start()


# ROI Masking
def _apply_roi_mask_permanently(self, roi_bounds):
    """
    Apply ROI bounds to the segmentation mask
    """
    mask = self.organ_segmentor.segmentation_mask

    if mask is None:
        return

    # Get ROI bounds
    z_start, z_end = roi_bounds.get('axial', (0, mask.shape[0]-1))
    y_start, y_end = roi_bounds.get('coronal', (0, mask.shape[1]-1))
    x_start, x_end = roi_bounds.get('sagittal', (0, mask.shape[2]-1))

    # Zero out everything OUTSIDE the ROI
    mask[:z_start, :, :] = 0
    mask[z_end+1:, :, :] = 0
    mask[:, :y_start, :] = 0
    mask[:, y_end+1:, :] = 0
    mask[:, :, :x_start] = 0
    mask[:, :, x_end+1:] = 0

    print(f"✓ ROI mask applied to TotalSegmentator output")


# Patch viewer methods
Viewer._orig_initialize_after_load = Viewer.initialize_after_load


def _init_after_load(self, *a, **k):
    res = Viewer._orig_initialize_after_load(self, *a, **k)
    _ensure_ai_fields(self)
    self._current_roi_bounds = None
    try:
        if self.current_nifti_path and hasattr(self, 'segment_btn'):
            self.segment_btn.setEnabled(True)
            self.seg_status_label.setText(
                'Status: Ready to run TotalSegmentator')
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

# Patch update_all_views
Viewer._orig_update_all_views = Viewer.update_all_views


def _update_all_views_optimized(self):
    res = Viewer._orig_update_all_views(self)

    # Update ROI status
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

    # Update current slice organs
    if (hasattr(self, 'organ_segmentor') and
        self.organ_segmentor.is_ready() and
            hasattr(self, 'slice_organs_label')):

        axial_idx = getattr(self, 'axial_slider', None)
        if axial_idx is not None:
            # Get ROI bounds if ROI-only mode is enabled
            roi_bounds = None
            if (hasattr(self, 'roi_only_checkbox') and
                self.roi_only_checkbox.isChecked() and
                hasattr(self, 'roi_slices') and
                    self.roi_slices):
                roi_bounds = self.roi_slices

            # Get organs in current axial slice using TotalSegmentator
            organs = self.organ_segmentor.get_organs_in_slice(
                axial_idx, axis=0, roi_bounds=roi_bounds
            )

            if organs:
                self.slice_organs_label.setText(
                    f"<b>Slice {axial_idx}:</b><br>" + ", ".join(organs))
                self.slice_organs_label.setStyleSheet(
                    "background:#e8f5e9;padding:8px;border-radius:4px;")
            else:
                self.slice_organs_label.setText(
                    f"<b>Slice {axial_idx}:</b><br>No organs detected in ROI")
                self.slice_organs_label.setStyleSheet(
                    "background:#ffebee;padding:8px;border-radius:4px;")

    return res


Viewer.update_all_views = _update_all_views_optimized

# Add the new methods
Viewer.create_organ_segmentation_tab = create_organ_segmentation_tab
Viewer.run_segmentation = run_segmentation
Viewer._apply_roi_mask_permanently = _apply_roi_mask_permanently

# Patch create_control_panel to add AI tab
Viewer._orig_create_control_panel = Viewer.create_control_panel


def _create_control_panel_with_ai(self, *a, **k):
    panel = Viewer._orig_create_control_panel(self, *a, **k)
    _ensure_ai_fields(self)

    # Find the tab widget
    tabs = panel.findChild(QTabWidget)
    if tabs:
        tabs.addTab(self.create_organ_segmentation_tab(), "TotalSegmentator")

    return panel


Viewer.create_control_panel = _create_control_panel_with_ai

# Main
if __name__ == '__main__':
    print(">>> Starting MPR Viewer with TotalSegmentator")
    app = _orig_gui.QApplication(sys.argv)
    viewer = Viewer()
    viewer.show()
    sys.exit(app.exec())
