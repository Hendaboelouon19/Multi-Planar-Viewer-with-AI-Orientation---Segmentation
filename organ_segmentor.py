"""
Organ Segmentation Module - with ROI filtering
Standalone file for TotalSegmentator integration

Save this as: organ_segmentor.py
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pickle
import tempfile
import os
import sys
import traceback


class OrganSegmentor:
    """
    Real-time organ segmentation using TotalSegmentator
    Supports filtering organs to display only those within ROI boundaries
    """

    def __init__(self, cache_dir="./segmentation_cache"):
        """
        Initialize organ segmentor
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.segmentation_mask = None
        self.organ_labels = None
        self.image_shape = None
        self.affine = None
        self.is_processing = False
        self.processing_complete = False

        # Set environment variables to prevent multiprocessing issues
        self._set_environment_vars()

        # Comprehensive organ name mapping
        self.organ_names = {
            1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
            5: "liver", 6: "stomach", 7: "aorta", 8: "inferior_vena_cava",
            9: "portal_vein_splenic_vein", 10: "pancreas", 11: "adrenal_gland_right",
            12: "adrenal_gland_left", 13: "lung_upper_lobe_left", 14: "lung_lower_lobe_left",
            15: "lung_upper_lobe_right", 16: "lung_middle_lobe_right", 17: "lung_lower_lobe_right",
            18: "vertebrae_L5", 19: "vertebrae_L4", 20: "vertebrae_L3", 21: "vertebrae_L2",
            22: "vertebrae_L1", 23: "vertebrae_T12", 24: "vertebrae_T11", 25: "vertebrae_T10",
            26: "vertebrae_T9", 27: "vertebrae_T8", 28: "vertebrae_T7", 29: "vertebrae_T6",
            30: "vertebrae_T5", 31: "vertebrae_T4", 32: "vertebrae_T3", 33: "vertebrae_T2",
            34: "vertebrae_T1", 35: "vertebrae_C7", 36: "vertebrae_C6", 37: "vertebrae_C5",
            38: "vertebrae_C4", 39: "vertebrae_C3", 40: "vertebrae_C2", 41: "vertebrae_C1",
            42: "heart", 43: "esophagus", 44: "trachea", 45: "thyroid_gland",
            46: "small_bowel", 47: "duodenum", 48: "colon", 49: "urinary_bladder",
            50: "prostate", 51: "kidney_cyst_left", 52: "kidney_cyst_right",
            90: "brain", 91: "cerebellum", 92: "brainstem", 93: "ventricles",
            94: "white_matter", 95: "gray_matter", 96: "corpus_callosum",
            97: "thalamus", 98: "hippocampus", 99: "amygdala"
        }

    def _set_environment_vars(self):
        """Set environment variables to prevent multiprocessing issues"""
        # Prevent multiprocessing issues on Windows
        if sys.platform == "win32":
            os.environ["NO_MP"] = "1"

        # Reduce memory usage
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # Disable multiprocessing for stability
        os.environ["NUMEXPR_MAX_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    def segment_nifti(self, nifti_path, use_cache=True, fast=True, callback=None):
        """
        Load NIfTI file and run TotalSegmentator
        """
        nifti_path = Path(nifti_path)
        cache_file = self.cache_dir / f"{nifti_path.stem}_segmentation.pkl"

        # Check cache first
        if use_cache and cache_file.exists():
            if callback:
                callback("Loading cached segmentation...")
            print("Loading cached segmentation...")

            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.segmentation_mask = cache_data['mask']
                    self.organ_labels = cache_data['labels']
                    self.image_shape = cache_data['shape']
                    self.affine = cache_data.get('affine', np.eye(4))

                nii = nib.load(str(nifti_path))
                image_data = nii.get_fdata()

                print(
                    f"✓ Loaded from cache: {len(self.organ_labels)} organs found")
                self.processing_complete = True

                if callback:
                    callback(f"Cache loaded: {len(self.organ_labels)} organs")

                return image_data, False

            except Exception as e:
                print(
                    f"⚠ Cache load failed: {e}, running fresh segmentation...")
                if callback:
                    callback("Cache failed, running fresh segmentation...")

        # Load image
        if callback:
            callback("Loading NIfTI file...")
        print("Loading NIfTI file...")

        try:
            nii = nib.load(str(nifti_path))
            image_data = nii.get_fdata()
            self.image_shape = image_data.shape
            self.affine = nii.affine
        except Exception as e:
            print(f"✗ Error loading NIfTI file: {e}")
            if callback:
                callback(f"Error loading file: {e}")
            return None, False

        print(f"Image shape: {image_data.shape}")

        if callback:
            callback("Running TotalSegmentator AI...")
        print("Running TotalSegmentator AI...")

        # Run TotalSegmentator with error handling
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "segmentation.nii"

            try:
                self.is_processing = True

                if callback:
                    callback(
                        "AI model processing (this may take several minutes)...")

                # Import and run TotalSegmentator
                from totalsegmentator.python_api import totalsegmentator

                # Run TotalSegmentator with proper parameters
                totalsegmentator(
                    input=str(nifti_path),
                    output=str(output_path),
                    fast=fast,
                    ml=True,
                    preview=False,
                    statistics=False,
                    radiomics=False,
                    multiprocessing="disabled",
                    verbose=False,
                    force_split=False
                )

                # Load segmentation result
                if output_path.exists():
                    seg_nii = nib.load(str(output_path))
                    self.segmentation_mask = seg_nii.get_fdata().astype(np.uint8)

                    # Get unique labels
                    unique_labels = np.unique(self.segmentation_mask)
                    self.organ_labels = {
                        label: self.organ_names.get(
                            label, f"structure_{label}")
                        for label in unique_labels if label > 0
                    }

                    print(
                        f"✓ TotalSegmentator complete! Found {len(self.organ_labels)} organs")

                    # Cache results
                    cache_data = {
                        'mask': self.segmentation_mask,
                        'labels': self.organ_labels,
                        'shape': self.image_shape,
                        'affine': self.affine
                    }
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_data, f)

                    print(f"Cache saved: {cache_file}")

                    self.processing_complete = True
                    self.is_processing = False

                    if callback:
                        callback(
                            f"AI Complete! Found {len(self.organ_labels)} organs")

                else:
                    raise Exception(
                        "TotalSegmentator failed - no output file created")

            except Exception as e:
                print(f"✗ Error during TotalSegmentator: {e}")
                traceback.print_exc()

                # Create empty mask as fallback
                self.segmentation_mask = np.zeros_like(
                    image_data, dtype=np.uint8)
                self.organ_labels = {}
                self.is_processing = False
                self.processing_complete = True

                if callback:
                    callback(f"AI Error: {str(e)}")

                return image_data, False

        return image_data, True

    def get_organs_in_slice(self, slice_index, axis=0, threshold=0.01, roi_bounds=None):
        """
        Get list of organs present in a specific slice, respecting ROI boundaries
        """
        if self.segmentation_mask is None:
            return []

        # Define ROI bounds (use full volume if no ROI provided)
        if roi_bounds is None:
            roi_bounds = {
                'axial': (0, self.image_shape[0]-1),
                'coronal': (0, self.image_shape[1]-1),
                'sagittal': (0, self.image_shape[2]-1)
            }

        # Check if slice is within ROI bounds
        if axis == 0:  # Axial (Z-axis)
            z_start, z_end = roi_bounds.get(
                'axial', (0, self.image_shape[0]-1))
            if not (z_start <= slice_index <= z_end):
                return []  # Outside ROI
        elif axis == 1:  # Coronal (Y-axis)
            y_start, y_end = roi_bounds.get(
                'coronal', (0, self.image_shape[1]-1))
            if not (y_start <= slice_index <= y_end):
                return []  # Outside ROI
        elif axis == 2:  # Sagittal (X-axis)
            x_start, x_end = roi_bounds.get(
                'sagittal', (0, self.image_shape[2]-1))
            if not (x_start <= slice_index <= x_end):
                return []  # Outside ROI

        # Extract the slice based on axis
        try:
            if axis == 0:  # Axial (Z-axis)
                # Get the full slice first
                full_slice = self.segmentation_mask[slice_index, :, :]

                # Apply ROI mask in 2D (coronal and sagittal planes)
                y_start, y_end = roi_bounds.get(
                    'coronal', (0, self.image_shape[1]-1))
                x_start, x_end = roi_bounds.get(
                    'sagittal', (0, self.image_shape[2]-1))

                # Create a mask for the ROI area
                roi_mask = np.zeros_like(full_slice, dtype=bool)
                roi_mask[y_start:y_end+1, x_start:x_end+1] = True

                # Apply the ROI mask
                slice_mask = np.where(roi_mask, full_slice, 0)

            elif axis == 1:  # Coronal (Y-axis)
                # Get the full slice first
                full_slice = self.segmentation_mask[:, slice_index, :]

                # Apply ROI mask in 2D (axial and sagittal planes)
                z_start, z_end = roi_bounds.get(
                    'axial', (0, self.image_shape[0]-1))
                x_start, x_end = roi_bounds.get(
                    'sagittal', (0, self.image_shape[2]-1))

                # Create a mask for the ROI area
                roi_mask = np.zeros_like(full_slice, dtype=bool)
                roi_mask[z_start:z_end+1, x_start:x_end+1] = True

                # Apply the ROI mask
                slice_mask = np.where(roi_mask, full_slice, 0)

            elif axis == 2:  # Sagittal (X-axis)
                # Get the full slice first
                full_slice = self.segmentation_mask[:, :, slice_index]

                # Apply ROI mask in 2D (axial and coronal planes)
                z_start, z_end = roi_bounds.get(
                    'axial', (0, self.image_shape[0]-1))
                y_start, y_end = roi_bounds.get(
                    'coronal', (0, self.image_shape[1]-1))

                # Create a mask for the ROI area
                roi_mask = np.zeros_like(full_slice, dtype=bool)
                roi_mask[z_start:z_end+1, y_start:y_end+1] = True

                # Apply the ROI mask
                slice_mask = np.where(roi_mask, full_slice, 0)
            else:
                return []
        except IndexError:
            return []

        # Find unique organs in this ROI-filtered slice
        unique_labels, counts = np.unique(slice_mask, return_counts=True)

        # Calculate total pixels in the ROI area only (not the entire slice)
        if axis == 0:
            y_start, y_end = roi_bounds.get(
                'coronal', (0, self.image_shape[1]-1))
            x_start, x_end = roi_bounds.get(
                'sagittal', (0, self.image_shape[2]-1))
            total_pixels = (y_end - y_start + 1) * (x_end - x_start + 1)
        elif axis == 1:
            z_start, z_end = roi_bounds.get(
                'axial', (0, self.image_shape[0]-1))
            x_start, x_end = roi_bounds.get(
                'sagittal', (0, self.image_shape[2]-1))
            total_pixels = (z_end - z_start + 1) * (x_end - x_start + 1)
        else:  # axis == 2
            z_start, z_end = roi_bounds.get(
                'axial', (0, self.image_shape[0]-1))
            y_start, y_end = roi_bounds.get(
                'coronal', (0, self.image_shape[1]-1))
            total_pixels = (z_end - z_start + 1) * (y_end - y_start + 1)

        organs_present = []

        for label, count in zip(unique_labels, counts):
            if label == 0:  # Skip background
                continue

            percentage = count / total_pixels
            if percentage >= threshold:
                organ_name = self.organ_labels.get(label, f"structure_{label}")
                organ_display = organ_name.replace('_', ' ').title()
                organs_present.append(organ_display)

        return sorted(organs_present)

    def get_mask(self):
        """Get the segmentation mask"""
        return self.segmentation_mask

    def get_organ_labels(self):
        """Get dictionary of detected organ labels"""
        return self.organ_labels

    def is_ready(self):
        """Check if segmentation is complete"""
        return self.processing_complete

    def clear_cache(self):
        """Clear all cached segmentation files"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
            print(f"✓ Cache cleared: {self.cache_dir}")


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Organ Segmentor - TotalSegmentator Test")
    print("=" * 60)

    segmentor = OrganSegmentor(cache_dir="./test_cache")

    if len(sys.argv) > 1:
        nifti_path = sys.argv[1]
        print(f"Testing TotalSegmentator with: {nifti_path}")

        def progress_callback(message):
            print(f"  → {message}")

        try:
            image_data, is_new = segmentor.segment_nifti(
                nifti_path,
                use_cache=True,
                callback=progress_callback
            )

            print(
                f"✓ TotalSegmentator Complete! Found {len(segmentor.get_organ_labels())} organs")

            # Test ROI filtering
            mid_slice = image_data.shape[0] // 2
            organs_full = segmentor.get_organs_in_slice(mid_slice, axis=0)
            print(f"Full volume: {organs_full}")

            # Test with ROI
            roi_bounds = {
                'axial': (mid_slice-5, mid_slice+5),
                'sagittal': (0, image_data.shape[2]//2),
                'coronal': (0, image_data.shape[1]//2)
            }
            organs_roi = segmentor.get_organs_in_slice(
                mid_slice, axis=0, roi_bounds=roi_bounds)
            print(f"ROI filtered: {organs_roi}")

        except Exception as e:
            print(f"✗ TotalSegmentator Error: {e}")
    else:
        print("No file provided - TotalSegmentator ready for use")

    print("=" * 60)
