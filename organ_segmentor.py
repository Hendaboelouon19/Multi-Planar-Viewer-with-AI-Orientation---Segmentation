"""
Organ Segmentation Module
Standalone file for TotalSegmentator integration

Save this as: organ_segmentor.py
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import pickle
import tempfile
from totalsegmentator.python_api import totalsegmentator


class OrganSegmentor:
    """
    Real-time organ segmentation using TotalSegmentator

    Usage:
        segmentor = OrganSegmentor()
        image_data, success = segmentor.segment_nifti('scan.nii.gz')
        organs = segmentor.get_organs_in_slice(slice_index, axis=0)
    """

    def __init__(self, cache_dir="./segmentation_cache"):
        """
        Initialize organ segmentor

        Args:
            cache_dir: Directory to cache segmentation results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.segmentation_mask = None
        self.organ_labels = None
        self.image_shape = None
        self.affine = None
        self.is_processing = False
        self.processing_complete = False

        # Comprehensive organ name mapping (50+ organs)
        self.organ_names = {
            # Major organs
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "aorta",
            8: "inferior_vena_cava",
            9: "portal_vein_splenic_vein",
            10: "pancreas",
            11: "adrenal_gland_right",
            12: "adrenal_gland_left",

            # Lungs
            13: "lung_upper_lobe_left",
            14: "lung_lower_lobe_left",
            15: "lung_upper_lobe_right",
            16: "lung_middle_lobe_right",
            17: "lung_lower_lobe_right",

            # Spine (vertebrae)
            18: "vertebrae_L5", 19: "vertebrae_L4", 20: "vertebrae_L3",
            21: "vertebrae_L2", 22: "vertebrae_L1", 23: "vertebrae_T12",
            24: "vertebrae_T11", 25: "vertebrae_T10", 26: "vertebrae_T9",
            27: "vertebrae_T8", 28: "vertebrae_T7", 29: "vertebrae_T6",
            30: "vertebrae_T5", 31: "vertebrae_T4", 32: "vertebrae_T3",
            33: "vertebrae_T2", 34: "vertebrae_T1", 35: "vertebrae_C7",
            36: "vertebrae_C6", 37: "vertebrae_C5", 38: "vertebrae_C4",
            39: "vertebrae_C3", 40: "vertebrae_C2", 41: "vertebrae_C1",

            # Other organs
            42: "heart",
            43: "esophagus",
            44: "trachea",
            45: "thyroid_gland",
            46: "small_bowel",
            47: "duodenum",
            48: "colon",
            49: "urinary_bladder",
            50: "prostate",
            51: "kidney_cyst_left",
            52: "kidney_cyst_right",

            # Brain structures
            90: "brain",
            91: "cerebellum",
            92: "brainstem",
            93: "ventricles",
            94: "white_matter",
            95: "gray_matter",
            96: "corpus_callosum",
            97: "thalamus",
            98: "hippocampus",
            99: "amygdala"
        }

    def segment_nifti(self, nifti_path, use_cache=True, fast=True, callback=None):
        """
        Load NIfTI file and run TotalSegmentator

        Args:
            nifti_path: Path to .nii or .nii.gz file
            use_cache: Use cached results if available (default True)
            fast: Use fast mode - less accurate but faster (default True)
            callback: Optional callback function(message) for progress updates

        Returns:
            tuple: (image_data: np.array, is_new: bool)
                   image_data is the loaded 3D volume
                   is_new is True if newly processed, False if from cache
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

                # Load original image
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

        nii = nib.load(str(nifti_path))
        image_data = nii.get_fdata()
        self.image_shape = image_data.shape
        self.affine = nii.affine

        print(f"Image shape: {image_data.shape}")

        if callback:
            callback(
                f"Running TotalSegmentator (shape: {image_data.shape})...")
        print("Running TotalSegmentator with GPU...")
        print("This may take 2-10 minutes depending on your GPU...")

        # Run TotalSegmentator
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "segmentation.nii"

            try:
                self.is_processing = True

                if callback:
                    callback(
                        "Segmenting organs (this may take several minutes)...")

                # Run segmentation
                totalsegmentator(
                    str(nifti_path),
                    str(output_path),
                    fast=fast,
                    ml=True
                )

                # Load segmentation result
                seg_nii = nib.load(str(output_path))
                self.segmentation_mask = seg_nii.get_fdata().astype(np.uint8)

                # Get unique labels
                unique_labels = np.unique(self.segmentation_mask)
                self.organ_labels = {
                    label: self.organ_names.get(label, f"structure_{label}")
                    for label in unique_labels if label > 0
                }

                print(
                    f"✓ Segmentation complete! Found {len(self.organ_labels)} organs")

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
                        f"Complete! Found {len(self.organ_labels)} organs")

            except Exception as e:
                print(f"✗ Error during segmentation: {e}")
                self.segmentation_mask = np.zeros_like(
                    image_data, dtype=np.uint8)
                self.organ_labels = {}
                self.is_processing = False
                self.processing_complete = True

                if callback:
                    callback(f"Error: {str(e)}")

                raise e

        return image_data, True

    def get_organs_in_slice(self, slice_index, axis=0, threshold=0.01):
        """
        Get list of organs present in a specific slice

        Args:
            slice_index: Slice index to check
            axis: Axis of the slice (0=axial, 1=coronal, 2=sagittal)
            threshold: Minimum percentage of slice that must contain organ (default 0.01 = 1%)

        Returns:
            list: List of organ names present in the slice (human-readable, sorted)
        """
        if self.segmentation_mask is None:
            return []

        # Extract the slice based on axis
        try:
            if axis == 0:  # Axial (Z-axis)
                slice_mask = self.segmentation_mask[slice_index, :, :]
            elif axis == 1:  # Coronal (Y-axis)
                slice_mask = self.segmentation_mask[:, slice_index, :]
            elif axis == 2:  # Sagittal (X-axis)
                slice_mask = self.segmentation_mask[:, :, slice_index]
            else:
                return []
        except IndexError:
            return []

        # Find unique organs in this slice
        unique_labels, counts = np.unique(slice_mask, return_counts=True)

        total_pixels = slice_mask.size
        organs_present = []

        for label, count in zip(unique_labels, counts):
            if label == 0:  # Skip background
                continue

            percentage = count / total_pixels
            if percentage >= threshold:
                organ_name = self.organ_labels.get(label, f"structure_{label}")
                # Convert to human-readable format
                organ_display = organ_name.replace('_', ' ').title()
                organs_present.append(organ_display)

        return sorted(organs_present)

    def get_mask(self):
        """
        Get the segmentation mask

        Returns:
            np.array: Segmentation mask (same shape as input volume)
        """
        return self.segmentation_mask

    def get_organ_labels(self):
        """
        Get dictionary of detected organ labels

        Returns:
            dict: {label_id: organ_name}
        """
        return self.organ_labels

    def is_ready(self):
        """
        Check if segmentation is complete and ready

        Returns:
            bool: True if segmentation complete, False otherwise
        """
        return self.processing_complete

    def clear_cache(self):
        """Clear all cached segmentation files"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
            print(f"✓ Cache cleared: {self.cache_dir}")


# ============================================================================
# STANDALONE TESTING
# ============================================================================
if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Organ Segmentor - Standalone Test")
    print("=" * 60)

    # Test 1: Initialize
    print("\n[Test 1] Initialize segmentor...")
    segmentor = OrganSegmentor(cache_dir="./test_cache")
    print(f"  Cache directory: {segmentor.cache_dir}")
    print(f"  Known organs: {len(segmentor.organ_names)}")

    # Test 2: Segment a file
    if len(sys.argv) > 1:
        nifti_path = sys.argv[1]
        print(f"\n[Test 2] Segmenting: {nifti_path}")

        def progress_callback(message):
            print(f"  → {message}")

        try:
            image_data, is_new = segmentor.segment_nifti(
                nifti_path,
                use_cache=True,
                fast=True,
                callback=progress_callback
            )

            print(f"\n  ✓ Segmentation complete!")
            print(f"    Image shape: {image_data.shape}")
            print(f"    New segmentation: {is_new}")
            print(f"    Organs found: {len(segmentor.get_organ_labels())}")

            # Test 3: Get organs in a slice
            print(f"\n[Test 3] Testing slice analysis...")
            mid_slice = image_data.shape[0] // 2
            organs_in_slice = segmentor.get_organs_in_slice(mid_slice, axis=0)
            print(f"  Middle axial slice ({mid_slice}):")
            for organ in organs_in_slice[:10]:
                print(f"    • {organ}")
            if len(organs_in_slice) > 10:
                print(f"    ... and {len(organs_in_slice) - 10} more")

        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\n[Test 2] Skipped (no NIfTI file provided)")
        print("  Usage: python organ_segmentor.py <nifti_file.nii.gz>")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
