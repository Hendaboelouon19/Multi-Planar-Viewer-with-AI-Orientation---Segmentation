"""
Orientation Detection Module
Standalone file for detecting medical image orientation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom


class Orientation3DCNN(nn.Module):
    """3D CNN for orientation classification"""

    def __init__(self, num_classes=3):
        super(Orientation3DCNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class OrientationDetector:
    """
    Orientation detector for medical images

    Usage:
        detector = OrientationDetector('model.pth')
        orientation, confidence, probs = detector.predict(volume_array)
    """

    def __init__(self, model_path=None):
        """
        Initialize detector

        Args:
            model_path: Path to .pth model file (optional, can load later)
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Orientation3DCNN(num_classes=3)
        self.loaded = False
        self.orientations = {
            0: 'axial',
            1: 'sagittal',
            2: 'coronal'
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load trained model from file

        Args:
            model_path: Path to .pth model file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)

            # Update orientations from checkpoint if available
            if 'orientations' in checkpoint:
                self.orientations = checkpoint['orientations']

            self.loaded = True
            print(f"✓ Orientation model loaded on {self.device}")
            print(f"  Orientations: {list(self.orientations.values())}")
            return True

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            self.loaded = False
            return False

    def preprocess_volume(self, volume, target_size=(64, 64, 64)):
        """
        Preprocess 3D volume for model input

        Args:
            volume: numpy array (Z, Y, X) or (Z, Y, X, C)
            target_size: target dimensions (default 64x64x64)

        Returns:
            torch.Tensor: preprocessed volume
        """
        # Handle 4D volumes (with channel dimension)
        if len(volume.shape) == 4:
            volume = volume[:, :, :, 0]

        # Resize to target size
        zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
        resized = zoom(volume, zoom_factors, order=1)

        # Normalize to [0, 1]
        resized = (resized - resized.min()) / \
            (resized.max() - resized.min() + 1e-8)

        # Convert to tensor (1, 1, D, H, W)
        tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0)

        return tensor

    def predict(self, volume):
        """
        Predict orientation of 3D volume

        Args:
            volume: numpy array (Z, Y, X) or (Z, Y, X, C)

        Returns:
            tuple: (orientation: str, confidence: float, probabilities: dict)
                   Returns (None, 0.0, {}) if model not loaded
        """
        if not self.loaded:
            print("⚠ Model not loaded. Call load_model() first.")
            return None, 0.0, {}

        try:
            # Preprocess
            volume_tensor = self.preprocess_volume(volume)
            volume_tensor = volume_tensor.to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(volume_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)

            # Extract results
            predicted_class = predicted.item()
            confidence_value = confidence.item()
            orientation = self.orientations[predicted_class]

            # Create probability dictionary
            prob_dict = {
                self.orientations[i]: probabilities[0][i].item()
                for i in range(len(self.orientations))
            }

            return orientation, confidence_value, prob_dict

        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return None, 0.0, {}

    def is_loaded(self):
        """Check if model is loaded"""
        return self.loaded

    def get_device(self):
        """Get current device (cuda/cpu)"""
        return str(self.device)


# ============================================================================
# STANDALONE TESTING
# ============================================================================
if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Orientation Detector - Standalone Test")
    print("=" * 60)

    # Test 1: Initialize detector
    print("\n[Test 1] Initialize detector...")
    detector = OrientationDetector()
    print(f"  Device: {detector.get_device()}")
    print(f"  Loaded: {detector.is_loaded()}")

    # Test 2: Load model
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"\n[Test 2] Loading model from: {model_path}")
        success = detector.load_model(model_path)
        if success:
            print("  ✓ Model loaded successfully")
        else:
            print("  ✗ Failed to load model")
            sys.exit(1)
    else:
        print("\n[Test 2] Skipped (no model path provided)")
        print("  Usage: python orientation_detector.py <model_path.pth>")
        sys.exit(0)

    # Test 3: Test with dummy data
    print("\n[Test 3] Testing with dummy volume...")
    dummy_volume = np.random.randn(
        100, 100, 100).astype(np.float32) * 100 + 500
    print(f"  Volume shape: {dummy_volume.shape}")

    orientation, confidence, probs = detector.predict(dummy_volume)

    if orientation:
        print(f"\n  ✓ Prediction successful!")
        print(f"    Orientation: {orientation}")
        print(f"    Confidence: {confidence*100:.1f}%")
        print(f"    All probabilities:")
        for orient, prob in probs.items():
            print(f"      {orient:12s}: {prob*100:5.1f}%")
    else:
        print("  ✗ Prediction failed")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
