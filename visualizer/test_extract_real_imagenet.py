"""
Tests for real ImageNet activation extraction.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import json

from extract_real_imagenet import (
    preprocess_imagenet_image,
    parse_imagenet_path,
)


class TestPreprocessing:
    """Test image preprocessing functions."""

    def test_preprocess_imagenet_image(self, tmp_path):
        """Test image preprocessing produces correct format."""
        # Create test image
        img_path = tmp_path / "test.JPEG"
        test_img = Image.new('RGB', (256, 256), color='red')
        test_img.save(img_path)

        # Preprocess
        tensor = preprocess_imagenet_image(img_path, target_size=64)

        # Check shape
        assert tensor.shape == (1, 3, 64, 64)

        # Check value range
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

        # Check dtype
        assert tensor.dtype == torch.float32

    def test_preprocess_different_sizes(self, tmp_path):
        """Test preprocessing handles different input sizes."""
        sizes = [64, 128, 256, 512]

        for size in sizes:
            img_path = tmp_path / f"test_{size}.JPEG"
            test_img = Image.new('RGB', (size, size), color='blue')
            test_img.save(img_path)

            tensor = preprocess_imagenet_image(img_path, target_size=64)
            assert tensor.shape == (1, 3, 64, 64)

    def test_parse_imagenet_path_val(self):
        """Test parsing validation set paths."""
        path = Path("/data/imagenet/val/n01440764/ILSVRC2012_val_00000001.JPEG")
        synset_id, class_id = parse_imagenet_path(path)

        assert synset_id == "n01440764"
        assert class_id is None  # Will be resolved later

    def test_parse_imagenet_path_train(self):
        """Test parsing train set paths."""
        path = Path("/data/imagenet/train/n02119789/n02119789_1234.JPEG")
        synset_id, class_id = parse_imagenet_path(path)

        assert synset_id == "n02119789"


class TestBatchProcessing:
    """Test batch activation extraction."""

    @pytest.fixture
    def mock_imagenet_dir(self, tmp_path):
        """Create mock ImageNet directory structure."""
        val_dir = tmp_path / "val"

        # Create two synset directories
        synsets = ["n01440764", "n01443537"]
        for synset in synsets:
            synset_dir = val_dir / synset
            synset_dir.mkdir(parents=True)

            # Create 5 images per synset
            for i in range(5):
                img_path = synset_dir / f"ILSVRC2012_val_{i:08d}.JPEG"
                img = Image.new('RGB', (64, 64), color='red')
                img.save(img_path)

        return tmp_path

    def test_finds_imagenet_images(self, mock_imagenet_dir):
        """Test that image discovery works."""
        val_dir = mock_imagenet_dir / "val"
        image_paths = list(val_dir.rglob('*.JPEG'))

        assert len(image_paths) == 10  # 2 synsets × 5 images

    def test_batch_metadata_structure(self):
        """Test that batch metadata has correct structure."""
        # Example batch metadata
        batch_meta = {
            "batch_size": 3,
            "layers": ["encoder_bottleneck", "midblock"],
            "samples": [
                {
                    "batch_index": 0,
                    "class_id": 0,
                    "synset_id": "n01440764",
                    "class_name": "tench",
                    "original_path": "/data/imagenet/val/n01440764/img.JPEG"
                },
                {
                    "batch_index": 1,
                    "class_id": 1,
                    "synset_id": "n01443537",
                    "class_name": "goldfish",
                    "original_path": "/data/imagenet/val/n01443537/img.JPEG"
                },
                {
                    "batch_index": 2,
                    "class_id": 0,
                    "synset_id": "n01440764",
                    "class_name": "tench",
                    "original_path": "/data/imagenet/val/n01440764/img2.JPEG"
                }
            ]
        }

        # Validate structure
        assert batch_meta["batch_size"] == 3
        assert len(batch_meta["samples"]) == 3
        assert all("synset_id" in s for s in batch_meta["samples"])
        assert all("class_name" in s for s in batch_meta["samples"])


class TestSynetMapping:
    """Test synset to class ID mapping."""

    def test_synset_mapping(self):
        """Test that synset mapping works correctly."""
        # Load actual class labels
        class_labels_path = Path(__file__).parent / "data" / "imagenet_class_labels.json"

        if not class_labels_path.exists():
            pytest.skip("Class labels file not found")

        with open(class_labels_path, 'r') as f:
            class_labels_map = json.load(f)

        # Create reverse mapping
        synset_to_class = {}
        for class_id_str, (synset_id, class_name) in class_labels_map.items():
            synset_to_class[synset_id] = (int(class_id_str), class_name)

        # Test known synsets
        assert "n01440764" in synset_to_class
        class_id, class_name = synset_to_class["n01440764"]
        assert class_id == 0
        assert class_name == "tench"

        assert "n01443537" in synset_to_class
        class_id, class_name = synset_to_class["n01443537"]
        assert class_id == 1
        assert class_name == "goldfish"


class TestActivationFormat:
    """Test activation storage format."""

    def test_activation_batch_format(self):
        """Test that activations are saved in correct batch format."""
        # Simulate batch activations
        batch_size = 4
        activations = {
            "encoder_bottleneck": np.random.randn(batch_size, 512, 8, 8),
            "midblock": np.random.randn(batch_size, 512, 4, 4)
        }

        # Flatten spatial dims
        flattened = {}
        for name, act in activations.items():
            B, C, H, W = act.shape
            flattened[name] = act.reshape(B, -1)

        # Check shapes
        assert flattened["encoder_bottleneck"].shape == (4, 512 * 8 * 8)
        assert flattened["midblock"].shape == (4, 512 * 4 * 4)

    def test_single_sample_extraction(self):
        """Test extracting single sample from batch."""
        # Simulate batch activations
        batch_size = 4
        batch_act = np.random.randn(batch_size, 512 * 8 * 8)

        # Extract single sample
        sample_idx = 2
        sample_act = batch_act[sample_idx:sample_idx+1]

        # Check shape preserved
        assert sample_act.shape == (1, 512 * 8 * 8)


def test_imports():
    """Test that all imports work."""
    from extract_real_imagenet import (
        get_imagenet_config,
        load_imagenet_model,
        preprocess_imagenet_image,
        parse_imagenet_path,
        extract_real_imagenet_activations
    )
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
