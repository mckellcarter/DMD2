"""
Unit tests for core/generator.py - adapted from original test_generate_from_activation.py
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import json

from core.generator import (
    tensor_to_uint8_image,
    get_denoising_sigmas,
    generate_with_mask,
    save_generated_sample,
    infer_layer_shape
)
from core.masking import ActivationMasker
from adapters import GeneratorAdapter, HookMixin


class MockAdapter(HookMixin, GeneratorAdapter):
    """Mock adapter for testing generation."""

    def __init__(self, output_shape=(1, 3, 64, 64)):
        HookMixin.__init__(self)
        self.output_shape = output_shape
        self.calls = []
        self._device = 'cpu'

    @property
    def model_type(self): return 'mock'

    @property
    def resolution(self): return 64

    @property
    def num_classes(self): return 1000

    @property
    def hookable_layers(self):
        return ['encoder_bottleneck', 'midblock']

    def forward(self, x, sigma, class_labels=None, **kwargs):
        self.calls.append({
            'x': x,
            'sigma': sigma,
            'class_labels': class_labels
        })
        return torch.randn(*self.output_shape).to(x.device)

    def register_activation_hooks(self, layer_names, hook_fn):
        return []

    def get_layer_shapes(self):
        return {'encoder_bottleneck': (768, 4, 4), 'midblock': (768, 4, 4)}

    @classmethod
    def from_checkpoint(cls, path, device='cuda', **kwargs):
        return cls()

    @classmethod
    def get_default_config(cls):
        return {'img_resolution': 64, 'label_dim': 1000}

    def eval(self):
        return self


class TestTensorToUint8Image:
    """Test image tensor conversion."""

    def test_basic_conversion(self):
        """Test basic [-1, 1] to [0, 255] conversion."""
        tensor = torch.zeros(1, 3, 64, 64)
        result = tensor_to_uint8_image(tensor)

        assert result.dtype == torch.uint8
        assert result.shape == (1, 64, 64, 3)
        # 0 in [-1, 1] maps to ~127
        assert result.float().mean() == pytest.approx(127.5, abs=1)

    def test_min_max_values(self):
        """Test conversion of min/max values."""
        tensor = torch.ones(1, 3, 64, 64)
        result = tensor_to_uint8_image(tensor)
        assert result.max() == 255

        tensor = torch.ones(1, 3, 64, 64) * -1
        result = tensor_to_uint8_image(tensor)
        assert result.min() == 0

    def test_clamping(self):
        """Test values outside [-1, 1] are clamped."""
        tensor = torch.ones(1, 3, 64, 64) * 2  # Out of range
        result = tensor_to_uint8_image(tensor)
        assert result.max() == 255

        tensor = torch.ones(1, 3, 64, 64) * -2
        result = tensor_to_uint8_image(tensor)
        assert result.min() == 0


class TestGetDenoisingSigmas:
    """Test sigma schedule generation."""

    def test_basic_schedule(self):
        """Test basic sigma schedule."""
        sigmas = get_denoising_sigmas(4, sigma_max=80.0, sigma_min=0.002)

        assert len(sigmas) == 4
        assert sigmas[0] > sigmas[-1]  # Descending
        assert sigmas[0] == pytest.approx(80.0, rel=0.01)
        assert sigmas[-1] == pytest.approx(0.002, rel=0.1)

    def test_single_step(self):
        """Test single step schedule."""
        sigmas = get_denoising_sigmas(1, sigma_max=80.0, sigma_min=0.002)

        assert len(sigmas) == 1
        assert sigmas[0] == pytest.approx(80.0, rel=0.01)

    def test_different_num_steps(self):
        """Test various step counts."""
        for n in [2, 4, 10, 50]:
            sigmas = get_denoising_sigmas(n, sigma_max=80.0, sigma_min=0.002)
            assert len(sigmas) == n
            # Should be strictly decreasing
            for i in range(len(sigmas) - 1):
                assert sigmas[i] > sigmas[i + 1]


class TestGenerateWithMask:
    """Test image generation with masked activations."""

    def test_basic_generation(self):
        """Test basic generation call."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask(
            adapter,
            masker,
            class_label=42,
            num_samples=1,
            device='cpu'
        )

        assert images.shape == (1, 64, 64, 3)
        assert images.dtype == torch.uint8
        assert labels.shape == (1,)
        assert labels[0] == 42

        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call['x'].shape == (1, 3, 64, 64)
        assert call['class_labels'].shape == (1, 1000)
        assert call['class_labels'][0, 42] == 1.0

    def test_batch_generation(self):
        """Test generating multiple samples."""
        adapter = MockAdapter(output_shape=(4, 3, 64, 64))
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask(
            adapter,
            masker,
            class_label=100,
            num_samples=4,
            device='cpu'
        )

        assert images.shape == (4, 64, 64, 3)
        assert labels.shape == (4,)
        assert torch.all(labels == 100)

    def test_random_labels(self):
        """Test generation with random class labels."""
        adapter = MockAdapter(output_shape=(3, 3, 64, 64))
        masker = ActivationMasker(adapter)

        images, labels = generate_with_mask(
            adapter,
            masker,
            class_label=None,
            num_samples=3,
            device='cpu'
        )

        assert images.shape == (3, 64, 64, 3)
        assert labels.shape == (3,)
        assert torch.all(labels >= 0)
        assert torch.all(labels < 1000)

    def test_conditioning_sigma(self):
        """Test that conditioning sigma is applied correctly."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        sigma_value = 100.0

        generate_with_mask(
            adapter,
            masker,
            conditioning_sigma=sigma_value,
            num_samples=1,
            device='cpu'
        )

        call = adapter.calls[0]
        assert torch.allclose(call['sigma'], torch.tensor([sigma_value]))

    def test_image_value_range(self):
        """Test that output images are in valid uint8 range."""
        adapter = MockAdapter()
        masker = ActivationMasker(adapter)

        images, _ = generate_with_mask(
            adapter,
            masker,
            num_samples=1,
            device='cpu'
        )

        assert images.dtype == torch.uint8
        assert images.min() >= 0
        assert images.max() <= 255


class TestSaveGeneratedSample:
    """Test saving generated samples."""

    def test_save_basic(self):
        """Test basic sample saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
            activations = {'midblock': torch.randn(1, 256, 8, 8)}
            metadata = {
                'sample_id': 'test_sample',
                'class_label': 42,
                'model': 'imagenet'
            }

            record = save_generated_sample(
                image,
                activations,
                metadata,
                output_dir,
                'test_sample'
            )

            assert record['sample_id'] == 'test_sample'
            assert record['class_label'] == 42
            assert 'image_path' in record

            image_path = output_dir / record['image_path']
            assert image_path.exists()

            loaded_img = Image.open(image_path)
            assert loaded_img.size == (64, 64)

            activation_path = output_dir / "activations" / "imagenet" / "test_sample.npz"
            assert activation_path.exists()

            metadata_path = output_dir / "activations" / "imagenet" / "test_sample.json"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                loaded_meta = json.load(f)
            assert loaded_meta['class_label'] == 42

    def test_save_creates_directories(self):
        """Test that necessary directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
            activations = {'layer': torch.randn(1, 128, 16, 16)}
            metadata = {'sample_id': 'test', 'model': 'imagenet'}

            save_generated_sample(
                image,
                activations,
                metadata,
                output_dir,
                'test'
            )

            assert (output_dir / "images" / "imagenet").exists()
            assert (output_dir / "activations" / "imagenet").exists()

    def test_save_flattens_activations(self):
        """Test that spatial activations are flattened."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)

            activations = {
                'encoder': torch.randn(1, 256, 8, 8),
                'midblock': torch.randn(1, 512, 4, 4)
            }
            metadata = {'sample_id': 'test', 'model': 'imagenet'}

            save_generated_sample(
                image,
                activations,
                metadata,
                output_dir,
                'test'
            )

            activation_path = output_dir / "activations" / "imagenet" / "test.npz"
            loaded = np.load(activation_path)

            assert loaded['encoder'].shape == (1, 256 * 8 * 8)
            assert loaded['midblock'].shape == (1, 512 * 4 * 4)

    def test_save_multiple_samples(self):
        """Test saving multiple samples to same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            for i in range(3):
                image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
                activations = {'layer': torch.randn(1, 128, 16, 16)}
                metadata = {
                    'sample_id': f'sample_{i:03d}',
                    'model': 'imagenet'
                }

                save_generated_sample(
                    image,
                    activations,
                    metadata,
                    output_dir,
                    f'sample_{i:03d}'
                )

            image_dir = output_dir / "images" / "imagenet"
            assert len(list(image_dir.glob("*.png"))) == 3

            activation_dir = output_dir / "activations" / "imagenet"
            assert len(list(activation_dir.glob("*.npz"))) == 3

    def test_save_empty_activations(self):
        """Test saving with empty activations dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            image = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
            metadata = {'sample_id': 'test', 'model': 'imagenet'}

            record = save_generated_sample(
                image,
                {},  # Empty activations
                metadata,
                output_dir,
                'test'
            )

            # Image should still be saved
            image_path = output_dir / record['image_path']
            assert image_path.exists()


class TestIntegration:
    """Integration tests."""

    def test_generate_and_save_workflow(self):
        """Test complete generate and save workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            adapter = MockAdapter()
            masker = ActivationMasker(adapter)
            masker.set_mask("midblock", torch.randn(1, 256, 8, 8))

            images, labels = generate_with_mask(
                adapter,
                masker,
                class_label=42,
                num_samples=1,
                device='cpu'
            )

            activations = {'midblock': torch.randn(1, 256, 8, 8)}
            metadata = {
                'sample_id': 'generated_001',
                'class_label': int(labels[0]),
                'model': 'imagenet',
                'generated_from_neighbors': [1, 2, 3]
            }

            record = save_generated_sample(
                images[0],
                activations,
                metadata,
                output_dir,
                'generated_001'
            )

            assert record['sample_id'] == 'generated_001'
            assert record['class_label'] == 42

            image_path = output_dir / record['image_path']
            assert image_path.exists()

            activation_path = output_dir / "activations" / "imagenet" / "generated_001.npz"
            assert activation_path.exists()

            loaded_img = Image.open(image_path)
            assert loaded_img.size == (64, 64)

            loaded_act = np.load(activation_path)
            assert 'midblock' in loaded_act


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
