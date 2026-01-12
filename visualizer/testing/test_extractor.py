"""
Unit tests for ActivationExtractor using adapter interface.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from extract_activations import ActivationExtractor, flatten_activations, load_activations
from adapters import HookMixin, GeneratorAdapter


class MockModule:
    """Mock PyTorch module for hook testing."""
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        handle = MockHandle(fn, self)
        self.hooks.append(handle)
        return handle


class MockHandle:
    """Mock hook handle."""
    def __init__(self, fn, module):
        self.fn = fn
        self.module = module
        self.removed = False

    def remove(self):
        self.removed = True


class MockAdapter(HookMixin, GeneratorAdapter):
    """Mock adapter for testing ActivationExtractor."""

    def __init__(self):
        HookMixin.__init__(self)
        self._modules = {
            'encoder_bottleneck': MockModule(),
            'midblock': MockModule()
        }
        self._device = 'cpu'

    @property
    def model_type(self):
        return 'mock'

    @property
    def resolution(self):
        return 64

    @property
    def num_classes(self):
        return 1000

    @property
    def hookable_layers(self):
        return ['encoder_bottleneck', 'midblock']

    def _get_layer_module(self, name):
        if name not in self._modules:
            raise ValueError(f"Unknown layer: {name}")
        return self._modules[name]

    def forward(self, x, sigma, class_labels=None, **kwargs):
        # Simulate triggering hooks
        for name, module in self._modules.items():
            for handle in module.hooks:
                output = torch.randn(x.shape[0], 768, 4, 4)
                handle.fn(module, None, output)
        return torch.randn_like(x)

    def register_activation_hooks(self, layer_names, hook_fn):
        handles = []
        for name in layer_names:
            module = self._get_layer_module(name)
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)
            self.add_handle(handle)
        return handles

    def get_layer_shapes(self):
        return {'encoder_bottleneck': (768, 4, 4), 'midblock': (768, 4, 4)}

    @classmethod
    def from_checkpoint(cls, path, device='cuda', **kwargs):
        return cls()

    @classmethod
    def get_default_config(cls):
        return {'img_resolution': 64, 'label_dim': 1000}


class TestActivationExtractor:
    """Test ActivationExtractor class."""

    def test_init_without_adapter(self):
        """Test extractor can init without adapter."""
        extractor = ActivationExtractor()
        assert extractor.adapter is None
        assert extractor.activations == {}
        assert extractor.hooks == []

    def test_init_with_adapter(self):
        """Test extractor can init with adapter."""
        adapter = MockAdapter()
        extractor = ActivationExtractor(adapter)
        assert extractor.adapter is adapter

    def test_register_hooks_without_adapter_raises(self):
        """Test register_hooks raises if no adapter."""
        extractor = ActivationExtractor()
        with pytest.raises(ValueError, match="No adapter provided"):
            extractor.register_hooks(['encoder_bottleneck'])

    def test_register_hooks_with_adapter(self):
        """Test register_hooks works with adapter."""
        adapter = MockAdapter()
        extractor = ActivationExtractor(adapter)
        extractor.register_hooks(['encoder_bottleneck', 'midblock'])

        # Should have 2 hooks registered
        assert len(extractor.hooks) == 2

    def test_register_hooks_adapter_override(self):
        """Test passing adapter to register_hooks overrides init adapter."""
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        extractor = ActivationExtractor(adapter1)
        extractor.register_hooks(['encoder_bottleneck'], adapter=adapter2)

        assert extractor.adapter is adapter2

    def test_extraction_hook_captures_activations(self):
        """Test extraction hook stores activations."""
        adapter = MockAdapter()
        extractor = ActivationExtractor(adapter)
        extractor.register_hooks(['encoder_bottleneck'])

        # Simulate forward pass (triggers hooks)
        x = torch.randn(2, 3, 64, 64)
        sigma = torch.ones(2)
        adapter.forward(x, sigma)

        # Check activations captured
        acts = extractor.get_activations()
        assert 'encoder_bottleneck' in acts
        assert acts['encoder_bottleneck'].shape[0] == 2  # batch size

    def test_clear_activations(self):
        """Test clear_activations empties store."""
        adapter = MockAdapter()
        extractor = ActivationExtractor(adapter)
        extractor.register_hooks(['encoder_bottleneck'])

        # Add some activations
        x = torch.randn(2, 3, 64, 64)
        adapter.forward(x, torch.ones(2))
        assert len(extractor.get_activations()) > 0

        # Clear
        extractor.clear_activations()
        assert len(extractor.get_activations()) == 0

    def test_remove_hooks(self):
        """Test remove_hooks clears hook list."""
        adapter = MockAdapter()
        extractor = ActivationExtractor(adapter)
        extractor.register_hooks(['encoder_bottleneck'])

        assert len(extractor.hooks) > 0
        extractor.remove_hooks()
        assert len(extractor.hooks) == 0

    def test_context_manager(self):
        """Test context manager removes hooks on exit."""
        adapter = MockAdapter()

        with ActivationExtractor(adapter) as extractor:
            extractor.register_hooks(['encoder_bottleneck'])
            assert len(extractor.hooks) > 0

        assert len(extractor.hooks) == 0

    def test_save_activations(self):
        """Test saving activations to disk."""
        adapter = MockAdapter()
        extractor = ActivationExtractor(adapter)
        extractor.register_hooks(['encoder_bottleneck'])

        # Simulate forward to capture activations
        x = torch.randn(2, 3, 64, 64)
        adapter.forward(x, torch.ones(2))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_activations"
            metadata = {"test_key": "test_value"}
            extractor.save_activations(output_path, metadata)

            # Check files created
            assert output_path.with_suffix('.npz').exists()
            assert output_path.with_suffix('.json').exists()

            # Check metadata
            with open(output_path.with_suffix('.json')) as f:
                loaded_meta = json.load(f)
            assert loaded_meta['test_key'] == 'test_value'

            # Check activations
            data = np.load(str(output_path.with_suffix('.npz')))
            assert 'encoder_bottleneck' in data.keys()


class TestFlattenActivations:
    """Test flatten_activations utility."""

    def test_flatten_single_layer(self):
        """Test flattening single layer activation."""
        activations = {
            'layer1': np.random.randn(2, 768 * 4 * 4)  # Already flat
        }
        result = flatten_activations(activations)
        assert result.shape == (2, 768 * 4 * 4)

    def test_flatten_multiple_layers(self):
        """Test flattening multiple layers."""
        activations = {
            'layer1': np.random.randn(2, 768 * 4 * 4),
            'layer2': np.random.randn(2, 512 * 8 * 8)
        }
        result = flatten_activations(activations)
        expected_features = 768 * 4 * 4 + 512 * 8 * 8
        assert result.shape == (2, expected_features)

    def test_flatten_layers_sorted(self):
        """Test layers are concatenated in sorted order."""
        activations = {
            'z_layer': np.ones((1, 10)),
            'a_layer': np.zeros((1, 5))
        }
        result = flatten_activations(activations)
        # a_layer (zeros) should come first
        assert result[0, 0] == 0.0
        assert result[0, 5] == 1.0

    def test_flatten_4d_activation(self):
        """Test flattening 4D (B, C, H, W) activations."""
        activations = {
            'layer1': np.random.randn(2, 64, 8, 8)  # 4D
        }
        result = flatten_activations(activations)
        assert result.shape == (2, 64 * 8 * 8)


class TestLoadActivations:
    """Test load_activations utility."""

    def test_load_saved_activations(self):
        """Test loading previously saved activations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_act"

            # Save
            test_activations = {'layer1': np.random.randn(2, 100)}
            test_metadata = {'test': 'metadata'}
            np.savez_compressed(str(output_path.with_suffix('.npz')), **test_activations)
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(test_metadata, f)

            # Load
            loaded_acts, loaded_meta = load_activations(output_path)

            assert 'layer1' in loaded_acts
            assert loaded_acts['layer1'].shape == (2, 100)
            assert loaded_meta['test'] == 'metadata'

    def test_load_without_metadata(self):
        """Test loading when metadata file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_act"

            # Save only activations
            test_activations = {'layer1': np.random.randn(2, 100)}
            np.savez_compressed(str(output_path.with_suffix('.npz')), **test_activations)

            # Load
            loaded_acts, loaded_meta = load_activations(output_path)

            assert 'layer1' in loaded_acts
            assert loaded_meta == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
