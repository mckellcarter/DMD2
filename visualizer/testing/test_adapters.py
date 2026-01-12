"""
Unit tests for generator adapters.
"""

import pytest
import torch
from typing import Dict, List, Tuple

# Import adapter interface
from adapters import (
    GeneratorAdapter,
    HookMixin,
    get_adapter,
    list_adapters,
    register_adapter,
    DMD2ImageNetAdapter,
    EDMImageNetAdapter,
)
from adapters.base import GeneratorAdapter as BaseAdapter
from adapters.hooks import HookMixin as BaseHookMixin


class TestAdapterRegistry:
    """Test adapter registration and discovery."""

    def test_list_adapters_returns_both(self):
        """Test that both DMD2 and EDM adapters are registered."""
        adapters = list_adapters()
        assert 'dmd2-imagenet-64' in adapters
        assert 'edm-imagenet-64' in adapters

    def test_get_adapter_dmd2(self):
        """Test getting DMD2 adapter class."""
        adapter_cls = get_adapter('dmd2-imagenet-64')
        assert adapter_cls is DMD2ImageNetAdapter
        assert issubclass(adapter_cls, GeneratorAdapter)

    def test_get_adapter_edm(self):
        """Test getting EDM adapter class."""
        adapter_cls = get_adapter('edm-imagenet-64')
        assert adapter_cls is EDMImageNetAdapter
        assert issubclass(adapter_cls, GeneratorAdapter)

    def test_get_adapter_unknown_raises(self):
        """Test that unknown adapter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter('nonexistent-adapter')

    def test_register_adapter_decorator(self):
        """Test custom adapter registration."""
        @register_adapter('test-adapter-001')
        class TestAdapter(GeneratorAdapter):
            @property
            def model_type(self): return 'test'
            @property
            def resolution(self): return 64
            @property
            def num_classes(self): return 10
            @property
            def hookable_layers(self): return []
            def forward(self, x, sigma, class_labels=None, **kwargs): return x
            def register_activation_hooks(self, layer_names, hook_fn): return []
            def get_layer_shapes(self): return {}
            @classmethod
            def from_checkpoint(cls, path, device='cuda', **kwargs): return cls()
            @classmethod
            def get_default_config(cls): return {}

        assert 'test-adapter-001' in list_adapters()
        assert get_adapter('test-adapter-001') is TestAdapter


class TestHookMixin:
    """Test HookMixin functionality."""

    def test_hook_mixin_initialization(self):
        """Test HookMixin initializes empty stores."""
        class TestClass(HookMixin):
            def __init__(self):
                HookMixin.__init__(self)

        obj = TestClass()
        assert obj._activations == {}
        assert obj._handles == []
        assert obj._masks == {}

    def test_make_extraction_hook(self):
        """Test extraction hook captures output."""
        class TestClass(HookMixin):
            def __init__(self):
                HookMixin.__init__(self)

        obj = TestClass()
        hook = obj.make_extraction_hook('test_layer')

        # Simulate forward pass
        output = torch.randn(2, 64, 8, 8)
        hook(None, None, output)

        assert 'test_layer' in obj._activations
        assert obj._activations['test_layer'].shape == (2, 64, 8, 8)
        assert obj._activations['test_layer'].device.type == 'cpu'

    def test_make_extraction_hook_tuple_output(self):
        """Test extraction hook handles tuple outputs."""
        class TestClass(HookMixin):
            def __init__(self):
                HookMixin.__init__(self)

        obj = TestClass()
        hook = obj.make_extraction_hook('test_layer')

        # Simulate tuple output
        output = (torch.randn(2, 64, 8, 8), torch.randn(2, 128))
        hook(None, None, output)

        # Should extract first element
        assert obj._activations['test_layer'].shape == (2, 64, 8, 8)

    def test_make_mask_hook(self):
        """Test mask hook replaces output."""
        class TestClass(HookMixin):
            def __init__(self):
                HookMixin.__init__(self)

        obj = TestClass()
        mask = torch.ones(1, 64, 8, 8)
        hook = obj.make_mask_hook('test_layer', mask)

        # Simulate forward pass with different output
        output = torch.zeros(1, 64, 8, 8)
        result = hook(None, None, output)

        # Should return mask, not output
        assert torch.allclose(result, mask)

    def test_make_mask_hook_batch_expansion(self):
        """Test mask hook expands to batch size."""
        class TestClass(HookMixin):
            def __init__(self):
                HookMixin.__init__(self)

        obj = TestClass()
        mask = torch.ones(1, 64, 8, 8)
        hook = obj.make_mask_hook('test_layer', mask)

        # Simulate batch of 4
        output = torch.zeros(4, 64, 8, 8)
        result = hook(None, None, output)

        assert result.shape == (4, 64, 8, 8)
        # All should be ones
        assert torch.allclose(result, torch.ones(4, 64, 8, 8))

    def test_set_get_clear_mask(self):
        """Test mask set/get/clear operations."""
        class TestClass(HookMixin):
            def __init__(self):
                HookMixin.__init__(self)

        obj = TestClass()
        mask = torch.randn(1, 64, 8, 8)

        # Set
        obj.set_mask('layer1', mask)
        assert obj.get_mask('layer1') is not None
        assert torch.equal(obj.get_mask('layer1'), mask)

        # Get nonexistent
        assert obj.get_mask('nonexistent') is None

        # Clear single
        obj.clear_mask('layer1')
        assert obj.get_mask('layer1') is None

        # Clear all
        obj.set_mask('layer1', mask)
        obj.set_mask('layer2', mask)
        obj.clear_masks()
        assert len(obj._masks) == 0

    def test_activations_operations(self):
        """Test activation get/clear operations."""
        class TestClass(HookMixin):
            def __init__(self):
                HookMixin.__init__(self)

        obj = TestClass()
        obj._activations['layer1'] = torch.randn(1, 64, 8, 8)
        obj._activations['layer2'] = torch.randn(1, 128, 4, 4)

        # Get all
        acts = obj.get_activations()
        assert len(acts) == 2
        assert 'layer1' in acts

        # Get single
        act = obj.get_activation('layer1')
        assert act.shape == (1, 64, 8, 8)

        # Clear
        obj.clear_activations()
        assert len(obj._activations) == 0


class TestGeneratorAdapterInterface:
    """Test GeneratorAdapter abstract interface."""

    def test_adapter_has_required_properties(self):
        """Test adapter classes have required properties."""
        for adapter_name in ['dmd2-imagenet-64', 'edm-imagenet-64']:
            cls = get_adapter(adapter_name)
            # These should be defined (can't instantiate without model)
            assert hasattr(cls, 'model_type')
            assert hasattr(cls, 'resolution')
            assert hasattr(cls, 'num_classes')
            assert hasattr(cls, 'hookable_layers')

    def test_adapter_has_required_methods(self):
        """Test adapter classes have required methods."""
        for adapter_name in ['dmd2-imagenet-64', 'edm-imagenet-64']:
            cls = get_adapter(adapter_name)
            assert hasattr(cls, 'forward')
            assert hasattr(cls, 'register_activation_hooks')
            assert hasattr(cls, 'get_layer_shapes')
            assert hasattr(cls, 'from_checkpoint')
            assert hasattr(cls, 'get_default_config')

    def test_get_default_config_returns_dict(self):
        """Test get_default_config returns valid dict."""
        for adapter_name in ['dmd2-imagenet-64', 'edm-imagenet-64']:
            cls = get_adapter(adapter_name)
            config = cls.get_default_config()

            assert isinstance(config, dict)
            assert 'img_resolution' in config
            assert config['img_resolution'] == 64
            assert 'label_dim' in config
            assert config['label_dim'] == 1000


class TestDMD2AdapterConfig:
    """Test DMD2ImageNetAdapter specific configuration."""

    def test_model_type(self):
        """Test DMD2 adapter model type."""
        assert DMD2ImageNetAdapter.get_default_config()['model_type'] == 'DhariwalUNet'

    def test_inherits_from_correct_classes(self):
        """Test DMD2 adapter inheritance."""
        assert issubclass(DMD2ImageNetAdapter, GeneratorAdapter)
        assert issubclass(DMD2ImageNetAdapter, HookMixin)


class TestEDMAdapterConfig:
    """Test EDMImageNetAdapter specific configuration."""

    def test_model_type(self):
        """Test EDM adapter model type."""
        assert EDMImageNetAdapter.get_default_config()['model_type'] == 'DhariwalUNet'

    def test_inherits_from_correct_classes(self):
        """Test EDM adapter inheritance."""
        assert issubclass(EDMImageNetAdapter, GeneratorAdapter)
        assert issubclass(EDMImageNetAdapter, HookMixin)

    def test_has_sample_method(self):
        """Test EDM adapter has sample() for multi-step sampling."""
        assert hasattr(EDMImageNetAdapter, 'sample')

    def test_has_from_pickle_method(self):
        """Test EDM adapter has from_pickle() for loading pkl files."""
        assert hasattr(EDMImageNetAdapter, 'from_pickle')


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


class TestMockAdapterIntegration:
    """Integration tests with mock adapter."""

    def create_mock_adapter(self):
        """Create a minimal mock adapter for testing."""
        class MockUNet:
            def __init__(self):
                self.enc = {'64x64': MockModule(), '32x32': MockModule(), '16x16': MockModule()}
                self.dec = {'16x16': MockModule(), '32x32': MockModule(), '64x64': MockModule()}

        class MockModel:
            def __init__(self):
                self.model = MockUNet()

            def __call__(self, x, sigma, labels):
                return torch.randn_like(x)

        class MockAdapter(HookMixin, GeneratorAdapter):
            def __init__(self):
                HookMixin.__init__(self)
                self._model = MockModel()
                self._device = 'cpu'

            @property
            def model_type(self): return 'mock'
            @property
            def resolution(self): return 64
            @property
            def num_classes(self): return 1000
            @property
            def hookable_layers(self): return ['encoder_bottleneck', 'midblock']

            def _get_layer_module(self, name):
                if name == 'encoder_bottleneck':
                    return self._model.model.enc['16x16']
                elif name == 'midblock':
                    return self._model.model.dec['16x16']
                raise ValueError(f"Unknown layer: {name}")

            def forward(self, x, sigma, class_labels=None, **kwargs):
                return self._model(x, sigma, class_labels)

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

        return MockAdapter()

    def test_mock_adapter_properties(self):
        """Test mock adapter has correct properties."""
        adapter = self.create_mock_adapter()
        assert adapter.model_type == 'mock'
        assert adapter.resolution == 64
        assert adapter.num_classes == 1000
        assert 'encoder_bottleneck' in adapter.hookable_layers

    def test_mock_adapter_forward(self):
        """Test mock adapter forward pass."""
        adapter = self.create_mock_adapter()
        x = torch.randn(2, 3, 64, 64)
        sigma = torch.ones(2)
        labels = torch.zeros(2, 1000)

        output = adapter.forward(x, sigma, labels)
        assert output.shape == x.shape

    def test_mock_adapter_hook_registration(self):
        """Test hook registration on mock adapter."""
        adapter = self.create_mock_adapter()

        def dummy_hook(module, inp, output):
            pass

        handles = adapter.register_activation_hooks(['encoder_bottleneck'], dummy_hook)

        assert len(handles) == 1
        assert adapter.num_hooks == 1

        adapter.remove_hooks()
        assert adapter.num_hooks == 0

    def test_mock_adapter_get_layer_shapes(self):
        """Test get_layer_shapes on mock adapter."""
        adapter = self.create_mock_adapter()
        shapes = adapter.get_layer_shapes()

        assert 'encoder_bottleneck' in shapes
        assert shapes['encoder_bottleneck'] == (768, 4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
