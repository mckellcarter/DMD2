# Session: Generator Adapter Refactoring

## Date
2026-01-11

## Summary
Refactored visualization_app.py to use a generator adapter pattern, enabling support for multiple diffusion model architectures through a common interface.

## Changes Made

### New Files Created
- `adapters/base.py` - `GeneratorAdapter` abstract base class
- `adapters/hooks.py` - `HookMixin` for hook lifecycle management
- `adapters/registry.py` - `get_adapter()`, `register_adapter()` for discovery
- `core/__init__.py` - Core module exports
- `core/masking.py` - `ActivationMasker` using adapter interface
- `core/generator.py` - `generate_with_mask`, `generate_with_mask_multistep`
- `core/extractor.py` - `ActivationExtractor` for activation capture

### Files Modified
- `adapters/__init__.py` - Updated exports
- `adapters/dmd2_imagenet.py` - Changed imports from diffviews to local
- `visualization_app.py` - Refactored to use adapter pattern

### Key API Changes
| Old | New |
|-----|-----|
| `self.generator` | `self.adapter` |
| `create_imagenet_generator()` | `get_adapter(name).from_checkpoint()` |
| `ActivationMask(model_type=...)` | `ActivationMasker(adapter)` |
| `mask.register_hooks(model, layers)` | `masker.register_hooks(layers)` |
| `generate_with_masked_activation_multistep()` | `generate_with_mask_multistep()` |

### New CLI Arguments
- `--adapter` - Adapter name for model loading (default: `dmd2-imagenet-64`)

## Adapter Interface

```python
class GeneratorAdapter(ABC):
    @property
    def model_type(self) -> str: ...
    @property
    def resolution(self) -> int: ...
    @property
    def num_classes(self) -> int: ...
    @property
    def hookable_layers(self) -> List[str]: ...

    def forward(self, x, sigma, class_labels=None, **kwargs) -> Tensor: ...
    def register_activation_hooks(self, layer_names, hook_fn) -> List[Handle]: ...
    def get_layer_shapes(self) -> Dict[str, Tuple]: ...

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda', **kwargs): ...
```

## Files Deprecated (can be removed)
After verification, these old modules are no longer used:
- `activation_masking.py` - Replaced by `core/masking.py`
- `generate_from_activation.py` - Replaced by `core/generator.py`
- `extract_activations.py` - Replaced by `core/extractor.py`

## Testing
- Verified imports work correctly
- Adapter discovery finds `dmd2-imagenet-64`
- User confirmed generation works in visualizer

## Branch
`feature/generator-adapter-refactor`
