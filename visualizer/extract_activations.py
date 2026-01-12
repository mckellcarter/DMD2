"""
Extract UNet activations during inference for visualization.
Uses adapter interface for model-agnostic hook registration.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


class ActivationExtractor:
    """
    Hook-based activation extraction using adapter interface.

    Works with any GeneratorAdapter implementation (DMD2, EDM, etc.)
    by delegating hook registration to the adapter.

    Example:
        from adapters import get_adapter

        adapter = get_adapter('dmd2-imagenet-64').from_checkpoint(ckpt_path)
        extractor = ActivationExtractor(adapter)
        extractor.register_hooks(['encoder_bottleneck', 'midblock'])

        output = adapter.forward(x, sigma, labels)
        activations = extractor.get_activations()
    """

    def __init__(self, adapter=None):
        """
        Args:
            adapter: GeneratorAdapter instance (optional, can set later via register_hooks)
        """
        self.adapter = adapter
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _make_extraction_hook(self, layer_name: str):
        """Create hook function that stores activations."""
        def hook(_module, _inp, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[layer_name] = output.detach().cpu()
        return hook

    def register_hooks(self, layers: List[str], adapter=None):
        """
        Register extraction hooks on specified layers.

        Args:
            layers: List of layer names (e.g., ['encoder_bottleneck', 'midblock'])
            adapter: Optional adapter override (uses self.adapter if not provided)
        """
        if adapter is not None:
            self.adapter = adapter

        if self.adapter is None:
            raise ValueError("No adapter provided. Pass adapter to __init__ or register_hooks()")

        for layer_name in layers:
            handles = self.adapter.register_activation_hooks(
                [layer_name],
                self._make_extraction_hook(layer_name)
            )
            self.hooks.extend(handles)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get current activations."""
        return self.activations

    def save_activations(self, output_path: Path, metadata: Dict = None):
        """
        Save activations to disk.

        Args:
            output_path: Path to save activations (without extension)
            metadata: Optional metadata dict (label, prompt, etc.)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save activations as compressed numpy
        activation_dict = {}
        for name, activation in self.activations.items():
            # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
            if len(activation.shape) == 4:
                activation_dict[name] = activation.reshape(activation.shape[0], -1).numpy()
            else:
                activation_dict[name] = activation.numpy()

        np.savez_compressed(
            str(output_path.with_suffix('.npz')),
            **activation_dict
        )

        if metadata:
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Remove hooks on exit."""
        self.remove_hooks()


def flatten_activations(activations: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Flatten all layer activations to a single vector per sample.

    Args:
        activations: Dict of layer_name -> activation array (B, C*H*W)

    Returns:
        Flattened array of shape (B, total_features)
    """
    all_features = []
    for layer_name in sorted(activations.keys()):
        act = activations[layer_name]
        if len(act.shape) > 2:
            # Flatten everything except batch dim
            act = act.reshape(act.shape[0], -1)
        all_features.append(act)

    return np.concatenate(all_features, axis=1)


def load_activations(activation_path: Path):
    """
    Load activations and metadata from disk.

    Returns:
        (activations_dict, metadata_dict)
    """
    activation_path = Path(activation_path)

    # Load activations
    data = np.load(str(activation_path.with_suffix('.npz')))
    activations = {key: data[key] for key in data.keys()}

    # Load metadata if exists
    metadata = {}
    metadata_path = activation_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return activations, metadata
