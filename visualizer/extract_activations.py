"""
Extract UNet activations during inference for visualization.
Supports DhariwalUNet (ImageNet) and UNet2DConditionModel (SDXL/SDv1.5).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Callable
from pathlib import Path
import json


class ActivationExtractor:
    """Hook-based activation extraction for UNet models."""

    def __init__(self, model_type: str = "imagenet"):
        """
        Args:
            model_type: One of ["imagenet", "sdxl", "sdv1.5"]
        """
        self.model_type = model_type
        self.activations = {}
        self.hooks = []

    def _get_hook_fn(self, name: str) -> Callable:
        """Create hook function that stores activations."""
        def hook(module, input, output):
            # Store activation as numpy for memory efficiency
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach().cpu()
        return hook

    def register_imagenet_hooks(self, model, layers: List[str] = None):
        """
        Register hooks for DhariwalUNet (ImageNet).

        Args:
            model: EDMPrecond or DhariwalUNet model
            layers: List of layer names to extract. Options:
                - "encoder_bottleneck": Final encoder output
                - "midblock": Decoder midblock (first decoder layer)
                - "encoder_block_N": Specific encoder block
                - "decoder_block_N": Specific decoder block
        """
        if layers is None:
            # Default: extract encoder bottleneck and midblock
            layers = ["encoder_bottleneck", "midblock"]

        # Access underlying DhariwalUNet
        unet = model.model if hasattr(model, 'model') else model

        for layer_name in layers:
            if layer_name == "encoder_bottleneck":
                # Hook after all encoder blocks
                last_enc_key = list(unet.enc.keys())[-1]
                hook = unet.enc[last_enc_key].register_forward_hook(
                    self._get_hook_fn("encoder_bottleneck")
                )
                self.hooks.append(hook)

            elif layer_name == "midblock":
                # Hook first decoder block (contains bottleneck attention)
                first_dec_key = list(unet.dec.keys())[0]
                hook = unet.dec[first_dec_key].register_forward_hook(
                    self._get_hook_fn("midblock")
                )
                self.hooks.append(hook)

            elif layer_name.startswith("encoder_block_"):
                idx = int(layer_name.split("_")[-1])
                enc_keys = list(unet.enc.keys())
                if idx < len(enc_keys):
                    hook = unet.enc[enc_keys[idx]].register_forward_hook(
                        self._get_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

            elif layer_name.startswith("decoder_block_"):
                idx = int(layer_name.split("_")[-1])
                dec_keys = list(unet.dec.keys())
                if idx < len(dec_keys):
                    hook = unet.dec[dec_keys[idx]].register_forward_hook(
                        self._get_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

    def register_sd_hooks(self, model, layers: List[str] = None):
        """
        Register hooks for UNet2DConditionModel (SDXL/SDv1.5).

        Args:
            model: UNet2DConditionModel from diffusers
            layers: List of layer names. Options:
                - "down_block_N": Downsampling block N
                - "mid_block": Middle block
                - "up_block_N": Upsampling block N
        """
        if layers is None:
            layers = ["mid_block"]

        # Access UNet (may be wrapped)
        unet = model.feedforward_model if hasattr(model, 'feedforward_model') else model

        for layer_name in layers:
            if layer_name == "mid_block":
                hook = unet.mid_block.register_forward_hook(
                    self._get_hook_fn("mid_block")
                )
                self.hooks.append(hook)

            elif layer_name.startswith("down_block_"):
                idx = int(layer_name.split("_")[-1])
                if idx < len(unet.down_blocks):
                    hook = unet.down_blocks[idx].register_forward_hook(
                        self._get_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

            elif layer_name.startswith("up_block_"):
                idx = int(layer_name.split("_")[-1])
                if idx < len(unet.up_blocks):
                    hook = unet.up_blocks[idx].register_forward_hook(
                        self._get_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

    def register_hooks(self, model, layers: List[str] = None):
        """Register hooks based on model type."""
        if self.model_type == "imagenet":
            self.register_imagenet_hooks(model, layers)
        else:  # sdxl or sdv1.5
            self.register_sd_hooks(model, layers)

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
            # Flatten spatial dimensions but keep batch and channel dims
            # Shape: (B, C, H, W) -> (B, C*H*W)
            if len(activation.shape) == 4:
                B, C, H, W = activation.shape
                activation_dict[name] = activation.reshape(B, -1).numpy()
            else:
                activation_dict[name] = activation.numpy()

        np.savez_compressed(
            str(output_path.with_suffix('.npz')),
            **activation_dict
        )

        # Save metadata
        if metadata:
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    return activations, metadata
