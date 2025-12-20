"""
Activation masking for DMD2 models.
Allows holding specific layer outputs constant during generation.
"""

import torch
from typing import Dict, Optional


class ActivationMask:
    """Hook-based activation masking for UNet models."""

    def __init__(self, model_type: str = "imagenet"):
        """
        Args:
            model_type: One of ["imagenet", "sdxl", "sdv1.5"]
        """
        self.model_type = model_type
        self.masks = {}  # layer_name -> fixed activation tensor
        self.hooks = []

    def _get_masking_hook_fn(self, name: str):
        """Create hook function that replaces output with fixed activation."""
        def hook(module, input, output):
            if name in self.masks:
                # Replace output with fixed activation
                # Handle both tensor and tuple outputs
                if isinstance(output, tuple):
                    # Replace first element (main activation)
                    masked = self.masks[name].to(output[0].device, output[0].dtype)
                    # Expand to batch size if needed
                    if masked.shape[0] == 1 and output[0].shape[0] > 1:
                        masked = masked.expand(output[0].shape[0], -1, -1, -1)
                    return (masked,) + output[1:]
                else:
                    masked = self.masks[name].to(output.device, output.dtype)
                    # Expand to batch size if needed
                    if masked.shape[0] == 1 and output.shape[0] > 1:
                        masked = masked.expand(output.shape[0], -1, -1, -1)
                    return masked
            return output
        return hook

    def set_mask(self, layer_name: str, activation: torch.Tensor):
        """
        Set a fixed activation for a specific layer.

        Args:
            layer_name: Name of the layer to mask
            activation: Tensor to use as fixed output (will be moved to appropriate device)
        """
        # Store on CPU to save memory, will move to device during hook
        self.masks[layer_name] = activation.cpu()

    def clear_masks(self):
        """Clear all activation masks."""
        self.masks = {}

    def register_imagenet_hooks(self, model, layers: list = None):
        """
        Register masking hooks for DhariwalUNet (ImageNet).

        Args:
            model: EDMPrecond or DhariwalUNet model
            layers: List of layer names to mask (must match keys in self.masks)
        """
        if layers is None:
            layers = list(self.masks.keys())

        # Access underlying DhariwalUNet
        unet = model.model if hasattr(model, 'model') else model

        for layer_name in layers:
            if layer_name == "encoder_bottleneck":
                last_enc_key = list(unet.enc.keys())[-1]
                hook = unet.enc[last_enc_key].register_forward_hook(
                    self._get_masking_hook_fn("encoder_bottleneck")
                )
                self.hooks.append(hook)

            elif layer_name == "midblock":
                first_dec_key = list(unet.dec.keys())[0]
                hook = unet.dec[first_dec_key].register_forward_hook(
                    self._get_masking_hook_fn("midblock")
                )
                self.hooks.append(hook)

            elif layer_name.startswith("encoder_block_"):
                idx = int(layer_name.split("_")[-1])
                enc_keys = list(unet.enc.keys())
                if idx < len(enc_keys):
                    hook = unet.enc[enc_keys[idx]].register_forward_hook(
                        self._get_masking_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

            elif layer_name.startswith("decoder_block_"):
                idx = int(layer_name.split("_")[-1])
                dec_keys = list(unet.dec.keys())
                if idx < len(dec_keys):
                    hook = unet.dec[dec_keys[idx]].register_forward_hook(
                        self._get_masking_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

    def register_sd_hooks(self, model, layers: list = None):
        """
        Register masking hooks for UNet2DConditionModel (SDXL/SDv1.5).

        Args:
            model: UNet2DConditionModel from diffusers
            layers: List of layer names to mask
        """
        if layers is None:
            layers = list(self.masks.keys())

        # Access UNet (may be wrapped)
        unet = model.feedforward_model if hasattr(model, 'feedforward_model') else model

        for layer_name in layers:
            if layer_name == "mid_block":
                hook = unet.mid_block.register_forward_hook(
                    self._get_masking_hook_fn("mid_block")
                )
                self.hooks.append(hook)

            elif layer_name.startswith("down_block_"):
                idx = int(layer_name.split("_")[-1])
                if idx < len(unet.down_blocks):
                    hook = unet.down_blocks[idx].register_forward_hook(
                        self._get_masking_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

            elif layer_name.startswith("up_block_"):
                idx = int(layer_name.split("_")[-1])
                if idx < len(unet.up_blocks):
                    hook = unet.up_blocks[idx].register_forward_hook(
                        self._get_masking_hook_fn(layer_name)
                    )
                    self.hooks.append(hook)

    def register_hooks(self, model, layers: list = None):
        """Register masking hooks based on model type."""
        if self.model_type == "imagenet":
            self.register_imagenet_hooks(model, layers)
        else:  # sdxl or sdv1.5
            self.register_sd_hooks(model, layers)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove hooks on exit."""
        self.remove_hooks()


def load_activation_from_npz(npz_path, layer_name: str) -> torch.Tensor:
    """
    Load a specific layer's activation from saved NPZ file.

    Args:
        npz_path: Path to .npz file
        layer_name: Name of layer to load

    Returns:
        Activation tensor (1, C*H*W) - will need reshaping
    """
    import numpy as np
    data = np.load(npz_path)
    if layer_name not in data:
        raise ValueError(f"Layer '{layer_name}' not found in {npz_path}. Available: {list(data.keys())}")

    activation = torch.from_numpy(data[layer_name])
    # Add batch dimension if not present
    if len(activation.shape) == 1:
        activation = activation.unsqueeze(0)
    return activation


def unflatten_activation(flat_activation: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Reshape flattened activation to original spatial dimensions.

    Args:
        flat_activation: (1, C*H*W) tensor
        target_shape: (C, H, W) original shape

    Returns:
        Reshaped tensor (1, C, H, W)
    """
    if len(flat_activation.shape) == 1:
        flat_activation = flat_activation.unsqueeze(0)

    B = flat_activation.shape[0]
    C, H, W = target_shape
    return flat_activation.reshape(B, C, H, W)
