"""
Shared model loading utilities for DMD2 visualizer.
Handles different checkpoint formats: .pth, .safetensors, and accelerator directories.
"""

import os
import torch
from pathlib import Path


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Load model state dict from various checkpoint formats.

    Args:
        checkpoint_path: Path to checkpoint file or directory. Supports:
            - .pth files (PyTorch standard)
            - .safetensors files
            - Directories containing model.safetensors or pytorch_model.bin
        device: Device to map tensors to (default: "cpu")

    Returns:
        state_dict: Model state dictionary

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist or no valid model file found
    """
    checkpoint_path = str(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if os.path.isdir(checkpoint_path):
        # Accelerator checkpoint directory
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
            print(f"Loaded checkpoint from {safetensors_path}")
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location=device)
            print(f"Loaded checkpoint from {pytorch_path}")
        else:
            raise FileNotFoundError(
                f"No model file found in directory {checkpoint_path}. "
                f"Expected model.safetensors or pytorch_model.bin"
            )
    elif checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        # Standard .pth checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded checkpoint from {checkpoint_path}")

    return state_dict


def create_imagenet_generator(checkpoint_path: str, device: str = 'cuda', label_dropout: float = 0.0):
    """
    Create ImageNet DMD2 generator model.

    Args:
        checkpoint_path: Path to checkpoint (.pth file, .safetensors, or directory)
        device: Device to load model on
        label_dropout: Label dropout rate (use >0 for CFG-trained models)

    Returns:
        Loaded generator model in eval mode
    """
    from third_party.edm.training.networks import EDMPrecond
    from main.edm.edm_network import get_imagenet_edm_config

    base_config = {
        "img_resolution": 64,
        "img_channels": 3,
        "label_dim": 1000,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet"
    }
    base_config.update(get_imagenet_edm_config(label_dropout=label_dropout))

    generator = EDMPrecond(**base_config)
    del generator.model.map_augment
    generator.model.map_augment = None

    # Load checkpoint
    state_dict = load_checkpoint(checkpoint_path, device="cpu")
    generator.load_state_dict(state_dict, strict=True)

    generator = generator.to(device)
    generator.eval()

    return generator
