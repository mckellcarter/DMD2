"""
Generate new images from fixed activations using DMD2 models.
Supports both single-step and multi-step generation.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import os

from activation_masking import ActivationMask, unflatten_activation


def tensor_to_uint8_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor in [-1, 1] range to uint8 image tensor in [0, 255].

    Args:
        tensor: Image tensor (B, C, H, W) in range [-1, 1]

    Returns:
        uint8 tensor (B, H, W, C) in range [0, 255]
    """
    images = ((tensor + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu()
    return images


def get_denoising_sigmas(num_steps, sigma_max, sigma_min, rho=7.0):
    """
    Generate Karras sigma schedule for multi-step denoising.
    Returns sigmas in descending order (large to small).
    """
    ramp = torch.linspace(0, 1, num_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def create_imagenet_generator(checkpoint_path, device='cuda', label_dropout=0.0):
    """
    Create ImageNet DMD2 generator model.

    Args:
        checkpoint_path: Path to checkpoint (.pth file, .safetensors, or directory)
        device: Device to load model on
        label_dropout: Label dropout rate (use >0 for CFG-trained models)

    Returns:
        Loaded generator model
    """
    # Use shared model utilities
    from model_utils import create_imagenet_generator as _create_generator
    return _create_generator(checkpoint_path, device=device, label_dropout=label_dropout)


@torch.no_grad()
def generate_with_masked_activation(
    generator,
    activation_mask: ActivationMask,
    class_label: int = None,
    conditioning_sigma: float = 0.1,
    num_samples: int = 1,
    resolution: int = 64,
    device: str = 'cuda',
    seed: int = None
):
    """
    Generate images with fixed activation at specified layer.

    Args:
        generator: DMD2 generator model (EDMPrecond)
        activation_mask: ActivationMask with masks set
        class_label: ImageNet class label (0-999), random if None
        conditioning_sigma: Noise level for conditioning
        num_samples: Number of images to generate
        resolution: Image resolution (64 for ImageNet)
        device: Device to generate on
        seed: Random seed for noise

    Returns:
        Generated images as tensor (N, H, W, 3) uint8
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random labels if not specified
    if class_label < 0: #uniform weights across labels
        random_labels = torch.tensor([-1], device=device).repeat((num_samples,))
    elif class_label is None:
        random_labels = torch.randint(0, 1000, (num_samples,), device=device)
    else:
        random_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)

    # Create one-hot labels
    if class_label < 0: 
        #uniform weight across labels
        one_hot_labels = torch.ones((1000), dtype=torch.float16, device=device).repeat((num_samples, )) * (1/1000)
    else: 
        one_hot_labels = torch.eye(1000, device=device)[random_labels]

    # Generate noise
    noise = torch.randn(num_samples, 3, resolution, resolution, device=device)

    # Generate with masked activations
    timesteps = torch.ones(num_samples, device=device) * conditioning_sigma

    
    generated_images = generator(
        noise * conditioning_sigma,
        timesteps,
        one_hot_labels
    )

    # Convert to uint8 images
    images = ((generated_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu()

    return images, random_labels.cpu()


@torch.no_grad()
def generate_with_masked_activation_multistep(
    generator,
    activation_mask: ActivationMask,
    class_label: int = None,
    num_steps: int = 4,
    mask_steps: int = None,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    rho: float = 7.0,
    guidance_scale: float = 1.0,
    stochastic: bool = True,
    num_samples: int = 1,
    resolution: int = 64,
    device: str = 'cuda',
    seed: int = None,
    extract_layers: list = None,
    return_trajectory: bool = False,
    return_intermediates: bool = False
):
    """
    Generate images with fixed activation using multi-step denoising.

    Args:
        generator: DMD2 generator model (EDMPrecond)
        activation_mask: ActivationMask with masks set (hooks should be registered)
        class_label: ImageNet class label (0-999), random if None, -1 for uniform
        num_steps: Number of denoising steps (e.g., 4, 10)
        mask_steps: Number of steps to apply activation mask (default=num_steps).
                    Use mask_steps=1 to only constrain the first step.
        sigma_max: Maximum sigma for noise schedule
        sigma_min: Minimum sigma for noise schedule
        rho: Karras schedule parameter
        guidance_scale: CFG scale (0=uncond, 1=class, >1=amplify, <0=anti-class)
        stochastic: Whether to add noise between steps
        num_samples: Number of images to generate
        resolution: Image resolution (64 for ImageNet)
        device: Device to generate on
        seed: Random seed for noise
        extract_layers: List of layer names to extract for trajectory (concatenated)
        return_trajectory: If True, return activations at each step
        return_intermediates: If True, return intermediate images at each step

    Returns:
        Generated images as tensor (N, H, W, 3) uint8, labels
        If return_trajectory=True: also returns list of activations per step
        If return_intermediates=True: also returns list of intermediate images per step
    """
    # Default: mask all steps (backward compatible)
    if mask_steps is None:
        mask_steps = num_steps
    if seed is not None:
        torch.manual_seed(seed)

    # Set up trajectory extraction if requested
    trajectory_activations = []
    intermediate_images = []
    extractor = None
    if return_trajectory and extract_layers:
        from extract_activations import ActivationExtractor
        extractor = ActivationExtractor("imagenet")
        extractor.register_hooks(generator, extract_layers)

    # Generate random labels if not specified
    if class_label is not None and class_label < 0:
        # Uniform weights across labels
        random_labels = torch.tensor([-1], device=device).repeat((num_samples,))
        one_hot_labels = torch.ones((num_samples, 1000), dtype=torch.float32, device=device) * (1/1000)
        uncond_labels = one_hot_labels.clone()  # Same for CFG
    elif class_label is None:
        random_labels = torch.randint(0, 1000, (num_samples,), device=device)
        one_hot_labels = torch.eye(1000, device=device)[random_labels]
        uncond_labels = torch.zeros_like(one_hot_labels)
    else:
        random_labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
        one_hot_labels = torch.eye(1000, device=device)[random_labels]
        uncond_labels = torch.zeros_like(one_hot_labels)

    # Generate sigma schedule (descending: large to small)
    sigmas = get_denoising_sigmas(num_steps, sigma_max, sigma_min, rho).to(device)

    # Start from pure noise scaled by sigma_max
    noise = torch.randn(num_samples, 3, resolution, resolution, device=device)
    x = noise * sigma_max

    # Iterative denoising
    for i, sigma in enumerate(sigmas):
        # Remove activation mask after mask_steps
        if i == mask_steps and activation_mask is not None:
            activation_mask.remove_hooks()

        sigma_tensor = torch.ones(num_samples, device=device) * sigma

        if guidance_scale != 1.0:
            # Classifier-free guidance (CFG):
            #   scale < 0.0: anti-class (drive away from specified class)
            #   scale = 0.0: pure unconditional
            #   scale < 1.0: blend toward unconditional (reduced class influence)
            #   scale > 1.0: amplify class conditioning
            #
            # Full spectrum:
            #   -2.0  strongly anti-class
            #   -1.0  moderate anti-class
            #    0.0  pure unconditional
            #    0.5  weak class influence
            #    1.0  pure class conditioning (see else branch)
            #    2.0  amplified class
            pred_cond = generator(x, sigma_tensor, one_hot_labels)
            pred_uncond = generator(x, sigma_tensor, uncond_labels)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            # scale = 1.0: pure class conditioning, skip extra forward pass
            pred = generator(x, sigma_tensor, one_hot_labels)

        # Extract activation for trajectory if enabled
        # Flow: x (noisy input) -> activations (captured here) -> pred (denoised output)
        # x at step 0: noise * sigma_max; step i>0: pred[i-1] + sigma[i] * noise
        if extractor is not None:
            acts = extractor.get_activations()
            # Concatenate all layers in sorted order (same as UMAP training)
            layer_acts = []
            for layer_name in sorted(extract_layers):
                act = acts.get(layer_name)
                if act is not None:
                    # Flatten spatial dims: (B, C, H, W) -> (B, C*H*W)
                    if len(act.shape) == 4:
                        B, C, H, W = act.shape
                        act = act.reshape(B, -1)
                    layer_acts.append(act.numpy())
            if layer_acts:
                # Concatenate along feature dimension
                concat_act = np.concatenate(layer_acts, axis=1)
                trajectory_activations.append(concat_act)
            extractor.clear_activations()

        # Capture intermediate image if requested
        # Captures pred (denoised output); to capture noisy input use x instead
        if return_intermediates:
            intermediate_images.append(tensor_to_uint8_image(pred))

        # Transition to next step
        if i < len(sigmas) - 1:
            next_sigma = sigmas[i + 1]
            if stochastic:
                # Stochastic sampling - add noise for next step
                x = pred + next_sigma * torch.randn_like(pred)
            else:
                # Deterministic - just use prediction as starting point
                x = pred
        else:
            # Final step - use prediction directly
            x = pred

    # Cleanup extractor hooks
    if extractor is not None:
        extractor.remove_hooks()

    # Convert to uint8 images
    images = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu()

    # Build return tuple based on requested outputs
    result = [images, random_labels.cpu()]
    if return_trajectory:
        result.append(trajectory_activations)
    if return_intermediates:
        result.append(intermediate_images)

    return tuple(result) if len(result) > 2 else (result[0], result[1])


def save_generated_sample(
    image: torch.Tensor,
    activations: dict,
    metadata: dict,
    output_dir: Path,
    sample_id: str
):
    """
    Save generated image, activations, and metadata.

    Args:
        image: (H, W, 3) uint8 tensor
        activations: Dict of layer_name -> activation tensor
        metadata: Dict with sample info
        output_dir: Root output directory
        sample_id: Unique sample identifier
    """
    output_dir = Path(output_dir)

    # Save image
    image_dir = output_dir / "images" / metadata.get("model", "imagenet")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{sample_id}.png"

    image_pil = Image.fromarray(image.numpy())
    image_pil.save(image_path)

    # Save activations
    activation_dir = output_dir / "activations" / metadata.get("model", "imagenet")
    activation_dir.mkdir(parents=True, exist_ok=True)
    activation_path = activation_dir / f"{sample_id}"

    # Convert activations to numpy and save
    activation_dict = {}
    for name, activation in activations.items():
        if isinstance(activation, torch.Tensor):
            # Flatten spatial dimensions
            if len(activation.shape) == 4:
                B, C, H, W = activation.shape
                activation_dict[name] = activation.reshape(B, -1).cpu().numpy()
            else:
                activation_dict[name] = activation.cpu().numpy()
        else:
            activation_dict[name] = activation

    np.savez_compressed(str(activation_path.with_suffix('.npz')), **activation_dict)

    # Save metadata
    with open(activation_path.with_suffix('.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'sample_id': sample_id,
        'image_path': f"images/{metadata.get('model', 'imagenet')}/{sample_id}.png",
        **metadata
    }


def infer_activation_shape(generator, layer_name: str, device: str = 'cuda'):
    """
    Infer the spatial shape of a layer's activation by running a dummy forward pass.

    Args:
        generator: DMD2 generator model
        layer_name: Name of layer to infer shape for
        device: Device to run inference on

    Returns:
        (C, H, W) shape tuple
    """
    from extract_activations import ActivationExtractor

    # Create dummy input
    dummy_noise = torch.randn(1, 3, 64, 64, device=device)
    dummy_label = torch.zeros(1, 1000, device=device)
    dummy_label[0, 0] = 1.0
    dummy_sigma = torch.ones(1, device=device) * 80.0

    # Extract activation shape
    with ActivationExtractor("imagenet") as extractor:
        extractor.register_hooks(generator, [layer_name])
        _ = generator(dummy_noise * 80.0, dummy_sigma, dummy_label)
        activations = extractor.get_activations()

    if layer_name not in activations:
        raise ValueError(f"Layer {layer_name} not found during inference")

    shape = activations[layer_name].shape  # (B, C, H, W)
    return tuple(shape[1:])  # (C, H, W)
