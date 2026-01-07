"""
Multi-step sampling with CFG for EDM models.

This script provides standalone sampling functionality for 10-step
(or configurable) denoising with classifier-free guidance.
"""

import torch
import numpy as np
from PIL import Image
import argparse
import os


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    """
    Generate Karras sigma schedule.

    Args:
        n: Number of steps
        sigma_min: Minimum sigma
        sigma_max: Maximum sigma
        rho: Schedule parameter (default 7.0)

    Returns:
        Tensor of sigmas in descending order
    """
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


@torch.no_grad()
def sample_multistep_cfg(
    generator,
    noise,
    labels,
    num_steps=10,
    sigma_max=80.0,
    sigma_min=0.002,
    rho=7.0,
    guidance_scale=1.5,
    stochastic=True
):
    """
    Multi-step sampling with classifier-free guidance.

    Args:
        generator: Trained generator model (EDMPrecond)
        noise: Initial noise tensor [B, 3, H, W]
        labels: One-hot class labels [B, num_classes]
        num_steps: Number of denoising steps
        sigma_max: Maximum sigma (start of denoising)
        sigma_min: Minimum sigma (end of denoising)
        rho: Karras schedule parameter
        guidance_scale: CFG scale (1.0 = no guidance, >1.0 = guided)
        stochastic: Whether to add noise between steps

    Returns:
        Generated images [B, 3, H, W] in range [-1, 1]
    """
    batch_size = noise.shape[0]
    device = noise.device

    # Generate sigma schedule
    sigmas = get_sigmas_karras(num_steps, sigma_min, sigma_max, rho).to(device)

    # Start from pure noise scaled by sigma_max
    x = noise * sigma_max

    # Unconditional labels (zeros) for CFG
    uncond_labels = torch.zeros_like(labels)

    for i, sigma in enumerate(sigmas):
        sigma_tensor = torch.ones(batch_size, device=device) * sigma

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
            pred_cond = generator(x, sigma_tensor, labels)
            pred_uncond = generator(x, sigma_tensor, uncond_labels)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            # scale = 1.0: pure class conditioning, skip extra forward pass
            pred = generator(x, sigma_tensor, labels)

        # Transition to next step
        if i < len(sigmas) - 1:
            next_sigma = sigmas[i + 1]
            if stochastic:
                # Stochastic sampling - add fresh noise at next sigma level
                x = pred + next_sigma * torch.randn_like(pred)
            else:
                # Deterministic - direct prediction propagation
                x = pred
        else:
            # Final step - use prediction directly
            x = pred

    return x


@torch.no_grad()
def sample_multistep_deterministic(
    generator,
    noise,
    labels,
    num_steps=10,
    sigma_max=80.0,
    sigma_min=0.002,
    rho=7.0
):
    """
    Deterministic multi-step sampling (no CFG, no stochasticity).
    Useful for evaluation consistency.
    """
    return sample_multistep_cfg(
        generator=generator,
        noise=noise,
        labels=labels,
        num_steps=num_steps,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rho=rho,
        guidance_scale=1.0,
        stochastic=False
    )


@torch.no_grad()
def generate_samples(
    generator,
    num_samples,
    label_dim=1000,
    resolution=64,
    num_steps=10,
    sigma_max=80.0,
    sigma_min=0.002,
    rho=7.0,
    guidance_scale=1.5,
    batch_size=64,
    device='cuda',
    seed=None
):
    """
    Generate multiple samples with CFG.

    Args:
        generator: Trained generator model
        num_samples: Total number of samples to generate
        label_dim: Number of classes
        resolution: Image resolution
        num_steps: Number of denoising steps
        sigma_max: Maximum sigma
        sigma_min: Minimum sigma
        rho: Karras schedule parameter
        guidance_scale: CFG scale
        batch_size: Batch size for generation
        device: Device to use
        seed: Random seed (optional)

    Returns:
        Generated images as numpy array [N, H, W, 3] in uint8
    """
    if seed is not None:
        torch.manual_seed(seed)

    generator.eval()
    all_images = []

    # Pre-generate all labels to ensure uniform distribution
    all_labels = torch.arange(0, num_samples, device=device, dtype=torch.long) % label_dim
    eye_matrix = torch.eye(label_dim, device=device)

    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        # Generate noise
        noise = torch.randn(
            current_batch_size, 3, resolution, resolution,
            device=device
        )

        # Get labels for this batch
        batch_labels = all_labels[start_idx:end_idx]
        one_hot_labels = eye_matrix[batch_labels]

        # Sample
        images = sample_multistep_cfg(
            generator=generator,
            noise=noise,
            labels=one_hot_labels,
            num_steps=num_steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rho=rho,
            guidance_scale=guidance_scale,
            stochastic=True
        )

        # Convert to uint8 images
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        all_images.append(images)

    return np.concatenate(all_images, axis=0)[:num_samples]


def save_image_grid(images, output_path, grid_size=8):
    """
    Save images as a grid.

    Args:
        images: numpy array [N, H, W, 3] in uint8
        output_path: Path to save the grid
        grid_size: Number of images per row/column
    """
    n = min(len(images), grid_size * grid_size)
    h, w, c = images.shape[1:]

    grid = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)

    for idx in range(n):
        i = idx // grid_size
        j = idx % grid_size
        grid[i*h:(i+1)*h, j*w:(j+1)*w] = images[idx]

    Image.fromarray(grid).save(output_path)
    print(f"Saved grid to {output_path}")


def create_generator(checkpoint_path, args):
    """Create generator model from checkpoint."""
    from third_party.edm.training.networks import EDMPrecond
    from main.edm.edm_network import get_imagenet_edm_config

    base_config = {
        "img_resolution": args.resolution,
        "img_channels": 3,
        "label_dim": args.label_dim,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet"
    }
    base_config.update(get_imagenet_edm_config(label_dropout=args.label_dropout))

    generator = EDMPrecond(**base_config)
    del generator.model.map_augment
    generator.model.map_augment = None

    # Handle different checkpoint formats
    if os.path.isdir(checkpoint_path):
        # Accelerator checkpoint directory
        safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
        pytorch_path = os.path.join(checkpoint_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")
    elif checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    print(generator.load_state_dict(state_dict, strict=True))

    return generator


def main():
    parser = argparse.ArgumentParser(description="Multi-step sampling with CFG for EDM models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--resolution", type=int, default=64, help="Image resolution")
    parser.add_argument("--label_dim", type=int, default=1000, help="Number of classes")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Maximum sigma")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Minimum sigma")
    parser.add_argument("--rho", type=float, default=7.0, help="Karras schedule parameter")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="CFG guidance scale")
    parser.add_argument("--label_dropout", type=float, default=0.0, help="Label dropout (for model config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--save_grid", action="store_true", help="Save as image grid")
    parser.add_argument("--save_individual", action="store_true", help="Save individual images")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create generator
    generator = create_generator(args.checkpoint, args)
    generator = generator.to(args.device)
    generator.eval()

    print(f"Generating {args.num_samples} samples with {args.num_steps} steps, CFG scale {args.guidance_scale}")

    # Generate samples
    images = generate_samples(
        generator=generator,
        num_samples=args.num_samples,
        label_dim=args.label_dim,
        resolution=args.resolution,
        num_steps=args.num_steps,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        rho=args.rho,
        guidance_scale=args.guidance_scale,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed
    )

    print(f"Generated {len(images)} images")

    # Save outputs
    if args.save_grid:
        grid_path = os.path.join(args.output_dir, f"grid_cfg{args.guidance_scale}_steps{args.num_steps}.png")
        save_image_grid(images, grid_path)

    if args.save_individual:
        for i, img in enumerate(images):
            img_path = os.path.join(args.output_dir, f"sample_{i:05d}.png")
            Image.fromarray(img).save(img_path)
        print(f"Saved {len(images)} individual images to {args.output_dir}")

    # Save as npz for FID evaluation
    npz_path = os.path.join(args.output_dir, f"samples_cfg{args.guidance_scale}_steps{args.num_steps}.npz")
    np.savez(npz_path, images=images)
    print(f"Saved samples to {npz_path}")


if __name__ == "__main__":
    main()
