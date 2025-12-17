"""
Compute FID between generated samples and real images using clean-fid.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Add parent to path for DMD2 imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def generate_samples_for_fid(
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 50000,
    batch_size: int = 64,
    num_steps: int = 10,
    guidance_scale: float = 1.5,
    seed: int = 42,
    device: str = "cuda"
):
    """Generate samples from checkpoint for FID evaluation."""
    from main.edm.sample_edm_multistep import create_generator, generate_samples

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create argument namespace for create_generator
    class Args:
        resolution = 64
        label_dim = 1000
        label_dropout = 0.1  # Match training config

    args = Args()

    print(f"Loading checkpoint: {checkpoint_path}")
    generator = create_generator(checkpoint_path, args)
    generator = generator.to(device)
    generator.eval()

    print(f"Generating {num_samples} samples with {num_steps} steps, CFG={guidance_scale}")

    images = generate_samples(
        generator=generator,
        num_samples=num_samples,
        label_dim=1000,
        resolution=64,
        num_steps=num_steps,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        device=device,
        seed=seed
    )

    print(f"Saving {len(images)} images to {output_path}")
    for i, img in enumerate(tqdm(images, desc="Saving")):
        img_path = output_path / f"gen_{i:06d}.png"
        Image.fromarray(img).save(img_path)

    return len(images)


def compute_fid(real_dir: str, gen_dir: str, batch_size: int = 64, device: str = "cuda"):
    """Compute FID using clean-fid."""
    try:
        from cleanfid import fid
    except ImportError:
        print("clean-fid not installed. Installing...")
        os.system("pip install clean-fid")
        from cleanfid import fid

    print(f"\nComputing FID...")
    print(f"  Real images: {real_dir}")
    print(f"  Generated images: {gen_dir}")

    fid_score = fid.compute_fid(
        real_dir,
        gen_dir,
        mode="clean",
        batch_size=batch_size,
        device=torch.device(device)
    )

    print(f"\nFID Score: {fid_score:.2f}")
    return fid_score


def main():
    parser = argparse.ArgumentParser(description="Generate samples and compute FID")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--gen_dir", type=str, default="generated_samples", help="Output dir for generated images")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_steps", type=int, default=10, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation, only compute FID")

    args = parser.parse_args()

    if not args.skip_generation:
        generate_samples_for_fid(
            checkpoint_path=args.checkpoint,
            output_dir=args.gen_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            device=args.device
        )

    fid_score = compute_fid(
        real_dir=args.real_dir,
        gen_dir=args.gen_dir,
        batch_size=args.batch_size,
        device=args.device
    )

    # Save result
    result_path = Path(args.gen_dir) / "fid_result.txt"
    with open(result_path, "w") as f:
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Num samples: {args.num_samples}\n")
        f.write(f"Steps: {args.num_steps}\n")
        f.write(f"CFG scale: {args.guidance_scale}\n")

    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
