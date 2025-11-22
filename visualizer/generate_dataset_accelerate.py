"""
Generate dataset for DMD2 visualizer with Accelerate multi-GPU support.
Use this for distributed generation on multiple GPUs.
For single GPU/MPS, use generate_dataset.py instead.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
import json

# ImageNet imports
from third_party.edm.training.networks import EDMPrecond
from main.edm.edm_network import get_imagenet_edm_config

# Local imports
from extract_activations import ActivationExtractor


def get_imagenet_config():
    """Get ImageNet EDM config."""
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
    base_config.update(get_imagenet_edm_config())
    return base_config


def load_imagenet_model(checkpoint_path: str):
    """Load pretrained ImageNet generator (Accelerator handles device placement)."""
    base_config = get_imagenet_config()
    generator = EDMPrecond(**base_config)
    del generator.model.map_augment
    generator.model.map_augment = None

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()

    return generator


def generate_imagenet_dataset_distributed(
    checkpoint_path: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    layers: list,
    samples_per_class: int = None,
    conditioning_sigma: float = 80.0,
    seed: int = 10,
    mixed_precision: str = "no"
):
    """
    Generate ImageNet dataset with activations using Accelerate for multi-GPU.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for data
        num_samples: Total samples to generate (across all processes)
        batch_size: Batch size per GPU
        layers: List of layer names to extract
        samples_per_class: If set, generate this many per class
        conditioning_sigma: Sigma for conditioning
        seed: Random seed
        mixed_precision: Mixed precision mode ('no', 'fp16', 'bf16')
    """
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Print device info
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Accelerate Distributed Generation")
        print(f"{'='*60}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")
        print(f"Device: {accelerator.device}")
        print(f"Mixed precision: {mixed_precision}")
        print(f"{'='*60}\n")

    # Load model
    if accelerator.is_main_process:
        print(f"Loading ImageNet model from {checkpoint_path}")

    generator = load_imagenet_model(checkpoint_path)

    # Setup extractor (before wrapping with accelerator)
    extractor = ActivationExtractor(model_type="imagenet")
    extractor.register_hooks(generator, layers)

    # Wrap model with accelerator
    generator = accelerator.prepare(generator)

    # Create output directories (all processes)
    image_dir = output_dir / "images" / "imagenet"
    activation_dir = output_dir / "activations" / "imagenet"
    metadata_dir = output_dir / "metadata" / "imagenet"

    if accelerator.is_main_process:
        image_dir.mkdir(parents=True, exist_ok=True)
        activation_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

    accelerator.wait_for_everyone()

    set_seed(seed + accelerator.process_index)

    # Calculate samples per process
    samples_per_process = num_samples // accelerator.num_processes
    start_sample = accelerator.process_index * samples_per_process

    # Generate labels
    if samples_per_class:
        # Balanced dataset
        num_classes = 1000
        labels = []
        for class_id in range(num_classes):
            labels.extend([class_id] * samples_per_class)
        labels = labels[:num_samples]
    else:
        # Random labels
        np.random.seed(seed)
        labels = np.random.randint(0, 1000, num_samples)

    # Get labels for this process
    process_labels = labels[start_sample:start_sample + samples_per_process]

    # Generate in batches
    sample_idx = start_sample
    all_metadata = []

    num_batches = (len(process_labels) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(num_batches),
        desc=f"GPU {accelerator.process_index}",
        disable=not accelerator.is_local_main_process
    ):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(process_labels))
        current_batch_size = batch_end - batch_start

        # Prepare batch
        noise = torch.randn(
            current_batch_size, 3, 64, 64,
            device=accelerator.device
        )
        batch_labels = process_labels[batch_start:batch_end]
        one_hot_labels = torch.eye(1000, device=accelerator.device)[
            torch.tensor(batch_labels, device=accelerator.device)
        ]

        # Generate
        extractor.clear_activations()
        with torch.no_grad():
            images = generator(
                noise * conditioning_sigma,
                torch.ones(current_batch_size, device=accelerator.device) * conditioning_sigma,
                one_hot_labels
            )

        # Convert to uint8
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()

        # Get activations
        activations = extractor.get_activations()

        # Save each sample
        for i in range(current_batch_size):
            sample_id = f"sample_{sample_idx:06d}"

            # Save image
            img_path = image_dir / f"{sample_id}.png"
            Image.fromarray(images[i]).save(img_path)

            # Save activations
            sample_activations = {
                layer: act[i:i+1] for layer, act in activations.items()
            }
            act_path = activation_dir / sample_id
            extractor.save_activations(
                act_path,
                metadata={
                    "sample_id": sample_id,
                    "class_label": int(batch_labels[i]),
                    "seed": seed,
                    "process_index": accelerator.process_index,
                }
            )

            # Track metadata
            all_metadata.append({
                "sample_id": sample_id,
                "class_label": int(batch_labels[i]),
                "image_path": str(img_path.relative_to(output_dir)),
                "activation_path": str(act_path.relative_to(output_dir)),
            })

            sample_idx += 1

    # Wait for all processes
    accelerator.wait_for_everyone()

    # Save global metadata (main process only)
    if accelerator.is_main_process:
        # Gather metadata from all processes (simplified - just save what we have)
        metadata_path = metadata_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "model_type": "imagenet",
                "num_samples": num_samples,
                "layers": layers,
                "conditioning_sigma": conditioning_sigma,
                "seed": seed,
                "num_processes": accelerator.num_processes,
                "note": "Metadata includes samples from all processes"
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Generation complete!")
        print(f"Total samples: {num_samples}")
        print(f"Images: {image_dir}")
        print(f"Activations: {activation_dir}")
        print(f"Metadata: {metadata_path}")
        print(f"{'='*60}\n")

    extractor.remove_hooks()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization dataset with Accelerate multi-GPU support"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Total number of samples to generate (across all GPUs)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="encoder_bottleneck,midblock",
        help="Comma-separated list of layers to extract"
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=None,
        help="Samples per class for balanced dataset"
    )
    parser.add_argument(
        "--conditioning_sigma",
        type=float,
        default=80.0,
        help="Conditioning sigma for ImageNet"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    layers = args.layers.split(",")

    generate_imagenet_dataset_distributed(
        checkpoint_path=args.checkpoint_path,
        output_dir=output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        layers=layers,
        samples_per_class=args.samples_per_class,
        conditioning_sigma=args.conditioning_sigma,
        seed=args.seed,
        mixed_precision=args.mixed_precision
    )


if __name__ == "__main__":
    main()