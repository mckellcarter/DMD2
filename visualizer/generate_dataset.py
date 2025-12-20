"""
Generate dataset for DMD2 visualizer.
Creates images + activations for ImageNet/SDXL/SDv1.5 models.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from accelerate.utils import set_seed
import json

# ImageNet imports
from third_party.edm.training.networks import EDMPrecond
from main.edm.edm_network import get_imagenet_edm_config

# Local imports
from extract_activations import ActivationExtractor
from device_utils import get_device, get_device_info, move_to_device


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


def load_imagenet_model(checkpoint_path: str, device: str = "cuda"):
    """Load pretrained ImageNet generator."""
    from model_utils import load_checkpoint

    base_config = get_imagenet_config()
    generator = EDMPrecond(**base_config)
    del generator.model.map_augment
    generator.model.map_augment = None

    state_dict = load_checkpoint(checkpoint_path, device="cpu")
    generator.load_state_dict(state_dict, strict=True)
    generator = move_to_device(generator, device)
    generator.eval()

    return generator


def generate_imagenet_dataset(
    checkpoint_path: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    layers: list,
    samples_per_class: int = None,
    conditioning_sigma: float = 80.0,
    seed: int = 10,
    device: str = None
):
    """
    Generate ImageNet dataset with activations.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for data
        num_samples: Total samples to generate
        batch_size: Batch size for generation
        layers: List of layer names to extract
        samples_per_class: If set, generate this many per class
        conditioning_sigma: Sigma for conditioning
        seed: Random seed
        device: Device to use (auto-detect if None)
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()

    device_info = get_device_info(device)
    print(f"\nDevice: {device_info['device']} ({device_info['device_name']})")
    if device_info['memory_allocated'] != 'N/A':
        print(f"GPU Memory: {device_info['memory_allocated']:.2f} GB allocated")

    print(f"\nLoading ImageNet model from {checkpoint_path}")
    generator = load_imagenet_model(checkpoint_path, device)

    # Setup extractor
    extractor = ActivationExtractor(model_type="imagenet")
    extractor.register_hooks(generator, layers)

    # Create output directories
    image_dir = output_dir / "images" / "imagenet"
    activation_dir = output_dir / "activations" / "imagenet"
    metadata_dir = output_dir / "metadata" / "imagenet"
    image_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing dataset
    metadata_path = metadata_dir / "dataset_info.json"
    existing_metadata = []
    start_sample_idx = 0

    if metadata_path.exists():
        print(f"\nFound existing dataset at {metadata_path}")
        with open(metadata_path, 'r') as f:
            existing_data = json.load(f)
            existing_metadata = existing_data.get('samples', [])
            start_sample_idx = len(existing_metadata)
        print(f"Resuming from sample {start_sample_idx} (already have {start_sample_idx} samples)")

        if start_sample_idx >= num_samples:
            print(f"Dataset already has {start_sample_idx} samples (>= {num_samples} requested). Nothing to generate.")
            return
    else:
        print(f"\nStarting new dataset")

    set_seed(seed)

    # Generate labels (always generate full sequence to maintain consistency with seed)
    if samples_per_class:
        # Balanced dataset - repeat classes if needed
        num_classes = 1000
        labels = []
        while len(labels) < num_samples:
            for class_id in range(num_classes):
                if len(labels) >= num_samples:
                    break
                labels.extend([class_id] * samples_per_class)
        labels = labels[:num_samples]
        print(f"Generated {len(labels)} labels with {samples_per_class} samples per class (cycling through {num_classes} classes)")
    else:
        # Random labels
        labels = np.random.randint(0, 1000, num_samples)

    # Skip already-generated samples
    if start_sample_idx > 0:
        print(f"Skipping first {start_sample_idx} samples (already generated)")
        labels = labels[start_sample_idx:]
        samples_to_generate = num_samples - start_sample_idx
    else:
        samples_to_generate = num_samples

    # Generate in batches
    sample_idx = start_sample_idx
    all_metadata = existing_metadata.copy()

    num_batches = (samples_to_generate + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, samples_to_generate)
        current_batch_size = end_idx - start_idx

        # Prepare batch
        noise = torch.randn(
            current_batch_size, 3, 64, 64,
            device=device
        )
        batch_labels = labels[start_idx:end_idx]

        # Ensure batch_labels is a tensor with correct shape
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
        one_hot_labels = torch.eye(1000, device=device)[batch_labels_tensor]

        # Ensure all tensors have matching batch dimension
        assert noise.shape[0] == current_batch_size, f"Noise batch size mismatch: {noise.shape[0]} vs {current_batch_size}"
        assert one_hot_labels.shape[0] == current_batch_size, f"Labels batch size mismatch: {one_hot_labels.shape[0]} vs {current_batch_size}"

        # Generate
        extractor.clear_activations()
        with torch.no_grad():
            sigma = torch.ones(current_batch_size, device=device) * conditioning_sigma
            images = generator(
                noise * conditioning_sigma,
                sigma,
                one_hot_labels
            )

        # Convert to uint8
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()

        # Get activations
        activations = extractor.get_activations()

        # Save batch activations once
        batch_id = f"batch_{batch_idx:06d}"
        batch_act_path = activation_dir / batch_id
        extractor.save_activations(batch_act_path)

        # Save individual images and track metadata
        for i in range(current_batch_size):
            sample_id = f"sample_{sample_idx:06d}"

            # Save image
            img_path = image_dir / f"{sample_id}.png"
            Image.fromarray(images[i]).save(img_path)

            # Track metadata with batch info
            all_metadata.append({
                "sample_id": sample_id,
                "class_label": int(batch_labels[i]),
                "image_path": str(img_path.relative_to(output_dir)),
                "activation_path": str(batch_act_path.relative_to(output_dir)),
                "batch_index": i,
            })

            sample_idx += 1

    # Load ImageNet class labels
    class_labels_path = Path(__file__).parent / "data" / "imagenet_class_labels.json"
    with open(class_labels_path, 'r') as f:
        class_labels = json.load(f)

    # Save global metadata
    metadata_path = metadata_dir / "dataset_info.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "model_type": "imagenet",
            "num_samples": num_samples,
            "layers": layers,
            "conditioning_sigma": conditioning_sigma,
            "seed": seed,
            "class_labels": class_labels,
            "samples": all_metadata
        }, f, indent=2)

    print(f"\nGenerated {sample_idx} samples")
    print(f"Images: {image_dir}")
    print(f"Activations: {activation_dir}")
    print(f"Metadata: {metadata_path}")

    extractor.remove_hooks()


def generate_sdxl_dataset(
    checkpoint_path: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    layers: list,
    prompt_file: str = None,
    num_denoising_steps: int = 4,
    seed: int = 10,
    device: str = None
):
    """
    Generate SDXL dataset with activations.

    TODO: Implement when SDXL inference pipeline is needed.
    This requires loading the full SDXL pipeline with text encoders.
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()

    raise NotImplementedError(
        "SDXL generation not yet implemented. "
        "Requires SDXL pipeline integration."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization dataset for DMD2 models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["imagenet", "sdxl", "sdv1.5"],
        help="Model type"
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
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation"
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
        help="(ImageNet only) Samples per class for balanced dataset"
    )
    parser.add_argument(
        "--conditioning_sigma",
        type=float,
        default=80.0,
        help="Conditioning sigma for ImageNet"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="(SDXL/SDv1.5 only) Path to prompts file"
    )
    parser.add_argument(
        "--num_denoising_steps",
        type=int,
        default=4,
        help="(SDXL/SDv1.5 only) Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (auto-detect if not specified)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    layers = args.layers.split(",")

    if args.model == "imagenet":
        generate_imagenet_dataset(
            checkpoint_path=args.checkpoint_path,
            output_dir=output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            layers=layers,
            samples_per_class=args.samples_per_class,
            conditioning_sigma=args.conditioning_sigma,
            seed=args.seed,
            device=args.device
        )
    elif args.model in ["sdxl", "sdv1.5"]:
        generate_sdxl_dataset(
            checkpoint_path=args.checkpoint_path,
            output_dir=output_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            layers=layers,
            prompt_file=args.prompt_file,
            num_denoising_steps=args.num_denoising_steps,
            seed=args.seed,
            device=args.device
        )


if __name__ == "__main__":
    main()
