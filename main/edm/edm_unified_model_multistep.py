# Multi-step unified model for 10-step denoising training
# Extends EDMUniModel with backward simulation and sigma scheduling

from main.edm.edm_guidance import EDMGuidance, get_sigmas_karras
from accelerate.utils import broadcast
from torch import nn
import torch
import copy


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


class EDMUniModelMultistep(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        self.guidance_model = EDMGuidance(args, accelerator)

        self.guidance_min_step = self.guidance_model.min_step
        self.guidance_max_step = self.guidance_model.max_step

        if args.initialie_generator:
            self.feedforward_model = copy.deepcopy(self.guidance_model.fake_unet)
        else:
            raise NotImplementedError("Only support initializing generator from guidance model.")

        self.feedforward_model.requires_grad_(True)

        self.num_train_timesteps = args.num_train_timesteps

        # Multi-step denoising configuration
        self.denoising = getattr(args, 'denoising', False)
        self.num_denoising_step = getattr(args, 'num_denoising_step', 1)
        self.backward_simulation = getattr(args, 'backward_simulation', False)

        # Sigma schedule parameters
        self.sigma_max = args.sigma_max
        self.sigma_min = args.sigma_min
        self.rho = args.rho

        # Generate denoising sigma list
        if self.denoising and self.num_denoising_step > 1:
            self.denoising_sigma_list = get_denoising_sigmas(
                num_steps=self.num_denoising_step,
                sigma_max=self.sigma_max,
                sigma_min=self.sigma_min,
                rho=self.rho
            ).to(accelerator.device)
        else:
            # Single step - use conditioning_sigma
            self.denoising_sigma_list = torch.tensor(
                [args.conditioning_sigma], device=accelerator.device
            )

    @torch.no_grad()
    def sample_backward(self, batch_size, labels):
        """
        Backward simulation: generate clean images from noise using current generator.

        This simulates the inference process during training to reduce
        train-inference mismatch. The generator sees its own intermediate outputs.

        Args:
            batch_size: Number of images to generate
            labels: One-hot class labels [B, num_classes]

        Returns:
            generated_image: The generated clean image at the selected step
            return_sigma: The sigma at which we stopped denoising
        """
        device = labels.device

        # Select a random step to stop at (shared across all GPUs for consistency)
        selected_step = torch.randint(
            low=0, high=self.num_denoising_step,
            size=(1,), device=device, dtype=torch.long
        )
        selected_step = broadcast(selected_step, from_process=0)

        # Start from pure noise at sigma_max
        sigma_max = self.denoising_sigma_list[0]
        x = torch.randn(
            batch_size, 3, self.args.resolution, self.args.resolution,
            device=device
        ) * sigma_max

        generated_image = x

        # Iteratively denoise up to selected_step
        for step_idx in range(selected_step.item()):
            sigma = self.denoising_sigma_list[step_idx]
            sigma_tensor = torch.ones(batch_size, device=device) * sigma

            # Generator predicts clean image directly (EDM parameterization)
            generated_image = self.feedforward_model(x, sigma_tensor, labels)

            # Transition to next noise level (if not the last step we're doing)
            if step_idx < selected_step.item() - 1:
                next_sigma = self.denoising_sigma_list[step_idx + 1]
                noise = torch.randn_like(generated_image)
                x = generated_image + next_sigma * noise

        # Return the sigma at which we stopped
        return_sigma = self.denoising_sigma_list[selected_step.item()]
        return generated_image, return_sigma, selected_step.item()

    @torch.no_grad()
    def prepare_denoising_data(self, denoising_dict, noise):
        """
        Prepare training data for multi-step denoising.

        Args:
            denoising_dict: Dictionary containing 'images' and 'labels' from dataset
            noise: Pre-generated noise tensor

        Returns:
            sigmas: Selected sigma for each sample
            labels: One-hot labels
            noisy_images: Images with noise added at selected sigmas
            clean_images: Target clean images
        """
        batch_size = noise.shape[0]
        device = noise.device

        labels = denoising_dict['labels']

        if self.backward_simulation:
            # Generate clean images via backward simulation
            clean_images, sigmas, _ = self.sample_backward(batch_size, labels)
            # sigmas is a scalar here, expand to batch
            sigmas = torch.ones(batch_size, device=device) * sigmas
        else:
            # Use real images from dataset
            clean_images = denoising_dict['images']
            # Random sigma selection for each sample
            step_indices = torch.randint(
                0, self.num_denoising_step, (batch_size,), device=device
            )
            sigmas = self.denoising_sigma_list[step_indices]

        # Add noise at the selected sigma level
        noisy_images = clean_images + sigmas.view(-1, 1, 1, 1) * noise

        # Handle pure noise case (at sigma_max)
        sigma_max = self.denoising_sigma_list[0]
        pure_noise_mask = (sigmas >= sigma_max * 0.99)
        noisy_images[pure_noise_mask] = noise[pure_noise_mask] * sigma_max

        return sigmas, labels, noisy_images, clean_images

    def forward(
        self, scaled_noisy_image, timestep_sigma, labels,
        real_train_dict=None,
        compute_generator_gradient=False,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None,
        denoising_dict=None,
        noise=None
    ):
        """
        Forward pass supporting both single-step and multi-step modes.

        Args:
            scaled_noisy_image: Noisy input (used in single-step mode)
            timestep_sigma: Sigma values for conditioning
            labels: One-hot class labels
            real_train_dict: Real images for GAN loss
            compute_generator_gradient: Whether to compute generator gradients
            generator_turn: True if updating generator
            guidance_turn: True if updating guidance model
            guidance_data_dict: Data dict from generator turn (for guidance turn)
            denoising_dict: Dataset dict for denoising mode
            noise: Pre-generated noise for denoising mode
        """
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn)

        if generator_turn:
            # Multi-step denoising mode
            if self.denoising and denoising_dict is not None:
                sigmas, labels, noisy_images, clean_targets = self.prepare_denoising_data(
                    denoising_dict, noise
                )
                timestep_sigma = sigmas
                scaled_noisy_image = noisy_images
            # else: use provided scaled_noisy_image and timestep_sigma (single-step mode)

            if not compute_generator_gradient:
                with torch.no_grad():
                    generated_image = self.feedforward_model(
                        scaled_noisy_image, timestep_sigma, labels
                    )
            else:
                generated_image = self.feedforward_model(
                    scaled_noisy_image, timestep_sigma, labels
                )

            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,
                    "label": labels,
                    "real_train_dict": real_train_dict
                }

                # Disable guidance gradients to avoid side effects
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict
                )
                self.guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {}

            log_dict['generated_image'] = generated_image.detach()
            log_dict['denoising_sigma'] = timestep_sigma.detach() if isinstance(timestep_sigma, torch.Tensor) else timestep_sigma

            log_dict['guidance_data_dict'] = {
                "image": generated_image.detach(),
                "label": labels.detach(),
                "real_train_dict": real_train_dict
            }

        elif guidance_turn:
            assert guidance_data_dict is not None
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict
            )

        return loss_dict, log_dict


@torch.no_grad()
def sample_multistep(
    generator,
    noise,
    labels,
    num_steps=10,
    sigma_max=80.0,
    sigma_min=0.002,
    rho=7.0,
    guidance_scale=1.0,
    stochastic=True
):
    """
    Multi-step sampling with optional CFG.

    Args:
        generator: The trained generator model
        noise: Initial noise tensor [B, 3, H, W]
        labels: One-hot class labels [B, num_classes]
        num_steps: Number of denoising steps
        sigma_max: Maximum sigma
        sigma_min: Minimum sigma
        rho: Karras schedule parameter
        guidance_scale: CFG scale (1.0 = no guidance)
        stochastic: Whether to add noise between steps

    Returns:
        Generated images [B, 3, H, W]
    """
    batch_size = noise.shape[0]
    device = noise.device

    # Generate sigma schedule
    sigmas = get_denoising_sigmas(num_steps, sigma_min, sigma_max, rho).to(device)

    # Start from pure noise scaled by sigma_max
    x = noise * sigma_max

    # Unconditional labels (zeros) for CFG
    uncond_labels = torch.zeros_like(labels)

    for i, sigma in enumerate(sigmas):
        sigma_tensor = torch.ones(batch_size, device=device) * sigma

        if guidance_scale > 1.0:
            # Classifier-free guidance
            pred_cond = generator(x, sigma_tensor, labels)
            pred_uncond = generator(x, sigma_tensor, uncond_labels)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            # No guidance
            pred = generator(x, sigma_tensor, labels)

        # Transition to next step
        if i < len(sigmas) - 1:
            next_sigma = sigmas[i + 1]
            if stochastic:
                # Stochastic sampling - add noise
                x = pred + next_sigma * torch.randn_like(pred)
            else:
                # Deterministic - just use prediction
                x = pred
        else:
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
    return sample_multistep(
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
