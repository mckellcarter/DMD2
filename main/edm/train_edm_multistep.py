"""
Multi-step denoising training for ImageNet EDM.

This script trains a 10-step (or configurable) denoising model using
the DMD2 framework with backward simulation and optional CFG support.
"""

import matplotlib
matplotlib.use('Agg')

from main.utils import prepare_images_for_saving, draw_valued_array, cycle, draw_probability_histogram
from main.edm.edm_unified_model_multistep import EDMUniModelMultistep
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler
from main.data.lmdb_dataset import LMDBDataset
from accelerate.utils import set_seed
from accelerate import Accelerator
import argparse
import shutil
import wandb
import torch
import time
import os


class TrainerMultistep:
    def __init__(self, args):
        self.args = args

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        accelerator_project_config = ProjectConfiguration(logging_dir=args.output_path)

        # Use bf16 by default for EDM (safer with high sigma values)
        mixed_prec = getattr(args, 'mixed_precision', 'no')
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=mixed_prec,
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=None
        )
        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        if accelerator.is_main_process:
            output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)
            self.output_path = output_path

            if args.cache_dir != "":
                self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
                os.makedirs(self.cache_dir, exist_ok=False)

        # Use multi-step unified model
        self.model = EDMUniModelMultistep(args, accelerator)
        self.dataset_name = args.dataset_name
        self.real_image_path = args.real_image_path

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio
        self.num_train_timesteps = args.num_train_timesteps

        self.cls_loss_weight = args.cls_loss_weight

        self.gan_classifier = args.gan_classifier
        self.gen_cls_loss_weight = args.gen_cls_loss_weight
        self.no_save = args.no_save
        self.previous_time = None
        self.step = 0
        self.cache_checkpoints = (args.cache_dir != "")
        self.max_checkpoint = args.max_checkpoint

        # Multi-step specific
        self.denoising = args.denoising
        self.num_denoising_step = args.num_denoising_step
        self.backward_simulation = args.backward_simulation

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading checkpoints without optimizer states from {args.ckpt_only_path}")
            generator_path = os.path.join(args.ckpt_only_path, "pytorch_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "pytorch_model_1.bin")

            generator_state_dict = torch.load(generator_path, map_location="cpu")
            guidance_state_dict = torch.load(guidance_path, map_location="cpu")

            print(self.model.feedforward_model.load_state_dict(generator_state_dict, strict=False))
            print(self.model.guidance_model.load_state_dict(guidance_state_dict, strict=False))

            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator checkpoints from {args.generator_ckpt_path}")
            generator_path = os.path.join(args.generator_ckpt_path, "pytorch_model.bin")
            print(self.model.feedforward_model.load_state_dict(
                torch.load(generator_path, map_location="cpu"), strict=True
            ))

        # Real images dataloader (for GAN loss)
        real_dataset = LMDBDataset(args.real_image_path)
        real_image_dataloader = torch.utils.data.DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True,
            drop_last=True, num_workers=args.num_workers
        )
        real_image_dataloader = accelerator.prepare(real_image_dataloader)
        self.real_image_dataloader = cycle(real_image_dataloader)

        # Denoising dataloader (for multi-step training)
        if self.denoising:
            denoising_dataset = LMDBDataset(args.real_image_path)
            denoising_dataloader = torch.utils.data.DataLoader(
                denoising_dataset, batch_size=args.batch_size, shuffle=True,
                drop_last=True, num_workers=args.num_workers
            )
            denoising_dataloader = accelerator.prepare(denoising_dataloader)
            self.denoising_dataloader = cycle(denoising_dataloader)

        self.optimizer_guidance = torch.optim.AdamW(
            [param for param in self.model.guidance_model.parameters() if param.requires_grad],
            lr=args.guidance_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        self.optimizer_generator = torch.optim.AdamW(
            [param for param in self.model.feedforward_model.parameters() if param.requires_grad],
            lr=args.generator_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_guidance,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters
        )

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            optimizer=self.optimizer_generator,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters
        )

        (
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance,
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator
        ) = accelerator.prepare(
            self.model.feedforward_model, self.model.guidance_model, self.optimizer_guidance,
            self.optimizer_generator, self.scheduler_guidance, self.scheduler_generator
        )

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.conditioning_sigma = args.conditioning_sigma

        self.label_dim = args.label_dim
        self.eye_matrix = torch.eye(self.label_dim, device=accelerator.device)
        self.delete_ckpts = args.delete_ckpts
        self.keep_checkpoints = args.keep_checkpoints
        self.max_grad_norm = args.max_grad_norm

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        if self.accelerator.is_main_process:
            run = wandb.init(
                config=args, dir=self.output_path,
                **{"mode": "online", "entity": args.wandb_entity, "project": args.wandb_project}
            )
            wandb.run.log_code(".")
            wandb.run.name = args.wandb_name
            print(f"run dir: {run.dir}")
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

    def load(self, checkpoint_path):
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print("loading a previous checkpoints including optimizer and random seed")
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def save(self):
        output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        print(f"start saving checkpoint to {output_path}")

        self.accelerator.save_state(output_path)

        # Checkpoint cleanup logic
        if self.keep_checkpoints is not None:
            # New behavior: keep N most recent checkpoints
            checkpoints = sorted(
                [f for f in os.listdir(self.output_path) if f.startswith("checkpoint_model")]
            )
            if len(checkpoints) > self.keep_checkpoints:
                for folder in checkpoints[:-self.keep_checkpoints]:
                    shutil.rmtree(os.path.join(self.output_path, folder))
        elif self.delete_ckpts:
            # Legacy behavior: delete all but current
            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                    shutil.rmtree(os.path.join(self.output_path, folder))

        if self.cache_checkpoints:
            if os.path.exists(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")):
                shutil.rmtree(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}"))

            shutil.copytree(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"),
                os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")
            )

            checkpoints = sorted(
                [folder for folder in os.listdir(self.cache_dir) if folder.startswith("checkpoint_model")]
            )

            if len(checkpoints) > self.max_checkpoint:
                for folder in checkpoints[:-self.max_checkpoint]:
                    shutil.rmtree(os.path.join(self.cache_dir, folder))

        print("done saving")

    def train_one_step(self):
        self.model.train()
        accelerator = self.accelerator

        # Get real data for GAN loss
        real_dict = next(self.real_image_dataloader)
        real_image = real_dict["images"] * 2.0 - 1.0
        real_label = self.eye_matrix[real_dict["class_labels"].squeeze(dim=1)]

        real_train_dict = {
            "real_image": real_image,
            "real_label": real_label
        }

        # Generate noise
        noise = torch.randn(
            self.batch_size, 3, self.resolution, self.resolution,
            device=accelerator.device
        )

        # Prepare denoising data or use single-step mode
        if self.denoising:
            denoising_dict_raw = next(self.denoising_dataloader)
            denoising_dict = {
                'images': denoising_dict_raw["images"] * 2.0 - 1.0,
                'labels': self.eye_matrix[denoising_dict_raw["class_labels"].squeeze(dim=1)]
            }
            # For multi-step, sigma and noisy_image are prepared inside model
            scaled_noise = noise  # Will be used inside prepare_denoising_data
            timestep_sigma = None  # Will be set inside model
            labels = denoising_dict['labels']
        else:
            # Single-step mode (original behavior)
            denoising_dict = None
            scaled_noise = noise * self.conditioning_sigma
            timestep_sigma = torch.ones(self.batch_size, device=accelerator.device) * self.conditioning_sigma
            labels = torch.randint(
                low=0, high=self.label_dim, size=(self.batch_size,),
                device=accelerator.device, dtype=torch.long
            )
            labels = self.eye_matrix[labels]

        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0

        # Generator forward
        generator_loss_dict, generator_log_dict = self.model(
            scaled_noise, timestep_sigma, labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=True,
            guidance_turn=False,
            denoising_dict=denoising_dict,
            noise=noise
        )

        generator_loss = 0.0

        if COMPUTE_GENERATOR_GRADIENT:
            generator_loss += generator_loss_dict["loss_dm"]

            if self.gan_classifier:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(
                self.model.feedforward_model.parameters(), self.max_grad_norm
            )
            self.optimizer_generator.step()
            self.optimizer_generator.zero_grad()
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()

        # Guidance model forward
        guidance_loss_dict, guidance_log_dict = self.model(
            scaled_noise, timestep_sigma, labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=False,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict['guidance_data_dict']
        )

        guidance_loss = 0
        guidance_loss += guidance_loss_dict["loss_fake_mean"]

        if self.gan_classifier:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.cls_loss_weight

        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(
            self.model.guidance_model.parameters(), self.max_grad_norm
        )
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.scheduler_guidance.step()
        self.optimizer_generator.zero_grad()

        # Combine dictionaries
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        if self.step % self.wandb_iters == 0:
            log_dict['generated_image'] = accelerator.gather(log_dict['generated_image'])
            log_dict['dmtrain_grad'] = accelerator.gather(log_dict['dmtrain_grad'])
            log_dict['dmtrain_timesteps'] = accelerator.gather(log_dict['dmtrain_timesteps'])
            log_dict['dmtrain_pred_real_image'] = accelerator.gather(log_dict['dmtrain_pred_real_image'])
            log_dict['dmtrain_pred_fake_image'] = accelerator.gather(log_dict['dmtrain_pred_fake_image'])

        if accelerator.is_main_process and self.step % self.wandb_iters == 0:
            with torch.no_grad():
                generated_image = log_dict['generated_image']
                generated_image_brightness = (generated_image * 0.5 + 0.5).clamp(0, 1).mean()
                generated_image_std = (generated_image * 0.5 + 0.5).clamp(0, 1).std()

                generated_image_grid = prepare_images_for_saving(generated_image, resolution=self.resolution)

                data_dict = {
                    "generated_image": wandb.Image(generated_image_grid),
                    "generated_image_brightness": generated_image_brightness.item(),
                    "generated_image_std": generated_image_std.item(),
                    "generator_grad_norm": generator_grad_norm.item() if COMPUTE_GENERATOR_GRADIENT else 0,
                    "guidance_grad_norm": guidance_grad_norm.item()
                }

                # Multi-step specific logging
                if self.denoising and 'denoising_sigma' in log_dict:
                    denoising_sigma = log_dict['denoising_sigma']
                    if isinstance(denoising_sigma, torch.Tensor):
                        data_dict['mean_denoising_sigma'] = denoising_sigma.mean().item()

                (
                    dmtrain_noisy_latents, dmtrain_pred_real_image, dmtrain_pred_fake_image,
                    dmtrain_grad, dmtrain_gradient_norm
                ) = (
                    log_dict['dmtrain_noisy_latents'], log_dict['dmtrain_pred_real_image'],
                    log_dict['dmtrain_pred_fake_image'],
                    log_dict['dmtrain_grad'], log_dict['dmtrain_gradient_norm']
                )

                gradient_brightness = dmtrain_grad.mean()
                gradient_std = dmtrain_grad.std(dim=[1, 2, 3]).mean()

                dmtrain_pred_real_image_mean = (dmtrain_pred_real_image * 0.5 + 0.5).clamp(0, 1).mean()
                dmtrain_pred_fake_image_mean = (dmtrain_pred_fake_image * 0.5 + 0.5).clamp(0, 1).mean()

                dmtrain_pred_read_image_std = (dmtrain_pred_real_image * 0.5 + 0.5).clamp(0, 1).std()
                dmtrain_pred_fake_image_std = (dmtrain_pred_fake_image * 0.5 + 0.5).clamp(0, 1).std()

                dmtrain_noisy_latents_grid = prepare_images_for_saving(dmtrain_noisy_latents, resolution=self.resolution)
                dmtrain_pred_real_image_grid = prepare_images_for_saving(dmtrain_pred_real_image, resolution=self.resolution)
                dmtrain_pred_fake_image_grid = prepare_images_for_saving(dmtrain_pred_fake_image, resolution=self.resolution)

                gradient = dmtrain_grad
                gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
                gradient = (gradient - 0.5) / 0.5
                gradient = prepare_images_for_saving(gradient, resolution=self.resolution)

                gradient_scale_grid = draw_valued_array(
                    dmtrain_grad.abs().mean(dim=[1, 2, 3]).cpu().numpy(),
                    output_dir=self.wandb_folder
                )

                difference_scale_grid = draw_valued_array(
                    (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(dim=[1, 2, 3]).cpu().numpy(),
                    output_dir=self.wandb_folder
                )

                difference = (dmtrain_pred_fake_image - dmtrain_pred_real_image)
                difference_brightness = difference.mean()

                difference = (difference - difference.min()) / (difference.max() - difference.min())
                difference = (difference - 0.5) / 0.5
                difference = prepare_images_for_saving(difference, resolution=self.resolution)

                dmtrain_timesteps_grid = draw_valued_array(
                    log_dict['dmtrain_timesteps'].squeeze().cpu().numpy(),
                    output_dir=self.wandb_folder
                )

                data_dict.update({
                    "dmtrain_noisy_latents_grid": wandb.Image(dmtrain_noisy_latents_grid),
                    "dmtrain_pred_real_image_grid": wandb.Image(dmtrain_pred_real_image_grid),
                    "dmtrain_pred_fake_image_grid": wandb.Image(dmtrain_pred_fake_image_grid),
                    "loss_dm": loss_dict['loss_dm'].item(),
                    "loss_fake_mean": loss_dict['loss_fake_mean'].item(),
                    "dmtrain_gradient_norm": dmtrain_gradient_norm,
                    "gradient": wandb.Image(gradient),
                    "difference": wandb.Image(difference),
                    "gradient_scale_grid": wandb.Image(gradient_scale_grid),
                    "difference_norm_grid": wandb.Image(difference_scale_grid),
                    "dmtrain_timesteps_grid": wandb.Image(dmtrain_timesteps_grid),
                    "gradient_brightness": gradient_brightness.item(),
                    "difference_brightness": difference_brightness.item(),
                    "gradient_std": gradient_std.item(),
                    "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean.item(),
                    "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean.item(),
                    "dmtrain_pred_read_image_std": dmtrain_pred_read_image_std.item(),
                    "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std.item()
                })

                (
                    faketrain_latents, faketrain_noisy_latents, faketrain_x0_pred
                ) = (
                    log_dict['faketrain_latents'], log_dict['faketrain_noisy_latents'],
                    log_dict['faketrain_x0_pred']
                )

                faketrain_latents_grid = prepare_images_for_saving(faketrain_latents, resolution=self.resolution)
                faketrain_noisy_latents_grid = prepare_images_for_saving(faketrain_noisy_latents, resolution=self.resolution)
                faketrain_x0_pred_grid = prepare_images_for_saving(faketrain_x0_pred, resolution=self.resolution)

                data_dict.update({
                    "faketrain_latents": wandb.Image(faketrain_latents_grid),
                    "faketrain_noisy_latents": wandb.Image(faketrain_noisy_latents_grid),
                    "faketrain_x0_pred": wandb.Image(faketrain_x0_pred_grid)
                })

                if self.gan_classifier:
                    data_dict['guidance_cls_loss'] = loss_dict['guidance_cls_loss'].item()
                    data_dict['gen_cls_loss'] = loss_dict['gen_cls_loss'].item()

                    pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                    pred_realism_on_real = log_dict["pred_realism_on_real"]

                    hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.cpu().numpy())
                    hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.cpu().numpy())

                    data_dict.update({
                        "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                        "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real)
                    })

                wandb.log(data_dict, step=self.step)

        self.accelerator.wait_for_everyone()

    def train(self):
        for index in range(self.step, self.train_iters):
            self.train_one_step()

            if self.accelerator.is_main_process:
                if (not self.no_save) and self.step % self.log_iters == 0:
                    self.save()

                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.accelerator.wait_for_everyone()
            self.step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--label_dim", type=int, default=1000)
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Mixed precision mode. bf16 recommended for EDM (safer with high sigma)")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)

    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--dataset_name", type=str, default='imagenet')
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--delete_ckpts", action="store_true", help="(deprecated) Delete all but latest checkpoint")
    parser.add_argument("--max_checkpoint", type=int, default=200, help="(deprecated) Use --keep_checkpoints instead")
    parser.add_argument("--keep_checkpoints", type=int, default=None, help="Number of checkpoints to keep (deletes older ones). None = keep all.")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_grad_norm", type=int, default=10)
    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=5)

    parser.add_argument("--cls_loss_weight", type=float, default=1e-2)
    parser.add_argument("--gan_classifier", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=3e-3)
    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=1000)

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--generator_ckpt_path", type=str)

    # Multi-step specific arguments
    parser.add_argument("--denoising", action="store_true", help="Enable multi-step denoising mode")
    parser.add_argument("--num_denoising_step", type=int, default=10, help="Number of denoising steps")
    parser.add_argument("--backward_simulation", action="store_true",
                        help="Use backward simulation for training data")
    parser.add_argument("--label_dropout", type=float, default=0.0,
                        help="Label dropout rate for CFG training")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.wandb_iters % args.dfake_gen_update_ratio == 0, \
        "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args


if __name__ == "__main__":
    args = parse_args()
    trainer = TrainerMultistep(args)
    trainer.train()
