import os.path
import random
from datetime import datetime

import torch
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from fastprogress import progress_bar
from torch.utils.data import DataLoader, RandomSampler


class Diffuser_v2:
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        batch_size: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        noise_steps: int = 1000,
        max_lr: float = 1e-4,
        epochs: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__()

        # Platform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = "diffusion_v3"
        self.run_dir = os.path.join("../output/", self.run_name)
        self.run_time = datetime.now().strftime("%m%d%H%M")

        # Data
        self.train_dataset = train_dataset
        self.train_data = DataLoader(train_dataset, batch_size=batch_size)

        # Settings
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epochs = epochs
        self.noise_steps = noise_steps
        self.max_lr = max_lr

        # Dependencies
        self.model = UNet2DModel(
            sample_size=self.train_dataset.target_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type=None,
            num_class_embeds=self.train_dataset.class_tuple.__len__(),
        ).to(self.device)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.noise_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )
        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.max_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimiser,
            max_lr=self.max_lr,
            steps_per_epoch=len(self.train_data),
            epochs=self.epochs,
        )
        self.loss_func = torch.nn.MSELoss()

        self.__dict__.update(kwargs)

        # Hardware controlling
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=1,
            log_with="tensorboard",
            project_dir=self.run_dir,
        )

        # Wrap everything in with accelerator
        (
            self.model,
            self.optimiser,
            self.train_data,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            [self.model, self.optimiser, self.train_data, self.lr_scheduler]
        )

        # Tracker
        logging_hps = {
            "image_size": self.train_dataset.target_size,
            "batch_soze": batch_size,
            "max_lr": self.max_lr,
            "noise_steps": self.noise_steps,
            "epochs": self.epochs,
        }
        self.accelerator.init_trackers(self.run_time, config=logging_hps)

        # Step counter
        self.current_step = 0

        # Seed for replication
        self.seed = 42

    def fit(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch} ")
            self.one_epoch(epoch)

    def one_epoch(self, current_epoch: int):
        # Break down to batches
        batches = progress_bar(self.train_data, leave=False)

        # Loop
        for _, (images, _labels) in enumerate(batches):
            noised_images, time_steps, noise, labels = self.prepare_samples(
                images, _labels
            )

            # Forward progress
            with self.accelerator.accumulate(self.model):
                pred_noises = self.model(
                    sample=noised_images,
                    timestep=time_steps,
                    class_labels=labels,
                    return_dict=False,
                )[0]
                loss = self.loss_func(pred_noises, noise)
                progress_bar.comment = f"MSE={loss.item():2.3f}"

                # Backward weight updating
                self.backward(loss)

            # Log
            _log = {
                "loss": loss.detach().item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
                "epoch": current_epoch,
            }
            self.accelerator.log(_log, step=self.current_step)

            # Step count
            self.current_step += 1

    @torch.no_grad()
    def sample(self, epoch: int, num_samples: int = 4):
        # Get random samples
        samples_batch = self.random_sampler(num_samples=num_samples)

        # Unpack
        images = samples_batch[0]
        _labels = samples_batch[1]

        # Prepare
        noised_images, time_steps, _, labels = self.prepare_samples(images, _labels)

        # Push
        pred_noises = self.model(
            sample=noised_images,
            timestep=time_steps,
            class_labels=labels,
            return_dict=False,
        )[0]

        # Save samples
        # for i in range(len(images)):
        #     save_pil_image(
        #         images[i],
        #         os.path.join(
        #             self.run_dir,
        #             self.run_time,
        #             str(f"epoch{epoch}")
        #             + str(_labels[i])
        #             + str(time_steps[i])
        #             + "original",
        #         ),
        #     )
        #     save_pil_image(
        #         noised_images[i],
        #         os.path.join(
        #             self.run_dir,
        #             self.run_time,
        #             str(epoch) + str(_labels[i]) + str(time_steps[i]) + "noised",
        #         ),
        #     )
        #     save_pil_image(
        #         pred_noises[i],
        #         os.path.join(
        #             self.run_dir,
        #             self.run_time,
        #             str(f"epoch{epoch}")
        #             + str(_labels[i])
        #             + str(time_steps[i])
        #             + "pred_noises",
        #         ),
        #     )

    def backward(self, loss):
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimiser.step()
        self.lr_scheduler.step()
        self.optimiser.zero_grad()

    def prepare_samples(self, images, labels):
        # Push to device
        images = images.to(self.device)
        _labels = labels.to(self.device)

        # Noise and noise steps
        noise = torch.randn(images.shape).to(images.device)
        time_steps = torch.randint(
            low=0,
            high=self.noise_steps,
            size=(images.shape[0],),
            device=images.device,
        ).long()

        # Add noise to images
        noised_images = self.noise_scheduler.add_noise(images, noise, time_steps)
        return noised_images, time_steps, noise, _labels

    def random_sampler(self, num_samples: int = 4):
        sample_loader = DataLoader(
            self.train_dataset,
            batch_size=num_samples,
            sampler=RandomSampler(
                self.train_dataset, replacement=True, num_samples=num_samples
            ),
        )
        return next(iter(sample_loader))
