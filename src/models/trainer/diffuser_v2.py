import os.path
import gc
from datetime import datetime
import bitsandbytes as bnb

import torch
from accelerate import Accelerator
from diffusers import UNet2DModel, DDIMScheduler

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from entities.data.image_dataset import ImageDataset
from models.pipeline.ddim import DDIMPipeline
from utils import save_checkpoint, load_checkpoint
from utils.training_utils import to_uint8


class Diffuser_v2:
    def __init__(
        self,
        train_dataset: ImageDataset,
        batch_size: int = 10,
        checkpoint_path: str = None,
        num_workers: int = 30,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: tuple = (128, 128, 256, 256, 512, 512),
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        scheduler_prediction_type: str = "v_prediction",
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
        self.train_data = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        # Settings
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epochs = epochs
        self.noise_steps = noise_steps
        self.max_lr = max_lr

        # Dependencies
        self.model = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            sample_size=self.train_dataset.target_size,
            layers_per_block=2,
            block_out_channels=block_out_channels,
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
            num_class_embeds=len(train_dataset.class_dict),
        ).to(self.device)
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            prediction_type=scheduler_prediction_type,
        )
        self.optimiser = bnb.optim.AdamW(self.model.parameters(), lr=self.max_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimiser,
            max_lr=self.max_lr,
            steps_per_epoch=len(self.train_data),
            epochs=self.epochs,
        )
        self.loss_func = torch.nn.MSELoss()

        self.__dict__.update(kwargs)

        # If there is path, restore from checkpoint
        if checkpoint_path is not None:
            self.__from_checkpoint(checkpoint_path=checkpoint_path)

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

        # Logger
        self.log = SummaryWriter(
            log_dir=f"{os.path.join(self.run_dir, self.run_time)}/logs/"
        )

        # Step counter
        self.current_step = 0

        # Seed for replication
        self.seed = 42

        # Best loss for model saving conditioning
        self.best_loss = 1000

        # Class for samples
        self.class_samples = train_dataset.get_sample_labels(2)

        # Clear all unreferenced instances
        gc.collect()

    def __from_checkpoint(self, checkpoint_path):
        """
        Restore the model and optimiser from checkpoint and collect un-referenced instances
        :param checkpoint_path:
        :return:
        """
        print(f"Restored from checkpoint {checkpoint_path}")
        __checkpoint = load_checkpoint(
            checkpoint_path, device_str=str(self.device)
        )
        self.model.load_state_dict(__checkpoint["model"])
        self.optimiser.load_state_dict(__checkpoint["optimiser"])
        self.lr_scheduler.load_state_dict(__checkpoint["lr_scheduler"])
        gc.collect()

    def fit(self):
        """
        Training trigger
        :return:
        """
        print(f"Starting training for {self.epochs} epochs...")
        # Iterate the epochs
        for epoch in range(self.epochs):
            epoch_loss = self.__one_epoch(epoch)
            if epoch_loss <= self.best_loss:
                save_checkpoint(
                    {
                        "current_loss": epoch_loss,
                        "model": self.model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    },
                    self.run_name,
                    os.path.join(self.run_dir, self.run_time),
                )
                self.noise_scheduler.save_pretrained(
                    save_directory=os.path.join(self.run_dir, self.run_time)
                )
                self.best_loss = epoch_loss
                self.log.add_images(
                        tag=f"{epoch}_samples",
                        img_tensor=to_uint8(self.sample()),
                        global_step=self.current_step,
                        dataformats="NCHW")
                self.log.flush()

    @torch.no_grad()
    def sample(self):
        """
        Generate a sample
        :return: (N,C,H,W) tensor
        """
        pipeline = DDIMPipeline(
            unet=self.accelerator.unwrap_model(self.model),
            scheduler=self.noise_scheduler,
        )
        return pipeline(
            batch_size=len(self.class_samples),
            labels=self.class_samples,
            generator=torch.manual_seed(self.seed),
            output_type="numpy",
        ).images

    def __one_epoch(self, epoch):
        """
        Perform 1 training epoch
        :param epoch:
        :return:
        """
        # Array to store epoch loss
        __epoch_loss = 0.0

        # Iterate the dataloader
        for _, batch in enumerate(
            tqdm(
                self.train_data,
                desc=f"Epoch {epoch:5d}",
                disable=not self.accelerator.is_local_main_process,
            )
        ):
            # Prepare batch
            noised_images, time_steps, noises, labels = self.__prepare_batch(batch)

            # Forward and backward
            with self.accelerator.accumulate(self.model):
                pred_noise = self.__forward(noised_images, time_steps, labels)
                __loss = self.loss_func(pred_noise, noises)
                self.__backward(__loss)

                # Logs
                self.log.add_scalar("batch_loss", __loss.item(), self.current_step)
                self.log.add_scalar(
                    "learning_rate",
                    self.lr_scheduler.get_last_lr()[0],
                    self.current_step,
                )
                self.log.flush()

            # Step count
            self.current_step += 1

            # Store the loss
            __epoch_loss += __loss
        return __epoch_loss.mean().item()

    def __forward(self, x, time_steps, labels):
        """
        Forward function of model
        :param x:
        :param time_steps:
        :param labels:
        :return:
        """
        return self.model(
            sample=x, timestep=time_steps, class_labels=labels, return_dict=False
        )[0]

    def __backward(self, loss):
        """
        Get the loss and update the weights
        :param loss:
        :return:
        """
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimiser.step()
        self.lr_scheduler.step()
        self.optimiser.zero_grad()

    def __prepare_batch(self, batch):
        """
        Prepare the batch: transfer to desired device, add noises
        :param batch:
        :return:
        """
        # Unpack
        images, labels = batch

        # Push to device
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Noise and noise steps
        noises = torch.randn(images.shape).to(images.device)
        time_steps = torch.randint(
            low=0,
            high=self.noise_steps,
            size=(images.shape[0],),
            device=images.device,
        ).long()

        # Add noise to images
        noised_images = self.noise_scheduler.add_noise(images, noises, time_steps)

        return noised_images, time_steps, noises, labels
