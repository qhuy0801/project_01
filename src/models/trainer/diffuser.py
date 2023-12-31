import gc
import math
import os
from datetime import datetime

import torch
import bitsandbytes as bnb
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as functional
from torchinfo import summary
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from models.nets.unet_v2 import UNet_v2
from utils import (
    linear_schedule,
    save_checkpoint,
    de_normalise,
    load_checkpoint,
    quadratic_schedule,
    sigmoid_schedule,
    cosine_schedule,
)


class Diffuser:
    """
    The class for Diffusion models, which includes:
        - Neural network (U-Net)
        - Variance scheduler
        - Training and sampling functions
        - Step embedding function (cloned from Embedding class for the ease of apply)
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 24,
        num_workers: int = 8,
        run_name: str = "DDPM_v1",
        output_dir: str = "./output/",
        model_checkpoint: str = None,
        variance_schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        noise_steps: int = 1000,
        epochs: int = 10000,
        max_lr: float = 2e-4,
        eps: float = 1e-8,
        embedding_dim: int = 256,
        attn_heads: int = 1,
        wandb_run: Run = None,
        additional_note: str = "",
    ) -> None:
        """
        Constructor
        :param dataset:
        :param batch_size:
        :param num_workers:
        :param run_name:
        :param output_dir:
        :param model_checkpoint:
        :param variance_schedule_type:
        :param beta_start:
        :param beta_end:
        :param noise_steps:
        :param epochs:
        :param max_lr:
        :param eps:
        :param embedding_dim:
        :param attn_heads:
        :param wandb_run:
        :param additional_note:
        """
        super().__init__()
        # Platform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run name and directory
        self.run_name = run_name
        self.run_dir = os.path.join(output_dir, self.run_name)
        self.run_time = datetime.now().strftime("%m%d%H%M")

        # Data
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        # Sample loader
        random_sampler = torch.utils.data.RandomSampler(
            data_source=self.dataset, replacement=True, num_samples=1
        )
        self.sample_loader = DataLoader(
            dataset=self.dataset, batch_size=1, sampler=random_sampler
        )

        # Embedding dimensions
        self.embedding_dim = embedding_dim

        # Number of attention head in self attention
        self.attn_heads = attn_heads

        # Model
        self.model = UNet_v2(
            in_channels=3,
            out_channels=3,
            attn_heads=self.attn_heads,
            embedded_dim=self.embedding_dim,
        ).to(self.device)

        # If there is checkpoint for resuming training
        if model_checkpoint is not None:
            checkpoint = load_checkpoint(model_checkpoint, str(self.device))
            self.model.load_state_dict(checkpoint["model"])

            # Un-reference
            checkpoint = None

        # Logger
        self.log = SummaryWriter(
            log_dir=f"{os.path.join(self.run_dir, self.run_time)}/logs/"
        )

        # Training settings
        # Variance schedule (beta)
        self.noise_steps = noise_steps
        if variance_schedule_type == "quadratic":
            self.beta = quadratic_schedule(beta_start, beta_end, self.noise_steps).to(
                self.device
            )
        elif variance_schedule_type == "sigmoid":
            self.beta = sigmoid_schedule(beta_start, beta_end, self.noise_steps).to(
                self.device
            )
        elif variance_schedule_type == "cosine":
            self.beta = cosine_schedule(self.noise_steps).to(self.device)
        else:
            self.beta = linear_schedule(beta_start, beta_end, self.noise_steps).to(
                self.device
            )

        # Alpha
        self.alpha = 1.0 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)

        # Learning rate, decay, optimiser and scheduler
        self.epochs = epochs
        self.optimiser = bnb.optim.AdamW(self.model.parameters(), lr=max_lr, eps=eps)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimiser,
            max_lr=max_lr,
            steps_per_epoch=len(self.dataloader),
            epochs=self.epochs,
        )

        # Best loss for checkpoint
        self.best_loss: float = 2000.0

        # Log models and training stats
        self.model.eval()
        model_stats = str(summary(self.model, [(1, 3, 64, 64), (1, 256)]))
        with open(
            f"{os.path.join(self.run_dir, self.run_time)}/model.txt", "w"
        ) as file:
            file.write(model_stats)
            file.write(
                f"\nTotal epochs: {self.epochs}"
                f"\nDenoising steps: {self.noise_steps}"
                f"\nMax learning rate: {max_lr}"
                f"\nVariance schedule type: {variance_schedule_type}"
                f"\nBeta start: {beta_start}"
                f"\nBeta end: {beta_end}"
                f"\nEmbedding dimension: {self.embedding_dim}"
                f"\nNumber of attention heads: {self.attn_heads}"
                f"\nBatch size: {batch_size}"
                f"\nNum workers: {num_workers}"
                f"\nAdditional note: {additional_note}"
            )
            file.close()

        # Log wandb
        self.wandb_run = wandb_run

        # Clear memory
        gc.collect()

    def fit(self, sample_every: int = 100):
        """
        Training loop trigger
        :param sample_every:
        :return:
        """

        # Training mode
        self.model.train()

        print(f"Starting training {self.run_name} for {self.epochs} epochs...")

        # Training loop
        for epoch in range(self.epochs):
            epoch_loss = self.one_epoch(epoch)

            # Logs
            self.log.add_scalar("Epoch/MSE_loss", epoch_loss, epoch)
            if self.scheduler is not None:
                self.log.add_scalar(
                    "Epoch/Learning_rate",
                    self.optimiser.param_groups[0]["lr"],
                    epoch,
                )
            self.log.flush()

            # WandB log
            if self.wandb_run is not None:
                wandb.log({"train_mse_loss": epoch_loss})
                if self.scheduler is not None:
                    wandb.log({"learning_rate": self.optimiser.param_groups[0]["lr"]})

            # Save checkpoint if new loss archived
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                save_checkpoint(
                    {"model": self.model.state_dict()},
                    self.run_name,
                    os.path.join(self.run_dir, self.run_time),
                )

            # Log samples
            if epoch % sample_every == 0:
                sample = self.sample(epoch)
                self.log.add_images(
                    tag=f"Samples/Random/Epoch:{epoch}",
                    img_tensor=sample,
                    global_step=epoch,
                    dataformats="NCHW",
                )

        print(f"{self.epochs} training completed")
        return None

    def one_epoch(self, epoch):
        """
        Perform one epoch of training
        :param epoch:
        :return:
        """
        # Put model in training mode
        self.model.train()

        # Store epoch loss
        epoch_loss: float = 0.0

        # One batch iteration
        for _, batch in enumerate(
            tqdm(
                self.dataloader,
                desc=f"Epoch {epoch:5d}/{self.epochs}",
                position=0,
                leave=False,
            )
        ):
            # Preparing
            semantics, time_steps, noised_images, noises = self.prepare_batch(batch)

            # Conditioning (time_step + semantics)
            time_step_embeddings = self.step_embeddings(time_steps)
            embeddings = time_step_embeddings + semantics

            # Forwarding
            pred_noises = self.model(noised_images, embeddings)

            # Loss
            loss = functional.mse_loss(pred_noises, noises)

            # Take step
            self.step(loss)

            # Store the loss
            epoch_loss += loss

        return epoch_loss.mean().item()

    def step(self, loss):
        """
        Take training step - backward the loss, additional steps
        :param loss:
        :return:
        """
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def prepare_batch(self, batch):
        """
        Prepare the training input for one extracted batch
        :param batch:
        :return:
        """
        # Unpack batch
        images, _, semantics = batch

        # Transfer data to devices
        images = images.to(self.device)
        semantics = semantics.to(self.device)

        # Get a random timestep image batch
        time_steps = torch.randint(
            low=1, high=self.noise_steps, size=(images.shape[0],)
        )

        # Add noise to image
        noised_images, noises = self.add_noise(images, time_steps)

        return semantics, time_steps, noised_images, noises

    def add_noise(self, images, time_steps):
        """
        Add noise to image based on timestep
        :param images:
        :param time_steps:
        :return:
        """
        # Random gaussian noise with same shape of input
        epsilon = torch.randn_like(images)

        # Return images with added noise based on alpha
        return (
            torch.sqrt(self.alpha_cumulative[time_steps])[:, None, None, None] * images
            + torch.sqrt(1 - self.alpha_cumulative[time_steps])[:, None, None, None]
            * epsilon,
            epsilon,
        )

    def step_embeddings(self, steps):
        """
        This function implement Sinusoidal positional embeddings.
        Which generates embeddings using sin and cos functions
        Input: tensor shape (N)
        :return: embedding tensor shape of (N, self.embedding_dims)
        """
        steps = steps.unsqueeze(-1).to(self.device)
        normalising_factor = 1.0 / (
            10000
            ** (
                torch.arange(0, self.embedding_dim, 2, device=self.device).float()
                / self.embedding_dim
            )
        )
        position_a = torch.sin(
            steps.repeat(1, self.embedding_dim // 2) * normalising_factor
        )
        position_b = torch.cos(
            steps.repeat(1, self.embedding_dim // 2) * normalising_factor
        )
        return torch.cat([position_a, position_b], dim=-1)

    @torch.inference_mode()
    def sample(self, epoch, progress_sample: {float} = {0.75, 0.5, 0.25}):
        """
        The sampling process, we will also sample the generation progress
        :param epoch:
        :param progress_sample:
        :return:
        """
        # Convert progress sample to step to print output
        progress_sample = {math.ceil(x * self.noise_steps) for x in progress_sample}

        # Get a random embedding from dataset
        image_org, _, semantic = next(iter(self.sample_loader))
        image_org = image_org.to(self.device)
        semantic = semantic.to(self.device)

        # Get random gaussian noise with the shape of image
        image = torch.randn((1, 3, 64, 64)).to(self.device)

        # Get an array for displaying
        display = [
            de_normalise(image_org, self.device),
            de_normalise(image, self.device),
        ]

        # Put model in training mode
        self.model.eval()

        # Iterate through steps
        for _, time_step in enumerate(
            tqdm(
                reversed(range(1, self.noise_steps)),
                desc=f"Sampling {epoch:5d}/{self.epochs}",
                position=0,
                leave=False,
            )
        ):
            # Numeric timestep
            t = torch.tensor([time_step], dtype=torch.long).to(self.device)

            # Conditioning
            time_step_embedding = self.step_embeddings(t)
            embedding = time_step_embedding + semantic

            # Forward
            pred_noise = self.model(image, embedding)

            # Denoise parameters
            __alpha = self.alpha[t][:, None, None, None]
            __alpha_cumulative = self.alpha_cumulative[t][:, None, None, None]
            __beta = self.beta[t][:, None, None, None]

            # Iteration noises
            if time_step > 1:
                iteration_noise = torch.randn_like(image)
            else:
                iteration_noise = torch.zeros_like(image)

            # Denoising algorithms
            image = (
                1
                / torch.sqrt(__alpha)
                * (
                    image
                    - ((1 - __alpha) / (torch.sqrt(1 - __alpha_cumulative)))
                    * pred_noise
                )
                + torch.sqrt(__beta) * iteration_noise
            )

            # If timestep is in the selected tuple for sampling, add to image
            if time_step in progress_sample:
                display.append(de_normalise(image, self.device))

        # Final result append
        display.append(de_normalise(image, self.device))

        # Concatenation for displaying
        display = torch.cat(display, dim=0)

        return display
