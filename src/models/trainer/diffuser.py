import os
from datetime import datetime

import pandas as pd
import torch
import bitsandbytes as bnb
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as functional
from torchinfo import summary
from tqdm import tqdm

import CONST
from entities import WoundDataset
from models.embeddings.embedding_v1 import Embedding_v1
from models.nets.unet_v1 import UNet_v2
from utils import linear_noise_schedule
from utils.data_utils import split_and_flatten


class Diffuser:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        num_workers: int = 8,
        run_name: str = "DDPM_v1",
        output_dir: str = "./output/",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        noise_steps: int = 50,
        epochs: int = 2000,
        max_lr: float = 1e-4,
        eps: float = 1e-6,
        embedding_dim: int = 256,
    ) -> None:
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

        # Model
        self.model = UNet_v2(in_channels=3, out_channels=3, embedded_dim=256).to(
            self.device
        )

        # Logger
        self.log = SummaryWriter(
            log_dir=f"{os.path.join(self.run_dir, self.run_time)}/logs/"
        )

        # Training settings
        # Noise schedule (beta)
        self.noise_steps = noise_steps
        self.beta = linear_noise_schedule(
            beta_start, beta_end, self.noise_steps
        ).to(self.device)

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

        # Embedding dimmensions
        self.embedding_dim = embedding_dim

        # Log models and training stats
        self.model.eval()
        model_stats = str(
            summary(self.model, [(1, 3, 128, 128), (1, 256)])
        )
        with open(
                f"{os.path.join(self.run_dir, self.run_time)}/model.txt", "w"
        ) as file:
            file.write(model_stats)
            file.write(f"\nTotal epochs: {self.epochs}"
                       f"\nDenoising steps: {self.noise_steps}"
                       f"\nMax learning rate: {max_lr}"
                       f"\nBeta start: {beta_start}"
                       f"\nBeta end: {beta_end}"
                       f"\nEmbedding dimension: {self.embedding_dim}")
            file.close()

    def one_epoch(self, epoch):
        """
        Perform one epoch of training
        :param epoch:
        :return:
        """
        # Put model in training mode
        self.model.train()

        # One batch iteration
        for _, batch in enumerate(
                tqdm(self.dataloader, desc=f"Epoch {epoch:5d}/{self.epochs}", position=0, leave=False)
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

    def step(self, loss):
        """
        Take training step - backward the loss, additional steps
        :param loss:
        :return:
        """
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
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
