import os
from datetime import datetime
import bitsandbytes as bnb

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from entities.data.image_dataset import ImageDataset
from models.nets.vae import VAE
from utils import save_checkpoint


class VAETrainer:
    def __init__(
        self,
        train_dataset: ImageDataset,
        batch_size: int = 10,
        checkpoint_path: str = None,
        num_workers: int = 30,
        epochs: int = 1000,
        max_lr: float = 1e-4,
    ) -> None:
        super().__init__()
        # Platform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = "vae_v1"
        self.run_dir = os.path.join("../output/", self.run_name)
        self.run_time = datetime.now().strftime("%m%d%H%M")

        # Data
        self.train_dataset = train_dataset
        self.train_data = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        # Training settings
        self.epochs = epochs

        # Model
        self.model = VAE().to(self.device)

        # Dependencies
        self.optimiser = bnb.optim.AdamW(self.model.parameters(), lr=max_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimiser,
            max_lr=max_lr,
            steps_per_epoch=len(self.train_data),
            epochs=epochs,
            anneal_strategy="cos",
        )

        # Logger
        self.log = SummaryWriter(
            log_dir=f"{os.path.join(self.run_dir, self.run_time)}/logs/"
        )

        # Step count
        self.current_step = 0

        # Best loss for checkpoints
        self.best_mse_loss = 1000.0

    def fit(self):
        print(f"Starting training {self.run_name} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_kl_loss, epoch_mse_loss = self.__one_epoch(epoch)

            # Logs
            self.log.add_scalar(
                "Epoch_loss/KL_loss", epoch_kl_loss, self.current_step
            )
            self.log.add_scalar(
                "Epoch_loss/MSE_loss", epoch_mse_loss, self.current_step
            )
            self.log.flush()

            # Checkpoint
            if epoch_mse_loss < self.best_mse_loss:
                self.best_mse_loss = epoch_mse_loss
                save_checkpoint(
                    {
                        "mse_loss": epoch_mse_loss,
                        "model": self.model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                    },
                    self.run_name,
                    os.path.join(self.run_dir, self.run_time),
                )

        print("Training completed!")
        return None

    def __one_epoch(self, epoch):
        """
        Perform one epoch of training
        :param epoch:
        :return:
        """
        # Store the epoch loss
        __epoch_kl_loss = 0.0
        __epoch_mse_loss = 0.0

        # Iterate the dataloader
        for _, batch in enumerate(tqdm(self.train_data, desc=f"Epoch {epoch:5d}")):
            images, _ = batch
            images = images.to(self.device)

            # Forwarding
            pred_images, mu, sigma = self.model(images)

            # Loss = BCE loss + KL divergence loss
            kl = 0.5 * torch.sum(-1 - sigma + mu.pow(2) + sigma.exp())
            kl_loss = (
                functional.binary_cross_entropy(pred_images, images, size_average=False)
                + kl
            )

            # Reconstruction loss (MSE loss)
            mse_loss = functional.mse_loss(pred_images, images)

            # Logs
            self.log.add_scalar("Batch_loss/KL_loss", kl_loss.item(), self.current_step)
            self.log.add_scalar(
                "Batch_loss/MSE_loss", mse_loss.item(), self.current_step
            )
            self.log.flush()

            # Backward
            self.__backward(kl_loss)

            # Step count
            self.current_step += 1

            # Store the loss
            __epoch_kl_loss += kl_loss
            __epoch_mse_loss += mse_loss

        return __epoch_kl_loss.mean().item(), __epoch_mse_loss.mean().item()

    def __backward(self, loss):
        """
        Backward the loss and update weights
        :param loss:
        :return:
        """
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        self.lr_scheduler.step()
