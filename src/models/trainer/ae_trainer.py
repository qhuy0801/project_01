"""
This class is no longer used as we no longer continue with auto-encoder in this project
due to low-performance
"""
import gc
import os
from datetime import datetime
import bitsandbytes as bnb
from torch import nn
from torchinfo import summary

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import save_checkpoint, de_normalise, load_checkpoint


class AETrainer_v1:
    """
    The implementation of auto-encoder, which include the trainer and sampling methods
    """
    def __init__(
        self,
        train_dataset: Dataset,
        model: nn.Module,
        batch_size: int = 10,
        checkpoint_path: str = None,
        num_workers: int = 16,
        num_samples: int = 1,
        epochs: int = 5000,
        max_lr: float = 1e-4,
        min_lr: float = 5e-6,
        lr_decay: float = 0.95,
        lr_threshold: float = 0.2,
        lr_reducing_patience: int = 50,
        run_name: str = "ae",
        output_dir: str = "./output/",
    ) -> None:
        super().__init__()
        # Platform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = run_name
        self.run_dir = os.path.join(output_dir, self.run_name)
        self.run_time = datetime.now().strftime("%m%d%H%M")

        # Data
        self.train_dataset = train_dataset
        self.train_data = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )

        # Random sample
        random_sampler = torch.utils.data.RandomSampler(
            train_dataset, replacement=True, num_samples=num_samples
        )
        self.sample_loader = DataLoader(
            train_dataset, batch_size=num_samples, sampler=random_sampler
        )

        # Training settings
        self.epochs = epochs

        # Model
        self.model = model.to(self.device)

        # Dependencies
        self.optimiser = bnb.optim.AdamW(self.model.parameters(), lr=max_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimiser,
            mode="min",
            factor=lr_decay,
            threshold=lr_threshold,
            min_lr=min_lr,
            patience=lr_reducing_patience,
        )

        # If there is back-up
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(checkpoint_path, str(self.device))
            self.model.load_state_dict(checkpoint["model"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
            if checkpoint["lr_scheduler"] is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            # Un-reference and clear
            checkpoint = None
            gc.collect()

        # Logger
        self.log = SummaryWriter(
            log_dir=f"{os.path.join(self.run_dir, self.run_time)}/logs/"
        )

        # Log model stats
        self.model.eval()
        model_stats = str(
            summary(
                self.model,
                (
                    1,
                    self.model.encoder.input_dim,
                    self.train_dataset.target_tensor_size,
                    self.train_dataset.target_tensor_size,
                ),
                verbose=0,
            )
        )
        with open(
            f"{os.path.join(self.run_dir, self.run_time)}/model.txt", "w"
        ) as file:
            file.write(f"\nImage size: {self.train_dataset.target_tensor_size}")
            file.write(model_stats)
            file.close()

        # Best loss for checkpoints
        self.best_mse_loss = 2000.0

    def fit(self, sample_after: int = 100):
        self.model.train()
        print(f"Starting training {self.run_name} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_mse_loss = self.__one_epoch(epoch)
            self.__step_epoch(epoch_mse_loss)

            # Logs
            self.log.add_scalar("Epoch/MSE_loss", epoch_mse_loss, epoch)
            if self.lr_scheduler is not None:
                self.log.add_scalar(
                    "Epoch/Learning_rate",
                    self.optimiser.param_groups[0]["lr"],
                    epoch,
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
                        "lr_scheduler": self.lr_scheduler.state_dict()
                        if self.lr_scheduler is not None
                        else None,
                    },
                    self.run_name,
                    os.path.join(self.run_dir, self.run_time),
                )

            # Log a reconstructed image
            if epoch % sample_after == 0:
                sample, reconstructed_sample = self.reconstruct_sample()
                self.log.add_images(
                    tag=f"Samples/Random/Epoch:{epoch}",
                    img_tensor=torch.cat(
                        (
                            de_normalise(sample, self.device),
                            de_normalise(reconstructed_sample, self.device),
                        ),
                        dim=0,
                    ),
                    global_step=epoch,
                    dataformats="NCHW",
                )

        print("Training completed!")
        return None

    def __one_epoch(self, epoch):
        """
        Perform one epoch of training
        :param epoch:
        :return: combination loss (KL + BCE), reconstruction loss (MSE)
        """
        # Store the epoch loss
        __epoch_mse_loss = 0.0

        # Iterate the dataloader
        for _, batch in enumerate(
            tqdm(self.train_data, desc=f"Epoch {epoch:5d}/{self.epochs}", position=0, leave=False)
        ):
            images, _ = batch
            images = images.to(self.device)

            # Forwarding
            pred_images = self.model(images)

            # Reconstruction loss (MSE loss)
            mse = functional.mse_loss(pred_images, images)

            # Backward
            self.__backward(mse)

            # Store the loss
            __epoch_mse_loss += mse

        return __epoch_mse_loss.mean().item()

    def __backward(self, loss):
        """
        Backward the loss and update weights
        :param loss:
        :return:
        """
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def __step_batch(self):
        """
        Take step after one back, to be implemented later
        :return:
        """
        return None

    def __step_epoch(self, epoch_mse_loss):
        """
        Take step after one epoch
        :return:
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch_mse_loss)

    @torch.no_grad()
    def reconstruct_sample(self):
        """
        Reconstruct a random image
        :return:
        """
        sample_image, _ = next(iter(self.sample_loader))
        sample_image = sample_image.to(self.device)
        return sample_image, self.reconstruct(sample_image)

    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct an image
        :param x:
        :return:
        """
        return self.model(x)
