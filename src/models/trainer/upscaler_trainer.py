import os
from datetime import datetime
import bitsandbytes as bnb
import numpy as np

import torch
import torch.nn.functional as functional
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from models.nets.up_scaler import UpScaler
from utils import de_normalise, psnr, save_checkpoint


class UpscalerTrainer:
    """
    The trainer class of image up-scaler model
    Which include training trigger, checkpoint saving functions
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 8,
        num_samples: int = 1,
        hidden_channels: int = 64,
        middle_activation: str = "ReLU",
        output_module: str = "sub-pix",
        max_lr: float = 1e-4,
        eps: float = 1e-8,
        epochs: int = 5000,
        run_name: str = "decoder",
        output_dir: str = "./output/",
        wandb_run: Run = None,
        additional_note: str = "",
    ) -> None:
        """
        Constructor
        :param dataset:
        :param batch_size:
        :param num_workers:
        :param num_samples:
        :param hidden_channels:
        :param middle_activation:
        :param output_module:
        :param max_lr:
        :param eps:
        :param epochs:
        :param run_name:
        :param output_dir:
        :param wandb_run:
        :param additional_note:
        """
        super().__init__()
        # Platform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            data_source=self.dataset, replacement=True, num_samples=num_samples
        )
        self.sample_loader = DataLoader(
            dataset=self.dataset, batch_size=1, sampler=random_sampler
        )

        # Model
        self.model = UpScaler(
            hidden_channels=hidden_channels,
            middle_activation=middle_activation,
            output_module=output_module,
        ).to(self.device)

        # Number of epochs
        self.epochs = epochs

        # Learning rate, decay, optimiser and scheduler
        self.epochs = epochs
        self.optimiser = bnb.optim.AdamW(self.model.parameters(), lr=max_lr, eps=eps)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimiser,
            max_lr=max_lr,
            steps_per_epoch=len(self.dataloader),
            epochs=self.epochs,
        )

        # Logger
        self.log = SummaryWriter(
            log_dir=f"{os.path.join(self.run_dir, self.run_time)}/logs/"
        )

        # Log models and training stats
        self.model.eval()
        model_stats = str(summary(self.model, (1, 3, 64, 64)))
        with open(
            f"{os.path.join(self.run_dir, self.run_time)}/model.txt", "w"
        ) as file:
            file.write(model_stats)
            file.write(
                f"\nTotal epochs: {self.epochs}"
                f"\nMax learning rate: {max_lr}"
                f"\nMiddle activation method: {middle_activation}"
                f"\nOutput model: {output_module}"
                f"\nHidden channels: {hidden_channels}"
                f"\nBatch size: {batch_size}"
                f"\nNum workers: {num_workers}"
                f"\nAdditional note: {additional_note}"
            )
            file.close()

        # Wandb run
        self.wandb_run = wandb_run

        # Best epoch loss
        self.best_psnr: float = 0.0

    def fit(self, sample_every: int = 100):
        """
        Training trigger
        :param sample_every:
        :return:
        """
        # Training mode
        self.model.train()

        print(f"Starting training {self.run_name} for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            epoch_psnr = self.one_epoch(epoch)

            # Logs
            self.log.add_scalar("Epoch/PSNR", epoch_psnr, epoch)
            if self.scheduler is not None:
                self.log.add_scalar(
                    "Epoch/Learning_rate",
                    self.optimiser.param_groups[0]["lr"],
                    epoch,
                )
            self.log.flush()

            # WandB log
            if self.wandb_run is not None:
                wandb.log({"psnr": epoch_psnr})
                if self.scheduler is not None:
                    wandb.log({"learning_rate": self.optimiser.param_groups[0]["lr"]})

                    # Log samples
            if epoch % sample_every == 0:
                sample = self.sample()
                self.log.add_images(
                    tag=f"Samples/Random/Epoch:{epoch}",
                    img_tensor=sample,
                    global_step=epoch,
                    dataformats="NCHW",
                )

            # Save checkpoint if best PSNR archived
            if epoch_psnr > self.best_psnr:
                self.best_psnr = epoch_psnr
                save_checkpoint(
                    {"model": self.model.state_dict()},
                    self.run_name,
                    os.path.join(self.run_dir, self.run_time),
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
        epoch_psnr = []

        # One batch iteration
        for _, batch in enumerate(
            tqdm(
                self.dataloader,
                desc=f"Epoch {epoch:5d}/{self.epochs}",
                position=0,
                leave=False,
            )
        ):
            # Unpack the batch
            img_s, img_l, _ = batch
            img_s, img_l = de_normalise(
                img_s.to(self.device), self.device
            ), de_normalise(img_l.to(self.device), self.device)

            # Forwarding
            pred_img_l = self.model(img_s)

            # Loss
            loss = functional.mse_loss(pred_img_l, img_l)
            epoch_psnr.append(psnr(pred_img_l, img_l))

            # Take step
            self.step(loss)

        return np.mean(epoch_psnr)

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

    def sample(self):
        """
        Get a random low-resolution image and upscale it
        :return:
        """
        # Get a random embedding from dataset
        img_s, img_l, _ = next(iter(self.sample_loader))
        img_s = de_normalise(img_s.to(self.device), self.device)
        img_l = de_normalise(img_l.to(self.device), self.device)

        # Display
        display = [
            functional.interpolate(
                img_s, size=(256, 256), mode="bilinear", align_corners=False
            ),
            img_l,
        ]

        # Put model in training mode
        self.model.eval()

        # Forward
        pred_img_l = self.model(img_s)

        # Append the result to display
        display.append(pred_img_l)

        # Concatenation for displaying
        display = torch.cat(display, dim=0)

        return display
