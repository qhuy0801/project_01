import os
from datetime import datetime
import bitsandbytes as bnb

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from models.nets.autoencoder import SuperResAE
from models.nets.dual_decoder import DualDecoder
from utils import save_checkpoint, de_normalise


class DecoderTrainer:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 8,
        num_samples: int = 1,
        middle_channels: int = 256,
        max_lr: float = 1e-4,
        eps: float = 1e-8,
        epochs: int = 5000,
        run_name: str = "decoder",
        output_dir: str = "./output/",
        additional_note: str = "",
    ) -> None:
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
        self.model = DualDecoder().to(self.device)

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
        model_stats = str(summary(self.model, (1, 3, 256, 256)))
        with open(
            f"{os.path.join(self.run_dir, self.run_time)}/model.txt", "w"
        ) as file:
            file.write(model_stats)
            file.write(
                f"\nTotal epochs: {self.epochs}"
                f"\nMax learning rate: {max_lr}"
                f"\nBatch size: {batch_size}"
                f"\nNum workers: {num_workers}"
                f"\nAdditional note: {additional_note}"
            )
            file.close()

        # Best epoch loss
        self.best_loss: float = 2000.0

    def fit(self, sample_every: int = 20):
        # Training mode
        self.model.train()

        print(f"Starting training {self.run_name} for {self.epochs} epochs...")

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
                sample = self.sample()
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
            # Unpack the batch
            img_s, img_l, _ = batch
            img_s, img_l = img_s.to(self.device), img_l.to(self.device)

            # Forwarding
            pred_img_l = self.model(img_s)

            # Loss
            loss = functional.mse_loss(pred_img_l, img_l)

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

    def sample(self):
        # Get a random embedding from dataset
        img_s, img_l, _ = next(iter(self.sample_loader))
        img_s = img_s.to(self.device)
        img_l = img_l.to(self.device)

        # Display
        display = [
            de_normalise(
                functional.interpolate(
                    img_s, size=(256, 256), mode="bilinear", align_corners=False
                ),
                self.device,
            ),
            de_normalise(img_l, self.device),
        ]

        # Put model in training mode
        self.model.eval()

        # Forward
        pred_img_l = self.model(img_s)

        # Append the result to display
        display.append(
            de_normalise(
                pred_img_l,
                self.device,
            )
        )

        # Concatenation for displaying
        display = torch.cat(display, dim=0)

        return display
