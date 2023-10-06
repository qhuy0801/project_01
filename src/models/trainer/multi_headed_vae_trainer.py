import datetime
import itertools
import os

import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from models.nets.vae import VAE
from models.trainer.vae_trainer import VAETrainer


class MultiheadVAETrainer(VAETrainer):
    def __init__(
        self,
        train_dataset: Dataset,
        model: VAE,
        batch_size: int = 32,
        num_workers: int = 8,
        num_samples: int = 1,
        epochs: int = 5000,
        max_lr: float = 1e-4,
        min_lr: float = 5e-6,
        lr_decay: float = 0.999,
        lr_threshold: float = 0.2,
        patience_lr: int = 30,
        max_lr_additional: float = 1e-4,
        min_lr_additional: float = 5e-6,
        lr_decay_additional: float = 0.999,
        lr_threshold_additional: float = 0.2,
        patience_lr_additional: int = 30,
        run_name: str = "multi_headed_vae",
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
        # Main VAE
        main_vae_params = itertools.chain(
            self.model.encoder,
            self.model.decoder,
            self.model.fc_mu,
            self.model.fc_sigma,
            self.model.decoder_input,
        )
        self.optimiser = bnb.optim.AdamW(main_vae_params, lr=max_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimiser,
            mode="min",
            factor=lr_decay,
            threshold=lr_threshold,
            min_lr=min_lr,
            patience=patience_lr,
        )
        # Additional decoder
        self.optimiser_additional = bnb.optim.AdamW(
            self.model.additional_decoder_1, lr=max_lr
        )
        self.lr_scheduler_additional = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimiser_additional,
            mode="min",
            factor=lr_decay_additional,
            threshold=lr_threshold_additional,
            min_lr=min_lr_additional,
            patience=patience_lr_additional,
        )

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
                    self.model.input_size,
                    self.model.input_size,
                ),
                verbose=0,
            )
        )
        with open(
            f"{os.path.join(self.run_dir, self.run_time)}/model.txt", "w"
        ) as file:
            file.write(model_stats)
            file.write(f"\nInput_size: {self.model.input_size}")

        # Best loss for checkpoints
        self.best_mse_loss = 2000.0
