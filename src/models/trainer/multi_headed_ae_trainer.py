import gc
from datetime import datetime
import itertools
import os

import torch
import bitsandbytes as bnb
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from models.nets.vae_v4 import Multi_headed_AE
from utils import de_normalise, save_checkpoint, load_checkpoint


class MultiheadAETrainer:
    def __init__(
        self,
        train_dataset: Dataset,
        model: Multi_headed_AE,
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
        simple_ae_checkpoint: str = None,
        run_name: str = "multi_headed_ae",
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
        # Main AE
        main_vae_params = []
        main_vae_params.extend(self.model.encoder.parameters())
        main_vae_params.extend(self.model.decoder.parameters())
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
            self.model.additional_decoder.parameters(), lr=max_lr_additional
        )
        self.lr_scheduler_additional = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimiser_additional,
            mode="min",
            factor=lr_decay_additional,
            threshold=lr_threshold_additional,
            min_lr=min_lr_additional,
            patience=patience_lr_additional,
        )

        # In the case that there is a simple AE checkpoint provided (1 encoder - 1 decoder)
        # We will load the weights the then only train the additional decoder
        if simple_ae_checkpoint is not None:
            checkpoint = load_checkpoint(simple_ae_checkpoint, str(self.device))
            self.model.load_state_dict(checkpoint["model"].state_dict(), strict=False)

            # Then we will only need the additional optimiser and lr scheduler
            self.optimiser = None
            self.lr_scheduler = None

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
            file.write(model_stats)
            file.write(f"\nInput_size: {self.train_dataset.target_tensor_size}")

        # Best loss for checkpoints
        self.best_mse_loss = 2000.0

        # Clear
        gc.collect()

    def fit(self, sample_after: int = 100):
        self.model.train()
        print(f"Starting training {self.run_name} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_mse_loss, epoch_additional_mse_loss = self.__one_epoch(epoch)
            self.__step_epoch(epoch_mse_loss, epoch_additional_mse_loss)

            # Logs
            if self.optimiser is not None:
                self.log.add_scalar("Epoch/MSE_original", epoch_mse_loss, epoch)
                self.log.add_scalar(
                "Epoch/LR_original",
                self.optimiser.param_groups[0]["lr"],
                epoch,
                )
            self.log.add_scalar(
                "Epoch/MSE_additional", epoch_additional_mse_loss, epoch
            )
            self.log.add_scalar(
                "Epoch/LR_additional",
                self.optimiser_additional.param_groups[0]["lr"],
                epoch,
            )
            self.log.flush()

            # Save checkpoint
            if self.optimiser is None:
                epoch_mse_loss = epoch_additional_mse_loss
            if epoch_mse_loss < self.best_mse_loss:
                self.best_mse_loss = epoch_mse_loss
                save_checkpoint(
                    {"model": self.model.state_dict()},
                    self.run_name,
                    os.path.join(self.run_dir, self.run_time),
                )

            # Log samples
            if epoch % sample_after == 0:
                (
                    images,
                    reconstructed_img_org,
                    segment_residuals,
                    reconstructed_img,
                ) = self.__reconstruct()
                img_tensor = torch.cat(
                    (
                        de_normalise(images, self.device),
                        de_normalise(reconstructed_img_org, self.device),
                        de_normalise(segment_residuals, self.device),
                        de_normalise(reconstructed_img, self.device),
                    ),
                    dim=0,
                )
                self.log.add_scalars(
                    "Checkpoint/MSE_loss",
                    {
                        "Original": functional.mse_loss(
                            images, reconstructed_img_org
                        ).item(),
                        "Combined": functional.mse_loss(
                            images, reconstructed_img
                        ).item(),
                    },
                    epoch,
                )
                self.log.add_images(
                    tag=f"Samples/Random/Epoch:{epoch}",
                    img_tensor=img_tensor,
                    global_step=epoch,
                    dataformats="NCHW",
                )

    def __one_epoch(self, epoch):
        # Store the epoch loss
        __epoch_mse_loss = 0.0
        __epoch_additional_mse_loss = 0.0

        # Iterate the dataloader
        for _, batch in enumerate(
            tqdm(
                self.train_data,
                desc=f"Epoch {epoch:5d}/{self.epochs}",
                position=0,
                leave=False,
            )
        ):
            images, segments = batch
            images = images.to(self.device)

            # Original AE branch
            # Forward
            pred_original = self.model.original_forward(images)
            # If the optimiser of basic branch is None -> retrain additional encoder only:
            if self.optimiser is not None:
                # Loss
                mse_original = functional.mse_loss(pred_original, images)
                # Backward
                self.optimiser.zero_grad()
                mse_original.backward()
                self.optimiser.step()

            # Additional decoder
            # Target (residual between target image and predicted image - original branch)
            target_additional = self.__get_segment_residual(
                images, pred_original, segments, 1
            )
            # Forward
            pred_additional = self.model.additional_forward(images)
            # Loss
            mse_additional = functional.mse_loss(pred_additional, target_additional)
            # Backward
            self.optimiser_additional.zero_grad()
            mse_additional.backward()
            self.optimiser_additional.step()

            # Store the loss
            if self.optimiser is not None:
                __epoch_mse_loss += mse_original
                __epoch_additional_mse_loss += mse_additional
                return __epoch_mse_loss.mean().item(), __epoch_additional_mse_loss.mean().item()
            __epoch_additional_mse_loss += mse_additional
            return None, __epoch_additional_mse_loss.mean().item()

    def __get_segment_residual(
        self, images, pred_images, segmentations, selected_class
    ):
        residual = images - pred_images
        residual = residual.detach().clone()
        segmentations = torch.tensor(segmentations, dtype=torch.float32).to(self.device)
        segmentations = (segmentations == selected_class).float().unsqueeze(1)
        return residual * segmentations

    def __step_epoch(self, epoch_loss, epoch_additional_loss):
        if epoch_loss is not None:
            self.lr_scheduler.step(epoch_loss)
        if epoch_additional_loss is not None:
            self.lr_scheduler_additional.step(epoch_additional_loss)

    @torch.no_grad()
    def __reconstruct(self):
        images, segments = next(iter(self.sample_loader))
        images = images.to(self.device)
        reconstructed_img_org = self.model.original_forward(images)
        segment_residuals = self.__get_segment_residual(
            images, reconstructed_img_org, segments, 1
        )
        reconstructed_img = self.model(images)
        return images, reconstructed_img_org, segment_residuals, reconstructed_img
