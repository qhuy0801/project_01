import gc
import os
from datetime import datetime
import bitsandbytes as bnb
from torchsummary import summary

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.nets.vae import VAE
from utils import save_checkpoint, de_normalise, load_checkpoint


class VAETrainer:
    def __init__(
            self,
            train_dataset: Dataset,
            model: VAE,
            batch_size: int = 10,
            checkpoint_path: str = None,
            num_workers: int = 16,
            num_samples: int = 1,
            epochs: int = 5000,
            max_lr: float = 1e-4,
            lr_decay: float = 0.999,
            run_name: str = "vae",
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
            train_dataset, batch_size=batch_size, num_workers=num_workers
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
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimiser,
            step_size=len(self.train_data),
            gamma=lr_decay,
            last_epoch=-1,
        )

        # If there is back-up
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(checkpoint_path, str(self.device))
            self.model.load_state_dict(checkpoint["model"])
            self.optimiser.load_state_dict(checkpoint["optimiser"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            # Un-reference and clear
            checkpoint = None
            gc.collect()

        # Logger
        self.log = SummaryWriter(
            log_dir=f"{os.path.join(self.run_dir, self.run_time)}/logs/"
        )

        # Log information
        self.model.eval()
        summary(
            self.model,
            (
                self.model.dims[0][0],
                self.model.intput_size,
                self.model.intput_size,
            ),
        )

        # Step count
        self.current_step = 0

        # Best loss for checkpoints
        self.best_mse_loss = 2000.0

    def fit(self):
        self.model.train()
        print(f"Starting training {self.run_name} for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            epoch_kl_loss, epoch_mse_loss = self.__one_epoch(epoch)
            self.__step_epoch()

            # Logs
            self.log.add_scalar("Epoch_loss/KL+BCE", epoch_kl_loss, self.current_step)
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

                # Log a reconstructed image
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
                    global_step=self.current_step,
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
        __epoch_loss = 0.0
        __epoch_mse_loss = 0.0

        # Iterate the dataloader
        for _, batch in enumerate(tqdm(self.train_data, desc=f"Epoch {epoch:5d}")):
            images, segment = batch
            images = images.to(self.device)

            # Forwarding
            pred_images, mu, sigma = self.model(images)

            # Losses
            # KL
            kl = 0.5 * torch.sum(-1 - sigma + mu.pow(2) + sigma.exp())

            # BCE
            bce = functional.binary_cross_entropy(
                pred_images, images, size_average=False
            )

            # Reconstruction loss (MSE loss)
            mse = functional.mse_loss(pred_images, images)

            # Total loss
            loss = kl + bce

            # Logs
            self.log.add_scalar("Batch_loss/KL_loss", kl.item(), self.current_step)
            self.log.add_scalar("Batch_loss/MSE_loss", mse.item(), self.current_step)
            self.log.add_scalar("Batch_loss/BCE_loss", bce.item(), self.current_step)
            self.log.add_scalar("Batch_loss/KL+BCE", loss, self.current_step)
            self.log.add_scalar(
                "Learning_rate",
                self.lr_scheduler.get_last_lr()[0],
                self.current_step,
            )
            self.log.flush()

            # Backward
            self.__backward(loss)

            # Step count
            self.current_step += 1

            # Store the loss
            __epoch_loss += loss
            __epoch_mse_loss += mse

        return __epoch_loss.mean().item(), __epoch_mse_loss.mean().item()

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

    def __step_epoch(self):
        """
        Take step after one epoch
        :return:
        """
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

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
    def random_sample(self, sample_num):
        sample = torch.randn(sample_num, self.model.latent_dim).to(self.device)
        return self.model.decode(sample)

    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct an image
        :param x:
        :return:
        """
        return self.model(x)[0]
