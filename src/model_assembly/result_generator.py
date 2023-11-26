import gc
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from models.nets.unet_v2 import UNet_v2
from models.nets.up_scaler import UpScaler
from utils import (
    load_checkpoint,
    cosine_schedule,
    sigmoid_schedule,
    quadratic_schedule,
    linear_schedule,
    de_normalise,
)


class Generator:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 1,
        run_name: str = "Generator",
        variance_schedule_type: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        noise_steps: int = 1000,
        embedding_dim: int = 256,
        attn_heads: int = 1,
        ddpm_checkpoint: str = "",
        upscaler_checkpoint: str = "",
        upscaler_hidden_channels: int = 128,
    ) -> None:
        super().__init__()

        # Platform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        # Load model
        # DDPM
        self.ddpm_model = UNet_v2(
            in_channels=3,
            out_channels=3,
            attn_heads=attn_heads,
            embedded_dim=embedding_dim,
        ).to(self.device)

        ddpm_checkpoint = load_checkpoint(ddpm_checkpoint, str(self.device))
        self.ddpm_model.load_state_dict(ddpm_checkpoint["model"])

        self.noise_steps = noise_steps

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

        # Embedding dims
        self.embedding_dim = embedding_dim

        # Load model
        # Image upscaler
        self.upscale_model = UpScaler(
            hidden_channels=upscaler_hidden_channels,
            middle_activation="Tanh",
            output_module="sub-pix",
        )

        upscaler_checkpoint = load_checkpoint(upscaler_checkpoint, str(self.device))
        self.upscale_model.load_state_dict(upscaler_checkpoint["model"])

        # Clear memory
        gc.collect()

    @torch.inference_mode()
    def generate_all(self, result_dir: str = ""):
        self.ddpm_model.eval()
        for _, batch in enumerate(
            tqdm(
                self.dataloader,
                desc=f"Generating all images based on embeddings...",
                position=0,
                leave=False,
            )
        ):
            file_names, images = self.__ddpm_generate_batch(batch)
            images = self.__upscale_batch(images)
            for i, image in enumerate(images):
                # Save each image to the specified directory
                save_image(image, os.path.join(result_dir, f"{file_names[i]}.png"))

    @torch.inference_mode()
    def __upscale_batch(self, images_batch):
        images_batch = images_batch.to(self.device)
        return self.upscale_model(images_batch)

    @torch.inference_mode()
    def __ddpm_generate_batch(self, batch):
        file_names, semantics = batch
        semantics = semantics.to(self.device)

        # Get random gaussian noise with the shape of image
        images = torch.randn((semantics.size(0), 3, 64, 64)).to(self.device)

        # Iterate through steps
        for _, time_step in enumerate(
                tqdm(
                    reversed(range(1, self.noise_steps)),
                    position=0,
                    leave=True,
                )
        ):
            # Numeric timestep
            t = torch.full(
                size=(semantics.size(0),), fill_value=time_step, dtype=torch.long
            ).to(self.device)

            # Conditioning
            time_step_embeddings = self.step_embeddings(t)
            embeddings = time_step_embeddings + semantics

            # Forward
            pred_noise = self.ddpm_model(images, embeddings)

            # Denoise parameters
            __alpha = self.alpha[t][:, None, None, None]
            __alpha_cumulative = self.alpha_cumulative[t][:, None, None, None]
            __beta = self.beta[t][:, None, None, None]

            # Iteration noises
            if time_step > 1:
                iteration_noise = torch.randn_like(images)
            else:
                iteration_noise = torch.zeros_like(images)

            # Denoising algorithms
            images = (
                    1
                    / torch.sqrt(__alpha)
                    * (
                            images
                            - ((1 - __alpha) / (torch.sqrt(1 - __alpha_cumulative)))
                            * pred_noise
                    )
                    + torch.sqrt(__beta) * iteration_noise
            )

        return file_names, de_normalise(images, self.device)

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
