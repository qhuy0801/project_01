import torch
from fastprogress import progress_bar
from torch.utils.data import Dataset, DataLoader

from models.embeddings.embedder import Embedder
from models.nets.u_net import UNet
from utils import linear_noise_schedule


class Diffuser:
    """
    In this implementation of diffuser, we hard-coded the dependencies.
    We let the training parameters open for customisation

    This implementation follows the mechanism of Denoising Diffusion Probabilistic Model
    (DDPM).
    """

    # Training dependencies
    device: torch.device
    model: torch.nn.Module
    loss_func: torch.nn.modules.loss._Loss
    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    # Data
    train_data: torch.utils.data.DataLoader
    validate_data: torch.utils.data.DataLoader = None

    # Embedder
    embedder: Embedder
    embedded_dim: int = 256

    # Parameters
    # Model settings
    in_channels: int = 3
    out_channels: int = 3

    # Noise settings
    beta_start: float
    beta_end: float
    noise_steps: int

    # Training settings
    max_lr: float
    eps: float
    epochs: int

    def __init__(
        self,
        train_dataset: Dataset,
        embedder: Embedder,
        batch_size: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        noise_steps: int = 1000,
        max_lr: float = 1e-4,
        eps: float = 1e-5,
        epochs: int = 1000,
        **kwargs,
    ) -> None:
        """
        Constructor
        :param train_data:
        :param embedder:
        :param beta_start:
        :param beta_end:
        :param noise_steps:
        :param max_lr:
        :param eps:
        :param epochs:
        :param kwargs:
        """
        super().__init__()
        self.train_data = DataLoader(train_dataset, batch_size=batch_size)
        self.embedder = embedder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            embedded_dim=self.embedded_dim,
        ).to(self.device)
        self.loss_func = torch.nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler()

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.beta = linear_noise_schedule(
            self.beta_start, self.beta_end, self.noise_steps
        ).to(self.device)

        self.alpha = 1.0 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)

        self.max_lr = max_lr
        self.eps = eps
        self.epochs = epochs

        self.optimiser = torch.optim.AdamW(
            params=self.model.parameters(), lr=self.max_lr, eps=self.eps
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimiser,
            max_lr=self.max_lr,
            steps_per_epoch=len(self.train_data),
            epochs=self.epochs,
        )

        self.__dict__.update(kwargs)

    def fit(self):
        """
        Training trigger
        :return:
        """
        for epoch in progress_bar(range(self.epochs), total=self.epochs, leave=True):
            # Print log # TODO: update this
            print(f"Epoch: {epoch}")
            train_loss = self.one_epoch(is_training=True)
            print(f"Epoch train loss: {train_loss}")

            # Validation
            if self.validate_data is not None:
                validation_loss = self.one_epoch(is_training=False)
                print(f"Epoch validation loss: {validation_loss}")

    def one_epoch(self, is_training: bool = True):
        """
        Start training or inferring for one epoch
        :param is_training:
        :return:
        """
        # Storing the loss
        epoch_loss = []

        # Switch model to correct mode
        if is_training:
            self.model.train()
            data = self.train_data
        else:
            self.model.eval()
            data = self.validate_data

        # Breakdown data using progress bar
        batches = progress_bar(data, leave=False)
        for _, (images, semantics) in enumerate(batches):
            with torch.autocast(self.device.type) and (
                torch.enable_grad() if is_training else torch.inference_mode()
            ):
                # Transfer data to devices
                images = images.to(self.device)
                semantics = semantics.to(self.device)

                # Get a random timestep for each image
                timesteps = torch.randint(
                    low=1, high=self.noise_steps, size=(images.shape[0],)
                )

                # Add noise to image
                noised_images, noises = self.add_noise(images, timesteps)

                # Conditioning
                conditions = self.embedder.combine_embedding(
                    self.embedder.step_embedding(timesteps),
                    self.embedder.semantic_embedding(semantics),
                )

                # Push noised image to the network
                pred_noises = self.model(noised_images, conditions)

                # Loss function
                loss = self.loss_func(noises, pred_noises)
                print(loss.item())

                # Store loss
                epoch_loss.append(loss.item())

            if is_training:
                self.backward(loss)

        return epoch_loss

    def backward(self, loss):
        """
        Back-propagation and weights updating
        :param loss:
        :return:
        """
        self.optimiser.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimiser)
        self.scaler.update()
        self.scheduler.step()

    def add_noise(self, image, timestep):
        """
        Add noise to image based on timestep
        :param image:
        :param timestep:
        :return:
        """
        epsilon = torch.randn_like(image)
        return (
            torch.sqrt(self.alpha_cumulative[timestep])[:, None, None, None] * image
            + torch.sqrt(1 - self.alpha_cumulative[timestep])[:, None, None, None]
            * epsilon,
            epsilon,
        )
