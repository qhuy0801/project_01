import torch
from torch.utils.data import Dataset, DataLoader

from models.embeddings.embedder import Embedder
from models.nets.u_net import UNet


class Diffuser:
    """
    The implementation of diffuser
    """

    # Training dependencies
    device: torch.device
    model: torch.nn.Module
    loss_func: torch.nn.modules.loss._Loss
    optimiser: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler

    # Data
    train_data: torch.utils.data.DataLoader
    validate_data: torch.utils.data.DataLoader

    # Embedder
    embedder: Embedder
    embedded_dim: int = 256

    # Parameters
    # Model settings
    image_size: int = 512
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
        train_data: DataLoader,
        embedder: Embedder,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        noise_steps: int = 1000,
        max_lr: float = 1e-4,
        eps: float = 1e-5,
        epochs: int = 1000,
        **kwargs
    ) -> None:
        super().__init__()
        self.train_data = train_data
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
