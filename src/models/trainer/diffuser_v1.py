import torch
from fastprogress import progress_bar
from torch.utils.data import Dataset, DataLoader, RandomSampler

from entities.data.image_dataset import ImageDataset
from models.embeddings.embedder import Embedder
from models.nets.u_net import UNet
from utils import linear_noise_schedule, revert_transform
from utils.visualise_utils import plot_hwc


class Diffuser_v1:
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

    # Sampler
    sampler: torch.utils.data.DataLoader
    sample_num: int = 4

    # Embedder
    embedder: Embedder
    embedded_dim: int = 256

    # Parameters
    # Model settings
    image_size: int
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
        train_dataset: ImageDataset,
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
        # Data setting
        self.train_data = DataLoader(train_dataset, batch_size=batch_size)
        self.sampler = DataLoader(
            train_dataset,
            batch_size=self.sample_num,
            sampler=RandomSampler(
                train_dataset, replacement=True, num_samples=self.sample_num
            ),
        )
        self.image_size = train_dataset.target_size

        # Embedding
        self.embedder = embedder

        # Devices and models
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
        #
        # for epoch in progress_bar(range(self.epochs), total=self.epochs, leave=True):
        #     # Print log # TODO: update this
        #     print(f"Epoch: {epoch}")
        #     train_loss = self.one_epoch(is_training=True)
        #     print(f"Epoch train loss: {train_loss}")
        #
        #     # Validation
        #     if self.validate_data is not None:
        #         validation_loss = self.one_epoch(is_training=False)
        #         print(f"Epoch validation loss: {validation_loss}")

        # for _ in range(5):
        #     self.one_epoch(is_training=True)
        #
        # labels = torch.Tensor([1., 3., 0., 2.]).long()
        # images = self.sample(labels)
        # image = images[0]
        # plot_hwc(revert_transform(image))

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
        for _, batch in enumerate(batches):
            with torch.autocast(self.device.type) and (
                torch.enable_grad() if is_training else torch.inference_mode()
            ):
                semantics, time_steps, noised_images, noises = self.prepare_batch(batch)

                # Conditioning
                conditions = self.embedder.combine_embedding(
                    self.embedder.step_embedding(time_steps),
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

    def prepare_batch(self, batch):
        """
        Prepare the training input for one extracted batch
        :param batch:
        :return:
        """
        # Unpack batch
        images, semantics = batch

        # Transfer data to devices
        images = images.to(self.device)
        semantics = semantics.to(self.device)

        # Get a random timestep for each image
        time_steps = torch.randint(low=1, high=self.noise_steps, size=(images.shape[0],))

        # Add noise to image
        noised_images, noises = self.add_noise(images, time_steps)

        return semantics, time_steps, noised_images, noises

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

    @torch.inference_mode()
    def sample(self, labels, interpolate_weight: int = 0):
        """

        :param labels:
        :param interpolate_weight:
        :return:
        """
        # Number of samples
        sample_num = len(labels)

        # Get the model into correct mode
        self.model.eval()
        with torch.inference_mode():
            # Initial noised image (input) in generation (sampling) process is pure gaussian noise
            images = torch.randn(
                (sample_num, self.in_channels, self.image_size, self.image_size)
            ).to(self.device)

            # Iterate time-steps from maximum time-steps to 0
            for i in progress_bar(
                reversed(range(1, self.noise_steps)),
                total=self.noise_steps - 1,
                leave=False,
            ):
                # Get the time-steps tensor based on current iteration
                time_steps = (torch.ones(sample_num) * i).long().to(self.device)

                # Conditioning
                conditions = self.embedder.combine_embedding(
                    self.embedder.step_embedding(time_steps),
                    self.embedder.semantic_embedding(labels),
                )

                # Forward
                pred_noises = self.model(images, conditions)

                # If there is interpolation weight, we will perform alignment
                if interpolate_weight > 0:
                    non_semantic_pred_noises = self.model(
                        images, self.embedder.step_embedding(time_steps)
                    )
                    pred_noises = self.align_prediction(
                        non_semantic_pred_noises, pred_noises, interpolate_weight
                    )

                # If time-steps > 1, we still need to add gaussian noise to each iteration
                if i > 1:
                    iteration_noises = torch.randn_like(images)
                else:
                    iteration_noises = torch.zeros_like(images)

                images = self.denoise(images, pred_noises, iteration_noises, time_steps)

        return images

    def add_noise(self, images, time_steps):
        """
        Add noise to image based on timestep
        :param images:
        :param time_steps:
        :return:
        """
        epsilon = torch.randn_like(images)
        return (
            torch.sqrt(self.alpha_cumulative[time_steps])[:, None, None, None] * images
            + torch.sqrt(1 - self.alpha_cumulative[time_steps])[:, None, None, None]
            * epsilon,
            epsilon,
        )

    def denoise(self, noised_images, pred_noises, iteration_noises, time_steps):
        """
        Perform 1 iteration of denoising
        :param noised_images: at time-step t
        :param pred_noises: predicted noise at time-step t
        :param iteration_noises: random noise added in this iteration
        :param time_steps: t
        :return: images at time-step t-1
        """
        return (
            (
                1
                / torch.sqrt(self.alpha[time_steps][:, None, None, None])
                * (
                    noised_images
                    - (
                        (1 - self.alpha[time_steps][:, None, None, None])
                        / (
                            torch.sqrt(
                                1
                                - self.alpha_cumulative[time_steps][:, None, None, None]
                            )
                        )
                    )
                    * pred_noises
                )
                + torch.sqrt(self.beta[time_steps][:, None, None, None])
                * iteration_noises
            ).clamp(-1, 1)
            + 1
        ) / 2

    def align_prediction(self, no_semantic_pred_noise, pred_noise, weight):
        """
        Perform interpolation (alignment) on time-steps conditioned prediction with
        (semantics + time-steps) conditioned prediction
        :param no_semantic_pred_noise:
        :param pred_noise:
        :param weight:
        :return:
        """
        return torch.lerp(no_semantic_pred_noise, pred_noise, weight)
