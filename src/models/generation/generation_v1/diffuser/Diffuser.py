import torch
import copy
import logging

from torch.utils.data import DataLoader
from torch import optim, nn
from fastprogress import progress_bar

from models.generation.generation_v1.ema.EMA import EMA
from models.generation.generation_v1.u_net.UNet import UNet_Conditional
from utils.io_utils import make_run_dir


class Diffuser:
    def __init__(
        self,
        _train_data: DataLoader,
        #_validation_data: DataLoader,
        _img_size: int,
        _device: str,
        _beta_start: float,
        _beta_end: float,
        _noise_step: int,
        _class_count: int,
    ) -> None:
        super().__init__()
        # Input setting
        self.train_data = _train_data
        #self.validation_data = _validation_data
        self.img_size = _img_size

        # Device
        self.device = _device

        # Parameters for noise schedule
        self.beta_start = _beta_start
        self.beta_end = _beta_end
        self.noise_step = _noise_step
        self.beta = self.linear_noise_schedule()  # TODO: add to device

        # Alpha
        self.alpha = 1.0 - self.beta
        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)

        # Models
        self.unet_model = UNet_Conditional(_class_count=_class_count)
        self.ema_model = copy.deepcopy(self.unet_model).eval().requires_grad_(False)

    def linear_noise_schedule(self):
        """
        Get noise schedule (array) in linear distribution based on
        - Start beta
        - End beta
        - Number of steps
        :return:
        """
        return torch.linspace(
            start=self.beta_start, end=self.beta_end, steps=self.noise_step
        )

    def add_noise(self, _x, _t):
        """
        Add noise to image
        :param _x: image
        :param _t: time step
        :return: noised image, noise layer added to t
        """
        epsilon = torch.randn_like(_x)
        return (
            torch.sqrt(self.alpha_cumulative[_t])[:, None, None, None] * _x
            + torch.sqrt(1 - self.alpha_cumulative[_t])[:, None, None, None] * epsilon,
            epsilon,
        )

    def random_timestep(self, _n):
        """
        Get a random timestep for each image in the batch
        :param _n: number of image in the batch
        :return:
        """
        return torch.randint(low=1, high=self.noise_step, size=(_n,))

    def setting_up(self, args):
        """

        :param args:
        :return:
        """
        # Make directory to store model and run output
        make_run_dir(args.run_name)

        # Get training setups
        self.optimiser = optim.AdamW(
            self.unet_model.parameters(), lr=args.learning_rate, eps=1e-5
        )

        # Scheduler (to update learning rate efficiently)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimiser,
            max_lr=args.learning_rate,
            steps_per_epoch=len(self.train_data),
            epochs=args.epochs,
        )

        # Loss function
        self.loss_func = nn.MSELoss()

        # Exponential Moving Average setting
        self.ema = EMA(0.99)

        # Gradient scaling
        self.scaler = torch.cuda.amp.GradScaler()

    def step_train(self, loss):
        # Reset the optimiser
        self.optimiser.zero_grad()

        # Scale the loss function and update the weights
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimiser)
        self.scaler.update()

        # Update EMA model
        self.ema.take_step(self.ema_model, self.unet_model)
        self.scheduler.step()

    def step_epoch(self, _is_training: bool = True):
        """
        Start 1 epoch of training or inferring
        :param _is_training:
        :return:
        """
        # Parameter for storing average loss
        epoch_loss = 0

        # Get the model in correct mode
        if _is_training:
            self.unet_model.train()
        else:
            self.unet_model.eval()

        # Training breakdown using progress bar
        bar = progress_bar(self.train_data, leave=False)
        for idx, (images, labels) in enumerate(bar):
            # Casting to GPU or and correct pytorch mode
            with torch.autocast(self.device) and (
                torch.enable_grad() if _is_training else torch.inference_mode()
            ):
                # Move images and labels to correct device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get a sample timestep
                timesteps = self.random_timestep(images.shape[0]).to(self.device)

                # Add noise to image
                noised_img, noise = self.add_noise(images, timesteps)

                # Push noised image to the network
                pred_noise = self.unet_model(noised_img, timesteps, labels)

                # Loss function
                loss = self.loss_func(noise, pred_noise)

                # Add to average loss
                epoch_loss += loss

            if _is_training:
                self.step_train(loss)

            bar.comment = f"MSE={loss.item():2.3f}"

        return epoch_loss

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            loss = self.step_epoch(_is_training=True)

            logging.info(f"Epoch {epoch}: loss: {loss}")

            # Log output after selected epochs
            # if epoch % args.log_every_epoch == 0:
            #     self.log_images()

        # save model
        # self.save_model(run_name=args.run_name, epoch=epoch)
