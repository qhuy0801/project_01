import torch
import copy

from models.generation.generation_v1.u_net.UNet import UNet_Conditional


class Diffuser:
    def __init__(
        self, _beta_start: int, _beta_end: int, _noise_step: int, _class_count: int
    ) -> None:
        super().__init__()
        # Input image setting
        # TODO

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
        :return:
        """
        epsilon = torch.randn_like(_x)
        return (
            torch.sqrt(self.alpha_cumulative[_t])[:, None, None, None] * _x
            + torch.sqrt(1 - self.alpha_cumulative[_t])[:, None, None, None] * epsilon,
            epsilon,
        )

    @torch.inference_mode()
    def sample(self, _labels, _ema: bool = True):
        # Model usage (EMA or direct UNet)
        model = self.ema_model if _ema else self.unet_model

        # Class count
