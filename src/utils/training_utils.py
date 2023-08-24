import torch


def linear_noise_schedule(start, end, steps):
    """
    Linear noise schedule for diffusion model
    :return:
    """
    return torch.linspace(start, end, steps)
