import albumentations
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def arr_to_tuples(dims_arr: [int]):
    """
    # Convert dimensions array to tuples:
    For example: [3, 4, 4, 4, 4] to [(3, 4), (4, 4), (4, 4), (4, 4)]
    :return:
    """
    return [(dims_arr[i], dims_arr[i + 1]) for i in range(len(dims_arr) - 1)]


def resize_segmentation(_segmentation_matrix, _size):
    """
    Resize the segmentation matrix to fit the images
    :param _segmentation_matrix:
    :param _size:
    :return:
    """
    return cv2.resize(_segmentation_matrix, _size, interpolation=cv2.INTER_NEAREST)


def get_segmentation(_sample, _session):
    """
    Get segmentation by pushing the sample through ONNX inference session
    :param _sample:
    :param _session:
    :return:
    """
    return np.argmax(_session.run(None, {"input": _sample})[0], axis=1)[0]


def forward_transform(image, target_size, to_tensor: bool = True):
    """
    Simple transformation which include normalisation and resizing
    :param image:
    :param target_size:
    :return: CHW tensors
    """
    __transform_steps = [
        albumentations.Normalize(MEAN, STD),
        albumentations.Resize(
            target_size, target_size, interpolation=cv2.INTER_NEAREST
        ),
    ]
    if to_tensor:
        __transform_steps = [*__transform_steps, ToTensorV2()]
    __transform_pipeline = albumentations.Compose(__transform_steps)
    return __transform_pipeline(image=image)["image"]


def de_normalise(image_tensor, device):
    """
    Revert transformation which include de-normalisation
    :param image_tensor: Tensor(NCHW)
    :return: Tensor(NCHW)
    """
    __mean = torch.tensor(MEAN).view(1, 3, 1, 1).to(device)
    __std = torch.tensor(STD).view(1, 3, 1, 1).to(device)
    return image_tensor * __std + __mean


def get_conv_output_size(input_size, kernel_size, stride, padding):
    """
    Calculate the output size of a convolutional layer
    :param input_size:
    :param kernel_size:
    :param stride:
    :param padding:
    :return:
    """
    return ((input_size - kernel_size + 2 * padding) // stride) + 1


def linear_schedule(start, end, steps):
    """
    Linear variance schedi
    :param start:
    :param end:
    :param steps:
    :return:
    """
    return torch.linspace(start, end, steps)


def quadratic_schedule(start, end, steps):
    """
    Quadratic variance schedule for diffusion model
    :param start:
    :param end:
    :param steps:
    :return:
    """
    return torch.linspace(start**0.5, end**0.5, steps) ** 2


def sigmoid_schedule(start, end, steps, sigmoid_max: float = 10.0):
    """
    Sigmoid schedule with default ending value is 10
    :param start:
    :param end:
    :param steps:
    :param sigmoid_max:
    :return:
    """
    arr = torch.linspace(-sigmoid_max, sigmoid_max, steps)
    return torch.sigmoid(arr) * (end - start) + start


def cosine_schedule(steps, s: float = 0.008):
    """
    Cosine variance schedule, original paper: https://arxiv.org/abs/2102.09672
    :param steps:
    :param s: set by 0.008 by default, similar to the original paper
    :return:
    """
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumulative = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumulative = alphas_cumulative / alphas_cumulative[0]
    betas = 1 - (alphas_cumulative[1:] / alphas_cumulative[:-1])
    return torch.clamp(betas, 0.001, 0.999)


def to_uint8(image_tensor):
    """
    Convert float image matrix to uint8
    :param image_tensor:
    :return:
    """
    return (image_tensor * 255).round().astype("uint8")
