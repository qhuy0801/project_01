import albumentations
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def forward_transform(image, target_size):
    """
    Simple transformation which include normalisation and resizing
    :param image:
    :param target_size:
    :return: CHW tensors
    """
    __transform_pipeline = albumentations.Compose(
        [
            albumentations.Normalize(MEAN, STD),
            albumentations.Resize(
                target_size, target_size, interpolation=cv2.INTER_NEAREST
            ),
            ToTensorV2(),
        ]
    )
    return __transform_pipeline(image=image)["image"]


def revert_transform(image_tensor):
    """
    Revert transformation which include de-normalisation, convert from float32 to uint8
    :param image_tensor: single CHW tensor
    :return: HWC numpy matrix
    """
    __mean = torch.tensor(MEAN).view(3, 1, 1)
    __std = torch.tensor(STD).view(3, 1, 1)
    return (
        ((image_tensor * __std + __mean) * 225)
        .numpy()
        .transpose(1, 2, 0)
        .astype(np.uint8)
    )


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


def linear_noise_schedule(start, end, steps):
    """
    Linear noise schedule for diffusion model
    :return:
    """
    return torch.linspace(start, end, steps)


def to_uint8(image_tensor):
    """
    Convert float image matrix to uint8
    :param image_tensor:
    :return:
    """
    return (image_tensor * 255).round().astype("uint8")
