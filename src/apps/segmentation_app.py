import albumentations as A
import cv2
import numpy as np
import torch

import constraints.SEG_CONST as CONSTRAINTS
from entities.Image import Image
from utils.data_utils import init_images


def transform_image(_target_height: int, _target_width: int):
    """
    Image transformation template
    :param _target_height:
    :param _target_width:
    :return:
    """
    return A.Compose(
        [
            A.Resize(_target_height, _target_width, interpolation=cv2.INTER_NEAREST),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def initialise_image_matrix(_image_instance: Image, _transform):
    """

    :param _image_instance:
    :param _transform:
    :return:
    """
    _image_instance.load_image()
    image_matrix = _image_instance.img
    return _transform(image=image_matrix)["image"]


def get_segmentation_matrix(_model, _matrix):
    """

    :param _model:
    :param _matrix:
    :return:
    """
    return np.argmax((_model(torch.Tensor(_matrix))).detach().numpy(), axis=1)


if __name__ == "__main__":
    # Get the images
    images = init_images(
        CONSTRAINTS.METADATA_PATH,
        CONSTRAINTS.IMAGES_DIR,
        CONSTRAINTS.SAMPLE_COUNT,
        CONSTRAINTS.FILENAME_COLUMN,
    )
    for image in images:
        image.load_image()

    print("Testing...")
