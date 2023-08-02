import cv2
import albumentations as A
import numpy as np
import torch
import matplotlib.pyplot as plt

from entities.Image import Image
from utils.config_utils import read_config
from utils.data_utils import init_images
from utils.model_utils import load_pt, load_onnx, init_session_onnx
from utils.visualise_utils import visualise_segmentation

CONFIG_FILE = "../../config/segmentation-config.yml"


def transform_image(_target_height: int, _target_width: int):
    """

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
    _image_instance.load_img()
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
    config = read_config(CONFIG_FILE)
    input_height = config["model"]["input_size"]["height"]
    input_width = config["model"]["input_size"]["width"]
    transform = transform_image(input_height, input_width)

    images = init_images(config)
    segmented_matrices = []

    # model = load_pt(pt_path=config["model"]["path"], device=config["device"])

    # print(model)

    onnx_model = load_onnx(_onnx_path=config["model"]["path"])
    session = init_session_onnx(onnx_model)

    for image in images:
        matrix = np.expand_dims(initialise_image_matrix(image, transform), axis=0).transpose((0, 3, 1, 2))
        # segmented_matrix = get_segmentation_matrix(model, matrix)
        segmented_matrix = session.run(None, {"input": matrix})
        segmented_matrices.append(segmented_matrix)

    visualise_segmentation(images[1].img, segmented_matrices[1])
