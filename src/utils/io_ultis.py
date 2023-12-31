import os

import onnx
import onnxruntime
import torch
from PIL import Image


def find_path(root_dir, filename):
    """
    To find the complete file path inside a directory using the file name only
    :param root_dir:
    :param filename:
    :return:
    """
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def save_pil_image(image_ndarray, file_name, location, file_extension=".jpg"):
    """
    Save an image
    :param image_ndarray:
    :param file_name:
    :param file_extension:
    :param location:
    :return:
    """
    img = Image.fromarray(image_ndarray)
    img.save(os.path.join(location, file_name + file_extension))


def save_checkpoint(checkpoint, run_name, directory):
    """
    Save a checkpoint
    :param checkpoint:
    :param run_name:
    :param directory:
    :return:
    """
    path = os.path.join(directory, f"{run_name}.pt")
    torch.save(checkpoint, path)


def load_checkpoint(file_path, device_str):
    """
    Load a pt file
    :param file_path:
    :param device_str:
    :return:
    """
    return torch.load(file_path, map_location=device_str)


def load_onnx(onnx_path):
    """
    Load an ONNX model
    :param onnx_path:
    :return:
    """
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return model


def init_session_onnx(model):
    """
    Initialise an inference session using ONNX model
    :param model:
    :return:
    """
    return onnxruntime.InferenceSession(model.SerializeToString(), None)
