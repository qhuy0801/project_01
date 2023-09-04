import os

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
