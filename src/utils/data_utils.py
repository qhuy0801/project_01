import warnings

import pandas as pd
import utils.io_utils as io_utils
from entities.Image import Image


def init_image(_file_name, _file_path, _metadata):
    """
    Init Image class instance but not loading the image matrix yet to save the memory
    :param _file_name:
    :param _file_path:
    :param _metadata:
    :return:
    """
    if _file_path is not None:
        image = Image(_file_name, _file_path)
        image.load_metadata(_metadata)
        return image
    else:
        warnings.warn(f"File not found: {_file_name}\n")
        return None


def init_images(_metadata_path, _image_dirs, _number_of_sample, _filename_column):
    """
    Read the metadata (.csv) file, initialise a list of images
    :param _metadata_path:
    :param _image_dirs:
    :param _number_of_sample:
    :param _filename_column:
    :return: list of initialised image instance (class)
    """

    # Initialise empty array of images
    images = []

    # Read metadata and get only selected number of desired sample
    metadata = pd.read_csv(_metadata_path)
    if (
        _number_of_sample <= 0
        or _number_of_sample >= metadata.shape[0]
    ):
        sample_count = metadata.shape[0]
    else:
        sample_count = _number_of_sample
    metadata = metadata.sample(n=sample_count)

    # Creating the array
    for _, row in metadata.iterrows():
        file_name = row[_filename_column]
        file_path = io_utils.find_path(_image_dirs, file_name)
        image = init_image(file_name, file_path, row)
        if image is not None:
            images.append(image)
        else:
            continue

    return images


def save_images():
    # TODO
    return None
