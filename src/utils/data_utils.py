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


def init_images(_metadata, _filename_column, _image_dirs):
    """
    Create a list of image instance based on the filtered and sampled data
    :param _metadata:
    :param _filename_column:
    :param _image_dirs:
    :return:
    """

    # Initialise empty array of images
    images = []

    # Creating the array
    for _, row in _metadata.iterrows():
        file_name = row[_filename_column]
        file_path = io_utils.find_path(_image_dirs, file_name)
        image = init_image(file_name, file_path, row)
        if image is not None:
            images.append(image)
        else:
            continue

    return images


def get_metadata(_metadata_path, _number_of_samples, *args):
    """
    Read the metadata (.csv) file, filter the samples based on selected condition
    Return just a selected number of samples
    :param _metadata_path:
    :param _number_of_samples:
    :param args: list of tuples to filter (column_name, condition, value)
    Eligible condition operators:
        Equal: "", " ", "=", None
        Larger than: ">", "larger than"
        Less than: "<", "less than"
    For example: ("ImageDetail.hasRuler", "=", "TRUE") will return all images with ruler
    :return:
    """

    # Read the metadata from file
    metadata = pd.read_csv(_metadata_path)

    # Filter
    for arg in args:
        if arg[1] in ["", " ", "=", None]:
            metadata = metadata[metadata[arg[0]] == arg[2]]
        elif arg[1] in [">", "larger than"]:
            metadata = metadata[metadata[arg[0]] > arg[2]]
        elif arg[1] in ["<", "less than"]:
            metadata = metadata[metadata[arg[0]] > arg[2]]
        else:
            warnings.warn(f"Condition not valid: {arg[1]}\n")

    # Take only a selected number of samples and return
    if (
        _number_of_samples <= 0
        or _number_of_samples >= metadata.shape[0]
    ):
        sample_count = metadata.shape[0]
    else:
        sample_count = _number_of_samples
    return metadata.sample(n=sample_count)


def save_images():
    # TODO
    return None
