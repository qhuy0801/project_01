import pandas as pd
import utils.io_utils as io_utils
from entities.Image import Image


def init_images(config):
    """
    Load images based on the configuration:
    - Metadata path
    - Image source directories
    - Image metadata
    - Number of desired sample
    # TODO refactor this code so that it can be more re-usable
    :return:
    """

    images = []

    metadata_path = config["data"]["metadata_path"]
    image_dirs = config["data"]["images_dirs"]

    file_name_col = config["data"]["column_names"]["file_name"]

    metadata = pd.read_csv(metadata_path)
    if (
        config["samples"]["sample_count"] <= 0
        or config["samples"]["sample_count"] >= metadata.shape[0]
    ):
        sample_count = metadata.shape[0]
    else:
        sample_count = config["samples"]["sample_count"]
    metadata = metadata.sample(n=sample_count)

    for _, row in metadata.iterrows():
        file_name = row[file_name_col]
        file_path = io_utils.find_path(image_dirs, file_name)
        if file_path is not None:
            image = Image(file_name, file_path)
            image.load_metadata(row)
            images.append(image)

    return images


def save_images():
    # TODO
    return None
