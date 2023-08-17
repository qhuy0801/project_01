import warnings

import pandas as pd
import utils.io_utils as io_utils
from entities.Sample import Sample


def init_sample(_file_name, _file_path, _metadata):
    """
    Init Image class instance but not loading the sample matrix yet to save the memory
    :param _file_name:
    :param _file_path:
    :param _metadata:
    :return:
    """
    if _file_path is not None:
        sample = Sample(_file_name, _file_path)
        sample.load_metadata(_metadata)
        return sample
    else:
        warnings.warn(f"File not found: {_file_name}\n")
        return None


def init_samples(_metadata, _filename_column, _image_dirs):
    """
    Create a list of sample instance based on the filtered and sampled data
    :param _metadata:
    :param _filename_column:
    :param _image_dirs:
    :return:
    """

    # Initialise empty array of samples
    samples = []

    # Creating the array
    for _, row in _metadata.iterrows():
        file_name = row[_filename_column]
        file_path = io_utils.find_path(_image_dirs, file_name)
        sample = init_sample(file_name, file_path, row)
        if sample is not None:
            samples.append(sample)
        else:
            continue

    return samples


def get_metadata(_metadata_path, _number_of_samples, *args):
    """
    Read the metadata (.csv) file, filter the samples based on selected condition
    Return just a selected number of samples
    :param _metadata_path:
    :param _number_of_samples:
    :param args: tuples to filter (column_name, condition, value)
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


def save_sample():
    # TODO
    return None


def load_data():
    return None, None
