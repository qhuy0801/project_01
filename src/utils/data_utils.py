import math
import warnings

import re
import cv2
import pandas as pd
from pandas import DataFrame


def num_duplicated_values(df: DataFrame, column_name: str):
    """
    Count the number of duplicated values in a specific column
    :param df:
    :param column_name:
    :return:
    """
    return (df[column_name].value_counts() > 1).sum()


def filter_empty_columns(df: DataFrame, column_name):
    """
    Filter data where a specific column is null of empty
    :param df:
    :param column_name:
    :return:
    """
    return df[df[column_name].notnull() & (df[column_name] != '')]


def save_csv(df: DataFrame, file_path: str):
    """
    Save a dataframe to csv
    :param df:
    :param file_path:
    :return:
    """
    return df.to_csv(file_path, index=False)


def load_image(file_path):
    """
    Load the image using OpenCV
    :param file_path:
    :return: np.ndarray
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)


def remove_image_extension(file_name):
    """
    Remove image extension such
    :param file_name:
    :return:
    """
    return file_name.replace(".jpeg", "").replace(".jpg", "").replace(".png", "")


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


def binary_rounding(n):
    """
    Round a number to the nearest power of 2
    Example: 3 -> 4, 6 -> 8, 14 -> 16
    :param n:
    :return:
    """
    exponent = math.ceil(math.log2(n))
    return 2 ** exponent


def split_and_flatten(x):
    """
    Split a set of string
    {'apple, grape', 'banana; orange'} > {'apple', 'grape', 'banana', 'orange'}
    :param x: tuple
    :return:
    """
    return {item.strip() for s in x for item in re.split(r'[;,]', s)}


def split_string(x):
    """
    Split a string
    :param x:
    :return:
    """
    return [item.strip() for item in re.split(r'[;,]', x)]
