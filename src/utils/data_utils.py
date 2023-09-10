import warnings

import pandas as pd


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
