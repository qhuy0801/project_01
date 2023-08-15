import os


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


def make_segmentation_dir(_path):
    """
    To make a directory to save segmented images
    :param _path:
    :return:
    """
    os.makedirs(_path, exist_ok=True)
    os.makedirs(os.path.join(_path, "background"), exist_ok=True)
    os.makedirs(os.path.join(_path, "roi"), exist_ok=True)


def make_run_dir(_run_name):
    """
    To make the directory to save models weights and output in each run
    :param _run_name:
    :return:
    """
    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs(os.path.join("model", _run_name), exist_ok=True)
    os.makedirs(os.path.join("output", _run_name), exist_ok=True)


def make_dir(_path):
    """
    Make new directory if not exists
    :param _path:
    :return:
    """
    if not os.path.exists(_path):
        os.makedirs(_path, exist_ok=True)
