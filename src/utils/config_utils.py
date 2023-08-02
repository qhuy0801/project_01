import yaml


def read_config(_config_path):
    """
    Read file and return configuration
    :param _config_path:
    :return:
    """
    with open(_config_path, "r") as _config:
        return yaml.safe_load(_config)
