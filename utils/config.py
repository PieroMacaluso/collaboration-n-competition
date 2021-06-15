import yaml


def read_hp(path):
    """
    Return python dict from .yml file.

    Args:
        path (str): path to the .yml config.

    Returns (dict): configuration object.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg
