import os
import yaml
import collections.abc

DEFAULT_YAML = 'default.yaml'


def load_yaml(filename):
    """
    Appends the default yaml config with the requested yaml config
    :param filename: str
    :return: dict
    """
    # Retrieve name of parent yaml
    if os.path.basename(filename) == DEFAULT_YAML:
        return _read_yaml(filename)

    # Load the default file first
    if os.path.isabs(filename):
        default_file = os.path.join(os.path.dirname(filename), DEFAULT_YAML)
    else:
        default_file = os.path.join(os.path.abspath(os.path.dirname(filename)), DEFAULT_YAML)

    # Read both default and requested yaml
    default_file_content = _read_yaml(default_file)
    file_content = _read_yaml(os.path.abspath(filename))

    # Merge the two yaml by recursive update
    return __deep_update(default_file_content, file_content)


def _read_yaml(file):
    """
    Opens a yaml file, and returns the content as dict
    :param file: str
    :return: dict
    """
    yaml_file = open(file)
    content = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()

    return content


def __deep_update(a, b):
    """
    Updates dict a with values from dict b, recursively
    :param a: dict
    :param b: dict
    :return: dict
    """
    for k, v in b.items():
        if isinstance(v, collections.abc.Mapping):
            a[k] = __deep_update(a.get(k, {}), v)
        else:
            a[k] = v

    return a
