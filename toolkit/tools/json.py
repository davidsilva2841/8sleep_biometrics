import json

from toolkit.tools import files, arrays
from toolkit.logger import get_logger


log = get_logger()


def read_json_from_file(file_path):
    """
    Reads JSON file.

    Args:
        file_path:
    Returns:
        dict: Data from file
    """
    return json.loads(files.read_file(file_path))


def write_json_to_file(file_path, data):
    """
    Write json data to a file.

    Args:
        file_path (str):
        data (dict):
    Returns:

    """
    files.write_to_file(file_path, json.dumps(data, indent=2))


def format_json_file(file_path):
    """
    Formats a json file with 4 space indents.

    Args:
        file_path (str):

    Returns:

    """
    if file_path.endswith('.json'):
        data = read_json_from_file(file_path)
        write_json_to_file(file_path, data)


def append_to_json_array(file_path, key, item):
    """
    Appends an item to an array in a JSON file.

    Args:
        file_path:
        key:
        item:

    Returns:

    """
    js = { }
    try:
        js = read_json_from_file(file_path)

        js[key].append(item)
        write_json_to_file(file_path, js)
    except KeyError:
        log.error(f'Key error | File Path: {file_path} | Available Keys: {js.keys()}')


def remove_items_from_json_array(file_path, key, items):
    """

    Args:
        file_path (str):
        key (str):
        items (list):
    """
    array = read_json_from_file(file_path)[key]
    new_array = arrays.subtract_arrays(array, items)
    set_json_array(file_path, key, new_array)


def reset_json_array(file_path, key):
    """
    Resets an array in a JSON file.

    Args:
        file_path:
        key:

    Returns:

    """
    js = read_json_from_file(file_path)
    if not type(js[key]) == list:
        raise Exception(f'Select key is not an array')
    js[key] = []
    write_json_to_file(file_path, js)


def set_json_array(file_path, key, array):
    """
    Sets an array in a JSON file

    Args:
        file_path (str):
        key (str):
        array (list):
    """
    js = read_json_from_file(file_path)

    js[key] = array
    write_json_to_file(file_path, js)
