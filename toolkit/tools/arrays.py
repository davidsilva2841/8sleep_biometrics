from .files import read_file, write_to_file


def read_array_from_file(file_path, row_separator='\n'):
    """
    Read array from a file.

    Args:
        file_path (str):
        row_separator (str):

    Returns:
        list (list):
    """
    array = read_file(file_path).split(row_separator)
    if array[-1] == '':
        return array[:-1]

    return array


def write_array_to_file(file_path, array, row_separator='\n'):
    """
    Writes array to a file.

    Args:
        file_path (str):
        array (list):
        row_separator (str):
    """
    write_to_file(file_path, row_separator.join(array))


def subtract_arrays(array, subtract):
    """
    Subtract an array from another array.

    Args:
        array (list):
        subtract (list):

    Returns:
        list: List
    """
    return [item for item in array if item not in subtract]


def get_unique_items_from_array(array):
    """
    Removes duplicates from array.

    Args:
        array (list):

    Returns:
        list (list):
    """
    return list(set(array))
