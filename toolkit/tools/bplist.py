import plistlib
import json
from datetime import datetime

from toolkit import tools



def bplist_read_file(file_path):
    """
    Reads a bplist file.

    Args:
        file_path (sttr):

    Returns:
        dict:
    """
    with open(file_path, 'rb') as file:
        return plistlib.load(file)


def bplist_write_bplist_to_json(bplist_file_path, to_file_path=None):
    """
    Reads a bplist file and writes it to a json file.

    Args:
        bplist_file_path (str):
        to_file_path (str):
    """
    def _convert_datetime(o):
        if isinstance(o, datetime):
            return o.__str__()

    if to_file_path is None: to_file_path = bplist_file_path.replace('.plist', '.json')
    data = bplist_read_file(bplist_file_path)
    tools.write_to_file(to_file_path, json.dumps(data, indent=2, default=_convert_datetime))
    return to_file_path

def convert_all_bplist(root_folder):
    file_paths = tools.list_dir_tree(root_folder)
    for file_path in file_paths:
        if file_path.endswith('plist'):
            bplist_write_bplist_to_json(file_path)



