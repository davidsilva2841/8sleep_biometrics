import os
import zipfile
import shutil

from toolkit.tools import files
from toolkit.logger import get_logger


log = get_logger()


def zip_file(file_path):
    """

    Args:
        file_path (str):

    Returns:
         (str): File path of zip file
    """
    if os.path.isdir(file_path):
        log.warning(f'File path is a folder: {file_path}')
        return zip_folder(file_path)

    log.debug(f'Zipping file: {file_path}')

    out_file_name = os.path.basename(file_path).split('.')[0]
    out_folder_path = os.path.dirname(file_path) + '/'
    out_path = out_folder_path + out_file_name
    out_file_path = out_path + '.zip'

    files.delete_file(out_file_path)
    shutil.make_archive(out_path, 'zip', file_path)
    return out_file_path


def zip_folder(folder_path):
    """

    Args:
        folder_path (str):

    Returns:
         (str): File path of zip file
    """
    if os.path.isfile(folder_path):
        log.warning(f'Folder path is a file: {folder_path}')
        return zip_file(folder_path)

    log.debug(f'Zipping folder: {folder_path}')

    if folder_path[-1] == '/': folder_path = folder_path[:-1]

    out_file_name = os.path.basename(folder_path)
    out_folder_path = os.path.dirname(folder_path) + '/'
    out_path = out_folder_path + out_file_name
    out_file_path = out_path + '.zip'

    files.delete_file(out_file_path)
    shutil.make_archive(out_path, 'zip', folder_path)
    return out_file_path


def unzip_file(file_path, folder_path=None, is_directory=False):
    log.debug(f'Unzipping file: {file_path}')
    if not os.path.isfile(file_path):
        raise Exception(f'Invalid file path: {file_path}')

    if folder_path is None:
        folder_path = os.path.dirname(file_path) + '/'
        files.create_folder(folder_path)
        if is_directory:
            folder_path += os.path.basename(file_path).split('.')[0] + '/'
            files.create_folder(folder_path)

    log.debug(f'Unzipping: {file_path} | Destination: {folder_path}')
    zip_obj = zipfile.ZipFile(file_path, 'r')
    zip_obj.extractall(folder_path)

    return folder_path
