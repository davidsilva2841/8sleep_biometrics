import os
from pathlib import Path
import shutil
import inspect

from toolkit.logger import get_logger


log = get_logger()


def list_dir(folder_path, full_path=False):
    """
    List all files/folders in a directory

    Args:
        folder_path (str):
        full_path (bool): Return full paths

    Returns:
        list (list):
    """
    file_names = os.listdir(folder_path)
    if full_path:
        return [os.path.join(folder_path, file_name) for file_name in file_names]

    return file_names


def list_dir_files(folder_path, full_path=False):
    """
    List all files in directory.

    Args:
        folder_path (str):
        full_path (bool): Return full file paths

    Returns:
        list (list): Files list
    """

    file_names = os.listdir(folder_path)
    files_list = []

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            files_list.append(file_path if full_path else file_name)

    files_list.sort()
    return files_list


def list_dir_folders(folder_path, full_path=False):
    """
    List all folders from a parent folder.

    Args:
        folder_path (str):
        full_path (bool): Return full folder path.

    Returns:
        list (list): Folders list
    """

    file_names = os.listdir(folder_path)

    folder_paths = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        if os.path.isdir(file_path):
            folder_paths.append(file_path + '/' if full_path else file_name)

    return folder_paths


def list_dir_tree(root_folder, files_only=False):
    """
    Returns every single file path from a root folder.

    Args:
        root_folder (str):

    Returns:
        list (list): List of all file paths from root folder
    """
    if not os.path.isdir(root_folder):
        raise Exception("Folder doesn't exist")

    subfile_paths = []
    for root, folders, files in os.walk(root_folder):
        if not files_only:
            subfile_paths.append(root)
        for file in files:
            file_path = os.path.join(root, file)
            if files_only:
                if os.path.isfile(file_path):
                    subfile_paths.append(file_path)
            else:
                subfile_paths.append(file_path)

    return subfile_paths


def get_file_type(file_path):
    """
    Gets file type for given file path.

    Args:
        file_path (str):

    Returns:
        dict: { file_path: (str), info: (str), type: (str) }
    """

    try:
        import magic

    except Exception as e:
        log.error(f'Exception importing magic library. \nSee how to set up for your OS here. https://pypi.org/project/python-magic/ \nException: {str(e)}')
        return

    return {
        'file_path': file_path,
        'info': magic.from_file(file_path),
        'type': magic.from_file(file_path, True)
    }


def create_file(file_path):
    """
    Creates a file.

    Args:
        file_path (str):
    """
    log.debug(f'Creating file: {file_path}')
    if os.path.isfile(file_path):
        raise FileExistsError('File path already exists')
    os.mknod(file_path)


def write_to_file(file_path, data, overwrite=True):
    """
    Writes data to file.

    Args:
        file_path (str):
        data (str):
        overwrite (bool): If file exists, overwrite file
    """
    log.debug(f'Writing data to file: {file_path}')

    if not overwrite:
        if os.path.isfile(file_path):
            raise FileExistsError('File path already exists')

    with open(file_path, 'w+') as file:
        file.write(data)

    file.close()


def read_file(file_path):
    """
    Reads and returns file content.

    Args:
        file_path (str):

    Returns:
        str (str):
    """
    if not os.path.isfile(file_path):
        raise Exception(f"File path doesn't exist: {file_path}")

    with open(file_path, 'r') as file:
        text = file.read()
        file.close()
    return text


def delete_file(file_path):
    """
    Delete a file.

    Args:
        file_path (str):
    """
    if os.path.isfile(file_path):
        log.debug(f'Deleting file: {file_path}')
        os.remove(file_path)

def copy_file(from_file_path, to_file_path):
    shutil.copy(from_file_path, to_file_path)


def move_file(from_file_path, to_file_path='', to_folder_path=''):
    """
    Moves a file to a new destination.

    Args:
        to_file_path (str):
        from_file_path (str):
        to_folder_path (str):
    """

    if not to_folder_path and not to_file_path:
        raise Exception('Requires either args: to_file_path or to_folder_path')

    if to_folder_path:
        to_file_path = Path.joinpath(Path(to_folder_path), os.path.basename(from_file_path))


    if os.path.isfile(to_file_path):
        raise FileExistsError(f'File path already exists: {to_file_path}')

    if not os.path.isdir(os.path.dirname(to_file_path)):
        create_folder(os.path.dirname(to_file_path))

    shutil.move(from_file_path, to_file_path)


def rename_file(file_path, new_file_name):
    """
    Renames a file.

    Args:
        file_path (str):
        new_file_name (str):
    """
    if not os.path.isfile(file_path):
        raise Exception(f"File doesn't exist: {file_path}")

    head, tail = os.path.split(file_path)

    to_file_path = os.path.join(head, new_file_name)

    if os.path.isfile(to_file_path):
        raise FileExistsError(f'File path already exists: {to_file_path}')

    log.debug(f'Renaming file: {file_path} | New name: {new_file_name}')
    os.rename(file_path, to_file_path)
    return to_file_path


def create_folder(folder_path):
    """
    Make a folder.

    Args:
        folder_path (str):
    """

    if not os.path.isdir(folder_path):
        log.debug(f'Creating folder: {folder_path}')
        os.makedirs(folder_path)
    return folder_path


def delete_folder(folder_path):
    """
    Delete a folder.

    Args:
        folder_path (str):
    """
    log.debug(f'Deleting folder: {folder_path}')
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


def copy_folder(from_folder_path, to_folder_path):
    """
    Args:
        from_folder_path (str):
        to_folder_path (str):
    """
    shutil.copytree(from_folder_path, to_folder_path)


def fix_encoding(file_path, read_as='utf-8', write_as='utf-8', remove_chars=None):
    """
    Ignores data encoding errors and writes to utf-8.

    Args:
        file_path (str):
        read_as (str):
        write_as (str):
        remove_chars (list):
    """
    log.debug(f'Fixing encoding for file: {file_path}')
    with open(file_path, encoding=read_as, errors='replace') as file:
        text = file.read()
        file.close()

    if remove_chars:
        for remove_char in remove_chars:
            text = text.replace(remove_char, '')

    with open(file_path, 'w+', encoding=write_as) as file:
        file.write(text)
        file.close()


def move_all_files_from_root(root_folder, to_folder):
    """
    Moves all files from a folder root to a new folder.

    Args:
        root_folder (str):
        to_folder (str):
    """

    file_paths = list_dir_tree(root_folder)
    skipped_files = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            head, tail = os.path.split(file_path)
            to_file_path = os.path.join(to_folder, tail)
            try:
                move_file(file_path, to_file_path)
            except Exception as e:
                print(e)
                skipped_files.append(file_path)

    return skipped_files



def current_directory():
    """

    Returns:
         str:

    """
    file_path = inspect.stack()[1][1]
    return os.path.dirname(file_path) + '/'
    # return os.path.abspath(os.getcwd()) + '/'


def replace_text_in_file(file_path, old_text, new_text):
    """

    Args:
        file_path (str):
        old_text (str):
        new_text (str):

    Returns:

    """
    text = read_file(file_path)
    text = text.replace(old_text, new_text)
    write_to_file(file_path, text, overwrite=True)


def get_parent_folder_name(file_path=None):
    """

    Returns:
        str:
    """
    #     return os.path.basename(os.getcwd())
    if file_path:
        return os.path.basename(os.path.dirname(file_path))
    else:
        file_path = inspect.stack()[1][1]
        return file_path.split('/')[-3:][0]



