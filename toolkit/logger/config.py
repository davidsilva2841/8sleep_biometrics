import sys
import os
from pathlib import Path
from dotenv import load_dotenv


PROJECT_FOLDER_PATH = str(Path(__file__).parents[2]) + '/'
DB_FOLDER_PATH = PROJECT_FOLDER_PATH + 'db/'
TEMP_FOLDER_PATH = PROJECT_FOLDER_PATH + 'tmp/'


def _load_env_file(file_path):
    if os.path.isfile(file_path):
        print('Loading environment from ' + file_path)
        load_dotenv(dotenv_path=file_path)
    else:
        print("Couldn't find environment file: " + file_path)


def _load_env():
    """
    Loads the local variables to the ENV.
    """
    env_folder = str(Path(__file__).parents[0]) + '/env/'

    _load_env_file(env_folder + 'common.env')
    _load_env_file(env_folder + 'api_keys.env')

    # _load_env_file(env_folder + 'mac.env')
    if sys.platform == 'linux':
        _load_env_file(env_folder + 'linux.env')
    else:
        _load_env_file(env_folder + 'mac.env')


if not 'LOGGER_ENV_LOADED' in os.environ:
    _load_env()
