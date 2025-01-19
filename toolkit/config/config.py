import sys
import os
from pathlib import Path
from dotenv import load_dotenv


PROJECT_FOLDER_PATH = str(Path(__file__).parents[3]) + '/'
DB_FOLDER_PATH = PROJECT_FOLDER_PATH + 'python/lab/'
TEMP_FOLDER_PATH = PROJECT_FOLDER_PATH + 'python/temp/'


def _load_env():
    """
    Loads the local variables to the ENV.
    """
    if sys.platform == 'linux':
        env_file_path = PROJECT_FOLDER_PATH + 'linux.env'
    else:
        env_file_path = PROJECT_FOLDER_PATH + 'mac.env'

    if os.path.isfile(env_file_path):
        print('Loading environment from ' + env_file_path)
        load_dotenv(dotenv_path=env_file_path)
    else:
        print("Could't find environment file: " + env_file_path)


_load_env()




