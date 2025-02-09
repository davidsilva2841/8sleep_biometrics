from typing import Literal
import platform
import traceback
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from datetime import datetime
import os



class BaseLogger(logging.Logger):
    date: str
    start_time: str
    folder_path: str
    env: Literal['local', 'prod']

    def __init__(self, name):
        super().__init__(name)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        self.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

    def runtime(self):
        return str(datetime.now() - datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S'))


def _get_logger_instance():
    return logging.getLogger('presence_detection')


def _handle_exception(exc_type, exc_value, exc_traceback):
    logger = _get_logger_instance()
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    traceback.print_exception(exc_type, exc_value, exc_traceback)




def _get_log_level():
    return logging.INFO if os.getenv('LOG_LEVEL') == 'INFO' else logging.DEBUG


class FixedWidthFormatter(logging.Formatter):
    def format(self, record):
        # Format timestamp
        timestamp = self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S')

        # Fixed-width formatting for LEVEL (8 chars) and FILE:LINE (30 chars)
        level = f"{record.levelname:<8}"                      # Left-align to 8 chars
        file_info = f"{record.filename}:{record.lineno}"     # e.g., script.py:45
        file_info_padded = f"{file_info:<40}"                # Left-align to 40 chars

        # Add Process ID (PID), fixed-width of 6 characters
        pid = f"{record.process:<6}"                         # Left-align to 6 chars

        # Combine formatted parts
        formatted_message = f"{timestamp} UTC | PID: {pid} | {level} | {file_info_padded} | {record.getMessage()}"
        return formatted_message


FORMATTER = FixedWidthFormatter()

def _get_file_handler(data_folder_path: str):
    folder_path = f'{data_folder_path}logs/'

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    handler = TimedRotatingFileHandler(
        filename=f"{folder_path}/biometrics.log",    # Log file path
        when="midnight",            # Rotate at midnight
        interval=1,                 # Rotate every day
        backupCount=2               # Keep 7 days of logs
    )
    handler.setFormatter(FORMATTER)
    return handler


def _get_console_handler():
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    # Use the custom FixedWidthFormatter
    handler.setFormatter(FORMATTER)
    return handler


def _build_logger(logger: BaseLogger):
    logger.date = datetime.now().strftime('%Y-%m-%d')
    logger.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if platform.system().lower() == 'linux':
        logger.env = 'prod'
        logger.folder_path = '/persistent/free-sleep-data/'
    else:
        logger.env = 'local'
        logger.folder_path = '/Users/ds/free-sleep/server/free-sleep-data/'

    logger.setLevel(logging.DEBUG)
    logger.addHandler(_get_console_handler())
    # logger.addHandler(_get_file_handler(logger.folder_path))
    sys.excepthook = _handle_exception


def get_logger():
    """
    Returns:
        BaseLogger: Custom logger with fixed-width formatting
    """
    logging.setLoggerClass(BaseLogger)
    logger = _get_logger_instance()
    if not logger.handlers:
        _build_logger(logger)
    return logger


