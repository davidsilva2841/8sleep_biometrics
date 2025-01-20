import logging
import sys
from datetime import datetime
import os

from logger import config


class BaseLogger(logging.Logger):
    date: str
    start_time: str

    def __init__(self, name):
        super().__init__(name)

    def _handle_exception(self, exc_type, exc_value, exc_traceback):
        self.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

    def runtime(self):
        return str(datetime.now() - datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S%f'))


def _get_logger_instance():
    return logging.getLogger('lab2')


def _handle_exception(exc_type, exc_value, exc_traceback):
    log = _get_logger_instance()
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error(f"\nUncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def _get_folder_path(date):
    log_folder_path = config.TEMP_FOLDER_PATH + 'logs/python/'
    folder_path = log_folder_path + date + "/"
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path


def _get_file_path(date, start_time):
    return _get_folder_path(date) + start_time + ".log"


def _get_file_handler(file_path):
    handler = logging.FileHandler(file_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '''
%(levelname)s | %(asctime)s | pid: %(process)d | %(pathname)s:%(lineno)s #%(funcName)s
    %(message)s
        '''
    ))
    return handler


def _get_log_level():
    if 'LOG_LEVEL' in os.environ:
        if os.environ['LOG_LEVEL'] == 'INFO':
            return logging.INFO
    return logging.DEBUG


def _get_console_handler():
    handler = logging.StreamHandler()
    handler.setLevel(_get_log_level())
    handler.setFormatter(logging.Formatter(
        """
%(levelname)s | %(asctime)s | pid: %(process)d | %(pathname)s:%(lineno)s #%(funcName)s
%(message)s"""
    ))
    return handler


def _build_logger(logger):
    logger.date = datetime.strftime(datetime.now(), '%Y-%m-%d')
    logger.start_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S%f')
    logger.setLevel(_get_log_level())

    file_path = _get_file_path(logger.date, logger.start_time)
    logger.addHandler(_get_file_handler(file_path))
    logger.addHandler(_get_console_handler())
    logger.debug(f'Log file available @ {file_path}')
    #
    sys.excepthook = _handle_exception


def get_logger():
    """
    Returns:
        BaseLogger: Custom logger
    """
    logging.setLoggerClass(BaseLogger)
    logger = _get_logger_instance()
    if not logger.handlers:
        _build_logger(logger)
    return logger
