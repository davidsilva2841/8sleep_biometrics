import traceback
import logging
import sys
from datetime import datetime
import os


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
        return str(datetime.now() - datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S'))


def _get_logger_instance():
    return logging.getLogger('presence_detection')


def _handle_exception(exc_type, exc_value, exc_traceback):
    log = _get_logger_instance()
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error(f"\nUncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    print(exc_traceback)
    traceback.print_exc()



def _get_log_level():
    return logging.INFO if os.getenv('LOG_LEVEL') == 'INFO' else logging.DEBUG


class FixedWidthFormatter(logging.Formatter):
    def format(self, record):
        # Format timestamp
        timestamp = self.formatTime(record, datefmt='%Y-%m-%d %H:%M:%S')

        # Fixed-width formatting for LEVEL (8 chars) and FILE:LINE (30 chars)
        level = f"{record.levelname:<8}"                      # Left-align to 8 chars
        file_info = f"{record.filename}:{record.lineno}"     # e.g., script.py:45
        file_info_padded = f"{file_info:<30}"                # Left-align to 30 chars

        # Combine formatted parts
        formatted_message = f"{timestamp} | {level} | {file_info_padded} | {record.getMessage()}"
        return formatted_message


def _get_console_handler():
    handler = logging.StreamHandler()
    handler.setLevel(_get_log_level())

    # Use the custom FixedWidthFormatter
    formatter = FixedWidthFormatter()
    handler.setFormatter(formatter)
    return handler


def _build_logger(logger):
    logger.date = datetime.now().strftime('%Y-%m-%d')
    logger.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_get_console_handler())
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


