# python3 analyze_sleep.py --side=right --start_time="2025-02-07 04:00:00" --end_time="2025-02-07 15:00:00"
# cd /home/dac/free-sleep/biometrics/sleep_detection && python3 analyze_sleep.py --side=left --start_time="2025-02-13 03:00:00" --end_time="2025-02-13 15:00:00"
# TODO: Support users manually starting/ending sleep time

import sys
import platform

if platform.system().lower() == 'linux':
    sys.path.append('/home/dac/python_packages/')
    sys.path.append('/home/dac/free-sleep/biometrics/')

import json
import gc
import os
import traceback
from argparse import ArgumentParser, Namespace
import numpy as np

sys.path.append(os.getcwd())

from sleep_detector import detect_sleep
from resource_usage import get_memory_usage_unix, get_available_memory_mb
from utils import validate_datetime_utc
from get_logger import get_logger

logger = get_logger()

FOLDER_PATH = '/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-10/'
if platform.system().lower() == 'linux':
    FOLDER_PATH = '/persistent/'


def _parse_args() -> Namespace:
    # Argument parser setup
    parser = ArgumentParser(description="Process presence intervals with UTC datetime.")

    # Named arguments with default values if needed
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        required=True,
        help="Side of the bed to process (left or right)."
    )
    parser.add_argument(
        "--start_time",
        type=validate_datetime_utc,
        required=True,
        help="Start time in UTC format 'YYYY-MM-DD HH:MM:SS'."
    )
    parser.add_argument(
        "--end_time",
        type=validate_datetime_utc,
        required=True,
        help="End time in UTC format 'YYYY-MM-DD HH:MM:SS'."
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate that start_time is before end_time
    if args.start_time >= args.end_time:
        logger.error("Error: --start_time must be earlier than --end_time.")
        sys.exit(1)

    return args


def main():
    # args = Namespace(
    #     side="right",
    #     start_time=datetime.strptime("2025-01-10 08:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc),
    #     end_time=datetime.strptime("2025-01-10 14:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc),
    # )
    # side = args.side
    args = _parse_args()
    # Display parsed datetime objects

    # Example usage
    duration = args.end_time - args.start_time
    logger.debug(f"Total duration: {duration}")
    try:
        detect_sleep(
            args.side,
            args.start_time,
            args.end_time,
            FOLDER_PATH
        )
        logger.debug(f"END Memory Usage: {get_memory_usage_unix():.2f} MB")
        logger.debug(f"Free Memory: {get_available_memory_mb()} MB")

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        gc.collect()



if __name__ == "__main__":
    logger.debug(f"Memory Usage: {get_memory_usage_unix():.2f} MB")
    logger.debug(f"Free Memory: {get_available_memory_mb()} MB")
    if get_available_memory_mb() < 400:
        error = MemoryError('Available memory is too little, exiting...')
        logger.error(error)
        raise error

    main()




