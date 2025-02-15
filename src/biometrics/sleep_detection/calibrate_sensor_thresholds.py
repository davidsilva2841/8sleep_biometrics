# python3 calibrate_sensor_thresholds.py --side=left --start_time="2025-02-08 18:00:00" --end_time="2025-02-08 18:50:00"
# python3 calibrate_sensor_thresholds.py --side=right --start_time="2025-02-08 18:00:00" --end_time="2025-02-08 18:50:00"
import sys
import platform
import os

sys.path.append(os.getcwd())
if platform.system().lower() == 'linux':
    sys.path.append('/home/dac/python_packages/')
    sys.path.append('/home/dac/free-sleep/biometrics/')

import gc
from argparse import Namespace, ArgumentParser
import traceback

FOLDER_PATH = '/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-10/'
if platform.system().lower() == 'linux':
    FOLDER_PATH = '/persistent/'
    sys.path.append('/home/dac/python_packages/')
    sys.path.append('/home/dac/free-sleep/biometrics/')
from data_types import *
from load_raw_files import load_raw_files
from piezo_data import load_piezo_df, detect_presence_piezo, identify_baseline_period
from cap_data import load_cap_df, create_cap_baseline_from_cap_df, save_baseline
from resource_usage import get_memory_usage_unix, get_available_memory_mb
from get_logger import get_logger

logger = get_logger()


from utils import validate_datetime_utc

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


def calibrate_sensor_thresholds(side: Side, start_time: datetime, end_time: datetime, folder_path: str):
    data = load_raw_files(
        folder_path,
        start_time,
        end_time,
        side,
        sensor_count=1,
        raw_data_types=['capSense', 'piezo-dual']
    )

    piezo_df = load_piezo_df(data, side)
    detect_presence_piezo(
        piezo_df,
        side,
        rolling_seconds=10,
        threshold_percent=0.70,
        range_threshold=80_000,
        range_rolling_seconds=10,
        clean=False
    )

    cap_df = load_cap_df(data, side)
    # Cleanup data
    del data
    gc.collect()

    merged_df = piezo_df.merge(cap_df, on='ts', how='inner')
    # Free up memory from old dfs
    piezo_df.drop(piezo_df.index, inplace=True)
    cap_df.drop(cap_df.index, inplace=True)
    del piezo_df
    del cap_df
    gc.collect()

    # Create baseline
    baseline_start_time, baseline_end_time = identify_baseline_period(merged_df, side, threshold_range=10_000, empty_minutes=5)
    cap_baseline = create_cap_baseline_from_cap_df(merged_df, baseline_start_time, baseline_end_time, side, min_std=5)
    save_baseline(side, cap_baseline)


    # Cleanup
    merged_df.drop(merged_df.index, inplace=True)
    del merged_df
    gc.collect()


if __name__ == "__main__":
    if logger.env == 'prod':
        logger.debug(f"Memory Usage: {get_memory_usage_unix():.2f} MB")
        logger.debug(f"Free Memory: {get_available_memory_mb()} MB")
        if get_available_memory_mb() < 300:
            error = MemoryError('Available memory is too little, exiting...')
            logger.error(error)
            raise error

    args = _parse_args()

    # Display parsed datetime objects
    logger.debug(f"Processing side: {args.side}")
    logger.debug(f"Start time (UTC): {args.start_time} ({type(args.start_time)})")
    logger.debug(f"End time (UTC): {args.end_time} ({type(args.end_time)})")

    duration = args.end_time - args.start_time
    logger.debug(f"Total duration: {duration}")
    try:
        calibrate_sensor_thresholds(args.side, args.start_time, args.end_time, FOLDER_PATH)
    except Exception as e:
        gc.collect()
        logger.error(e)
        traceback.print_exc()



