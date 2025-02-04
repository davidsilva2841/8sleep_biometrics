# python3 analyze_sleep.py --side=right --start_time="2025-02-03 05:00:00" --end_time="2025-02-03 15:00:00"
import sys
sys.path.append('/home/dac/python_packages/')
import json
import gc
import os
import argparse
import traceback
import numpy as np

sys.path.append(os.getcwd())
from load_raw_file import load_raw_files
from piezo_data import load_piezo_df, detect_presence_piezo
from cap_data import *
from sleep_detector import *
from resource_usage import *
from logger import get_logger
from utils import *
from db import *

logger = get_logger()


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process presence intervals with UTC datetime.")

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

    # Display parsed datetime objects
    logger.debug(f"Processing side: {args.side}")
    logger.debug(f"Start time (UTC): {args.start_time} ({type(args.start_time)})")
    logger.debug(f"End time (UTC): {args.end_time} ({type(args.end_time)})")

    # Example usage
    duration = args.end_time - args.start_time
    logger.debug(f"Total duration: {duration}")
    try:
        create_db_and_table()
        data = load_raw_files('/persistent/', args.start_time, args.end_time, args.side)

        piezo_df = load_piezo_df(data, args.side)
        detect_presence_piezo(
            piezo_df,
            args.side,
            rolling_seconds=10,
            threshold_percent=0.70,
            range_threshold=100_000,
            range_rolling_seconds=10,
            clean=True
        )
        cap_df = load_cap_df(data, args.side)

        # Cleanup data
        del data
        gc.collect()


        merged_df = piezo_df.merge(cap_df, on='ts', how='inner')
        merged_df.drop_duplicates(inplace=True)

        # Free up memory from old dfs
        piezo_df.drop(piezo_df.index, inplace=True)
        cap_df.drop(cap_df.index, inplace=True)
        del piezo_df
        del cap_df
        gc.collect()

        cap_baseline = load_baseline(args.side)
        detect_presence_cap(
            merged_df,
            cap_baseline,
            args.side,
            occupancy_threshold=5,
            rolling_seconds=10,
            threshold_percent=0.90,
            clean=True
        )

        merged_df[f'final_{args.side}_occupied'] = merged_df[f'piezo_{args.side}1_presence'] + merged_df[f'cap_{args.side}_occupied']
        sleep_records = build_sleep_records(merged_df, args.side, max_gap_in_minutes=15)
        insert_sleep_records(sleep_records)
        print(json.dumps(sleep_records, default=custom_serializer, indent=4))

        print(merged_df.head())
        logger.debug(f"Memory Usage: {get_memory_usage_unix():.2f} MB")
        logger.debug(f"Free Memory: {get_free_memory_mb()} MB")

        # Cleanup
        merged_df.drop(merged_df.index, inplace=True)
        del merged_df
        gc.collect()
    except Exception as e:
        logger.error(e)
        traceback.print_exc()



if __name__ == "__main__":
    if get_free_memory_mb() < 500:
        error = MemoryError('Available memory is too little, exiting...')
        logger.error(error)
        raise error
    # logger.debug(f"Memory Usage: {get_memory_usage_unix():.2f} MB")
    # logger.debug(f"Free Memory: {get_free_memory_mb()} MB")

    main()
    # logger.debug(f"Memory Usage: {get_memory_usage_unix():.2f} MB")
    # logger.debug(f"Free Memory: {get_free_memory_mb()} MB")




