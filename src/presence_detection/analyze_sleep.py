# python3 analyze_sleep.py --side=right --start_time="2025-01-31 23:00:00" --end_time="2025-02-01 15:00:00"
# python3 analyze_sleep.py --side=left --start_time="2025-01-31 23:00:00" --end_time="2025-02-01 15:30:00"
# python3 analyze_sleep.py --side=left --start_time="2025-02-01 14:00:00" --end_time="2025-02-01 14:01:00"
import json
import gc
import pandas as pd
import os
import sys
import argparse
import traceback
from datetime import datetime, timezone, timedelta

sys.path.append(os.getcwd())
from load_raw_file import load_raw_data
from load_piezo_data import load_piezo_df, detect_presence_piezo
from load_cap_data import load_cap_df, load_baseline, detect_presence_cap
from sleep_detector import build_sleep_analysis
from logger import get_logger

logger = get_logger()

def validate_datetime_utc(date_str):
    """
    Validate and parse datetime input as UTC.
    Expects format: YYYY-MM-DD HH:MM:SS
    """
    try:
        # Parse datetime string and attach UTC timezone
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format: '{date_str}'. Use 'YYYY-MM-DD HH:MM:SS'."
        )



def get_memory_usage_unix():
    pid = os.getpid()  # Current process ID
    page_size = os.sysconf('SC_PAGE_SIZE')    # Page size in bytes
    rss = int(open(f'/proc/{pid}/statm').read().split()[1])  # Resident Set Size in pages
    memory_usage_mb = (rss * page_size) / (1024 ** 2)        # Convert to MB
    return memory_usage_mb


def get_free_memory_mb():
    """
    Returns the free memory in MB by reading /proc/meminfo (Linux only).
    """
    meminfo_path = '/proc/meminfo'
    if not os.path.exists(meminfo_path):
        raise EnvironmentError("This function is supported only on Linux systems with /proc/meminfo.")

    with open(meminfo_path, 'r') as meminfo:
        for line in meminfo:
            if line.startswith('MemFree:'):
                # Extract the value in KB
                free_kb = int(line.split()[1])
                # Convert to MB
                free_mb = free_kb / 1024
                return round(free_mb, 2)

    # If MemFree is not found
    return 0.0


def custom_serializer(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()  # Convert to ISO 8601 format
    elif isinstance(obj, (timedelta, pd.Timedelta)):
        return str(obj)  # Convert timedelta to string format
    raise TypeError(f"Type {type(obj)} not serializable")


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

        data = load_raw_data(args.start_time, args.end_time, args.side)

        piezo_df = load_piezo_df(data, args.side)
        detect_presence_piezo(piezo_df, args.side, rolling_seconds=180, threshold_percent=0.75, range_rolling_seconds=10, clean=True)
        cap_df = load_cap_df(data, args.side)

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

        cap_baseline = load_baseline()

        detect_presence_cap(merged_df, cap_baseline, args.side, occupancy_threshold=5, rolling_seconds=60, threshold_percent=0.75)

        merged_df[f'final_{args.side}_occupied'] = merged_df[f'piezo_{args.side}1_presence'] + merged_df[f'cap_{args.side}_occupied']
        sleep_analysis = build_sleep_analysis(merged_df, args.side, max_gap_in_minutes=15)
        print(json.dumps(sleep_analysis, default=custom_serializer, indent=4))
        count = (merged_df[f'final_{args.side}_occupied'] > 0).sum()
        logger.debug(count)



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
    logger.debug(f"Memory Usage: {get_memory_usage_unix():.2f} MB")
    logger.debug(f"Free Memory: {get_free_memory_mb()} MB")

    main()
    logger.debug(f"Memory Usage: {get_memory_usage_unix():.2f} MB")
    logger.debug(f"Free Memory: {get_free_memory_mb()} MB")




# # Check the type of each thermal zone
# cat /sys/class/thermal/thermal_zone0/type
# cat /sys/class/thermal/thermal_zone1/type
# cat /sys/class/thermal/thermal_zone2/type
# cat /sys/class/thermal/thermal_zone3/type
# Check the type of each trip point
# cat /sys/class/thermal/thermal_zone0/trip_point_0_type
# cat /sys/class/thermal/thermal_zone0/trip_point_1_type
#
# # Check the corresponding temperature limits
# cat /sys/class/thermal/thermal_zone0/trip_point_0_temp
# cat /sys/class/thermal/thermal_zone0/trip_point_1_temp
