import pandas as pd
from typing import List, Tuple
from datetime import datetime, timedelta

from logger import get_logger
from presence_types import *

logger = get_logger()


def _get_presence_intervals(df, side: Side):
    """
    Get time intervals when someone was present and not present on the bed.

    Parameters:
        df (pd.DataFrame): The input DataFrame with occupancy data.
        side (str): 'left' or 'right' to check occupancy.

    Returns:
        present_intervals (list): List of tuples (start_time, end_time) when occupied.
        not_present_intervals (list): List of tuples (start_time, end_time) when not occupied.
    """
    # Select the relevant column based on the chosen side
    occupancy_col = f'final_{side}_occupied'

    # Ensure timestamps are sorted

    # Initialize tracking variables
    present_intervals = []
    not_present_intervals = []
    current_status = None
    start_time = df.index[0]

    # Iterate over DataFrame to find state changes
    for timestamp, row in df.iterrows():
        status = row[occupancy_col] == 2  # Presence condition

        if current_status is None:
            current_status = status
            continue

        # Check for status change
        if status != current_status:
            end_time = timestamp

            if current_status:
                present_intervals.append((start_time, end_time))
            else:
                not_present_intervals.append((start_time, end_time))

            # Update for the new interval
            start_time = timestamp
            current_status = status

    # Capture the last interval
    end_time = df.index[-1]
    if current_status:
        present_intervals.append((start_time, end_time))
    else:
        not_present_intervals.append((start_time, end_time))

    return present_intervals, not_present_intervals


def _total_duration_seconds(intervals) -> int:
    """
    Given an array of (start_time, end_time) tuples, calculate the total duration.

    Args:
        intervals (list of tuples): List of (start_time, end_time) tuples.
    """
    total_time = sum((end - start for start, end in intervals), timedelta())
    return int(total_time.total_seconds())


def _identify_sleep_intervals(present_intervals, max_gap_in_minutes: int = 15):
    """
    Identifies sleep periods by merging intervals with small gaps.

    Args:
        present_intervals (list of tuples): List of (start_time, end_time) tuples representing presence periods.
        max_gap_in_minutes (int, optional): Maximum allowed minutes between intervals before merging them. Defaults to 15.

    Returns:
        list of dicts: A list of detected sleep periods, each containing:
            - 'entered_bed_at': Start time of the sleep period.
            - 'left_bed_at': End time of the sleep period.
            - 'sleep_period': Total sleep duration.
            - 'times_exited_bed': Number of times the person exited the bed.
    """
    logger.debug(f'Identifying sleep intervals... | max_gap_in_minutes: {max_gap_in_minutes}')
    max_gap = timedelta(minutes=max_gap_in_minutes)
    if not present_intervals:
        return []

    sleep_intervals = []
    current_start, current_end = present_intervals[0]  # Start with the first interval
    total_sleep_time = current_end - current_start
    exit_count = 0

    for ix in range(1, len(present_intervals)):
        next_start, next_end = present_intervals[ix]  # Get next interval
        gap = next_start - current_end  # Calculate gap between intervals

        if gap <= max_gap:
            # Merge into the current sleep period
            current_end = next_end
            total_sleep_time += (next_end - next_start)
            exit_count += 1
        else:
            # Only add sleep interval if it's greater than 3 hours
            if total_sleep_time > timedelta(hours=3):
                sleep_intervals.append({
                    'entered_bed_at': current_start,
                    'left_bed_at': current_end,
                    'sleep_period_seconds': int(total_sleep_time.total_seconds()),
                    'times_exited_bed': exit_count,
                })

            # Reset values for the new sleep period
            current_start, current_end = next_start, next_end
            total_sleep_time = current_end - current_start
            exit_count = 0

    # Ensure the last interval is added only if it meets the 3-hour requirement
    if total_sleep_time > timedelta(hours=3):
        sleep_intervals.append({
            'entered_bed_at': current_start,
            'left_bed_at': current_end,
            'sleep_period_seconds': int(total_sleep_time.total_seconds()),
            'times_exited_bed': exit_count,
        })

    return sleep_intervals



def _filter_intervals(
        intervals: List[Tuple[datetime, datetime]],
        start: datetime,
        end: datetime
) -> List[Tuple[datetime, datetime]]:
    """
    Filters intervals to include only those that overlap with the given start and end times.
    """
    filtered_intervals = [
        (max(interval_start, start), min(interval_end, end))
        for interval_start, interval_end in intervals
        if interval_end > start and interval_start < end  # Overlap condition
    ]
    return filtered_intervals


def build_sleep_records(merged_df: pd.DataFrame, side: Side, max_gap_in_minutes: int = 15) ->List[SleepRecord]:
    logger.debug('Building sleep records...')
    present_intervals, not_present_intervals = _get_presence_intervals(merged_df, side)
    sleep_intervals = _identify_sleep_intervals(present_intervals, max_gap_in_minutes=max_gap_in_minutes)

    sleep_records = []
    for sleep_interval in sleep_intervals:
        entered_bed_at = sleep_interval['entered_bed_at']
        left_bed_at = sleep_interval['left_bed_at']

        # Filter intervals specific to the current sleep interval
        filtered_present_intervals = _filter_intervals(present_intervals, entered_bed_at, left_bed_at)
        filtered_not_present_intervals = _filter_intervals(not_present_intervals, entered_bed_at, left_bed_at)

        sleep_records.append({
            "side": side,
            **sleep_interval,
            # 'total_seconds_in_bed': _total_duration_seconds(filtered_present_intervals),
            'present_intervals': filtered_present_intervals,
            'not_present_intervals': filtered_not_present_intervals,
        })

    return sleep_records
