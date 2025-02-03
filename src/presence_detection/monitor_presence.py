# region IMPORTS/TYPES
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import TypedDict, Union, Literal
import math

from toolkit import tools
from data_manager import DataManager
from plot_presence import plot_cap_presence, plot_df_column, plot_occupancy
from src.load_raw import load_raw_files
from data_types import Data


class Baseline(TypedDict):
    mean: float
    std: float

# Capacitance baseline
class CapBaseline(TypedDict):
    left_out: Baseline
    left_cen: Baseline
    left_in: Baseline
    right_out: Baseline
    right_cen: Baseline
    right_in: Baseline

# endregion
# ---------------------------------------------------------------------------------------------------
# region HELPERS


def _save_baseline(cap_baseline: CapBaseline):
    tools.write_json_to_file('./cap_baseline.json', cap_baseline)


def _load_settings():
    with open('/home/dac/free-sleep-database/settings.json', 'r') as settings_file:
        return json.load(settings_file)


def _load_cap_df(data: Data) -> pd.DataFrame:
    df = pd.DataFrame(data['cap_senses'])

    # Vectorized extraction:
    df['left_out']   = df['left'].str['out']
    df['left_cen']   = df['left'].str['cen']
    df['left_in']    = df['left'].str['in']
    df['left_status']  = df['left'].str['status']

    df['right_out']  = df['right'].str['out']
    df['right_cen']  = df['right'].str['cen']
    df['right_in']   = df['right'].str['in']
    df['right_status'] = df['right'].str['status']

    df.drop(columns=['left', 'right'], inplace=True)

    # Sort, parse, set index in one pass
    df.sort_values('ts', inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    return df

def _identify_baseline_period(merged_df: pd.DataFrame, threshold_range: int = 10_000, empty_minutes: int = 10):
    range_columns = ['left1_range', 'right1_range']
    stability_columns = ['left_out', 'left_cen', 'left_in', 'right_out', 'right_cen', 'right_in']

    # Convert index to datetime (if not already)
    merged_df = merged_df.sort_index()

    # Iterate over time chunks (efficient early exit)
    window_size = pd.Timedelta(f'{empty_minutes}min')

    for start_time in merged_df.index:
        end_time = start_time + window_size
        window_df = merged_df.loc[start_time:end_time]

        if len(window_df) == 0:
            continue  # Skip if no data

        # Condition 1: Max range values must be < threshold_range
        if window_df[range_columns].max().max() >= threshold_range:
            continue

        # Condition 2: Std must be ≤ 5% of mean for stability columns
        rolling_std = window_df[stability_columns].std()
        rolling_mean = window_df[stability_columns].mean()

        if ((rolling_std / rolling_mean) > 0.05).any():
            continue  # If any column exceeds the threshold, skip

        # If both conditions are met, return the first valid interval
        print(f"First valid interval: {start_time} to {end_time}")
        return start_time, end_time


def _create_cap_baseline_from_cap_df(merged_df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp, min_std: int = 5) -> CapBaseline:
    print(f'Creating baseline for capacitance sensors...')
    filtered_df = merged_df[start_time:end_time]
    baseline_stats = {}
    for sensor in ["left_out", "left_cen", "left_in", "right_out", "right_cen", "right_in"]:
        baseline_stats[sensor] = {
            "mean": filtered_df[sensor].mean(),
            "std": max(filtered_df[sensor].std(), min_std)
        }

    print(json.dumps(baseline_stats, indent=4))
    return baseline_stats


def _calculate_avg(arr: np.ndarray):
    return np.mean(arr)


def _load_piezo_df(data: Data) -> pd.DataFrame:
    df = pd.DataFrame(data['piezo_dual'])
    df.sort_values(by='ts', inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    df['left1_avg'] = df['left1'].apply(_calculate_avg)

    df['right1_avg'] = df['right1'].apply(_calculate_avg)
    for column in ['left1_avg', 'right1_avg']:
        upper_bound = np.percentile(df[column], 98)  # 99th percentile
        lower_bound = np.percentile(df[column], 2)  # 1st percentile
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df.drop(columns=['left1', 'left2', 'right1', 'right2', 'type', 'freq', 'adc', 'gain', 'seq'], inplace=True)
    return df


def _detect_presence_piezo(df: pd.DataFrame, rolling_seconds=180, threshold_percent=0.75, clean=True):
    """Detects presence on a bed using piezo sensor data.

     The function determines when a person is present based on the sensor's range values.
     A rolling window approach is applied to check if the range exceeds a given threshold
     for a specified duration. Presence is marked when the threshold is met.

     Args:
         df (pd.DataFrame):
             The input DataFrame with a DatetimeIndex and columns `right1_avg` and `left1_avg` representing piezo sensor readings.
         rolling_seconds (int, optional):
             The duration (in seconds) for which presence is checked using a rolling sum. Defaults to 180.
         threshold_percent (float, optional):
             The percentage of time within `rolling_seconds` that the sensor range must exceed 10,000 to be considered present. Defaults to 0.75.
         clean (bool, optional):
             If True, drops intermediate computation columns from the DataFrame. Defaults to True.
     Returns:
         pd.DataFrame:
             The DataFrame with added `piezo_right1_presence` and `piezo_left1_presence` columns
             indicating presence (1) or absence (0) based on the rolling threshold.
     """
    range_rolling_seconds = 10

    # Ensure the DataFrame has a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for time-based rolling windows.")

    # Compute min/max range
    df['right1_min'] = df['right1_avg'].rolling(window=range_rolling_seconds, center=True).min()
    df['right1_max'] = df['right1_avg'].rolling(window=range_rolling_seconds, center=True).max()
    df['right1_range'] = df['right1_max'] - df['right1_min']

    df['left1_min'] = df['left1_avg'].rolling(window=range_rolling_seconds, center=True).min()
    df['left1_max'] = df['left1_avg'].rolling(window=range_rolling_seconds, center=True).max()
    df['left1_range'] = df['left1_max'] - df['left1_min']

    # Apply presence detection
    df['piezo_right1_presence'] = (df['right1_range'] >= 10_000).astype(int)
    df['piezo_left1_presence'] = (df['left1_range'] >= 10_000).astype(int)

    threshold_count = math.ceil(threshold_percent * rolling_seconds)

    # Use rolling().sum() instead of apply()
    df['piezo_right1_presence'] = (df['piezo_right1_presence']
                                   .rolling(rolling_seconds, min_periods=1)
                                   .sum()
                                   >= threshold_count).astype(int)

    df['piezo_left1_presence'] = (df['piezo_left1_presence']
                                  .rolling(rolling_seconds, min_periods=1)
                                  .sum()
                                  >= threshold_count).astype(int)
    if clean:
        df.drop(
            columns=[
                'left1_avg',
                'left1_min',
                'left1_max',
                'left1_range',
                'right1_avg',
                'right1_min',
                'right1_max',
                'right1_range'
            ],
            inplace=True
        )


def _sensor_delta(row, sensor, cap_baseline):
    # difference from mean in terms of # of standard deviations
    return (row[sensor] - cap_baseline[sensor]["mean"]) / cap_baseline[sensor]["std"]



def _detect_presence_cap(merged_df: pd.DataFrame,
                        cap_baseline,
                        occupancy_threshold: int = 50,
                        rolling_seconds=120,
                        threshold_percent=0.75,
                        clean=True) -> pd.DataFrame:

    # Vectorized sensor deltas (removes the need for _sensor_delta row-wise function):
    merged_df["left_combined"] = (
            (merged_df["left_out"] - cap_baseline["left_out"]["mean"]) / cap_baseline["left_out"]["std"]
            + (merged_df["left_cen"] - cap_baseline["left_cen"]["mean"]) / cap_baseline["left_cen"]["std"]
            + (merged_df["left_in"]  - cap_baseline["left_in"]["mean"])  / cap_baseline["left_in"]["std"]
    )
    merged_df["right_combined"] = (
            (merged_df["right_out"] - cap_baseline["right_out"]["mean"]) / cap_baseline["right_out"]["std"]
            + (merged_df["right_cen"] - cap_baseline["right_cen"]["mean"]) / cap_baseline["right_cen"]["std"]
            + (merged_df["right_in"]  - cap_baseline["right_in"]["mean"])  / cap_baseline["right_in"]["std"]
    )

    merged_df["cap_left_occupied"] = (merged_df["left_combined"] > occupancy_threshold).astype(int)
    merged_df["cap_right_occupied"] = (merged_df["right_combined"] > occupancy_threshold).astype(int)

    threshold_count = math.ceil(threshold_percent * rolling_seconds)

    # Rolling presence detection
    merged_df["cap_left_occupied"] = (
            merged_df["cap_left_occupied"]
            .rolling(rolling_seconds, min_periods=1)
            .sum()
            >= threshold_count
    ).astype(int)

    merged_df["cap_right_occupied"] = (
            merged_df["cap_right_occupied"]
            .rolling(rolling_seconds, min_periods=1)
            .sum()
            >= threshold_count
    ).astype(int)

    if clean:
        merged_df.drop(columns=["left_combined", "right_combined"], inplace=True)

    return merged_df


def _get_presence_intervals(df, side: Literal['left', 'right']):
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


def _total_duration(intervals):
    """
    Given an array of (start_time, end_time) tuples, calculate the total duration.

    Args:
        intervals (list of tuples): List of (start_time, end_time) tuples.

    Returns:
        timedelta: The total duration.
    """
    total_time = sum((end - start for start, end in intervals), timedelta())
    return total_time


def _identify_sleep_intervals(present_intervals, max_gap_in_minutes: int = 15):
    """
    Identifies sleep periods by merging intervals with small gaps.

    Args:
        present_intervals (list of tuples): List of (start_time, end_time) tuples representing presence periods.
        max_gap_in_minutes (int, optional): Maximum allowed minutes between intervals to merge them. Defaults to 15.

    Returns:
        list of dicts: A list of detected sleep periods, each containing:
            - 'entered_bed_at': Start time of the sleep period.
            - 'left_bed_at': End time of the sleep period.
            - 'sleep_period': Total sleep duration.
            - 'times_exited_bed': Number of times the person exited the bed.
    """
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
                    'sleep_period': total_sleep_time,
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
            'sleep_period': total_sleep_time,
            'times_exited_bed': exit_count,
        })

    return sleep_intervals


def _build_sleep_analysis(merged_df: pd.DataFrame, max_gap_in_minutes: int = 15):
    # Example usage with provided DataFrame
    left_present_intervals, left_not_present_intervals = _get_presence_intervals(merged_df, side='left')
    right_present_intervals, right_not_present_intervals = _get_presence_intervals(merged_df, side='right')

    # Print the results
    print("\nLeft Side Present Intervals:", *left_present_intervals, sep='\n  ')
    print("\nLeft Side Not Present Intervals:", *left_not_present_intervals, sep='\n  ')

    print("\nRight Side Present Intervals:", *right_present_intervals, sep='\n  ')
    print("\nRight Side Not Present Intervals:", *right_not_present_intervals, sep='\n  ')

    sleep_analysis = {
        'left': {
            'present_intervals': left_present_intervals,
            'slept': _identify_sleep_intervals(left_present_intervals, max_gap_in_minutes=max_gap_in_minutes),
            'not_present_intervals': left_not_present_intervals,
            'total_time_in_bed': _total_duration(left_present_intervals),
        },
        'right': {
            'present_intervals': right_present_intervals,
            'slept': _identify_sleep_intervals(right_present_intervals, max_gap_in_minutes=max_gap_in_minutes),
            'not_present_intervals': right_not_present_intervals,
            'total_time_in_bed': _total_duration(right_present_intervals),
        }
    }
    return sleep_analysis


# endregion
# ---------------------------------------------------------------------------------------------------
# region MAIN



folder_path = '/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-20'
data = load_raw_files(folder_path=folder_path)

piezo_df = _load_piezo_df(data)
_detect_presence_piezo(piezo_df, clean=False, rolling_seconds=60, threshold_percent=0.75)


cap_df = _load_cap_df(data)
merged_df = piezo_df.merge(cap_df, on='ts', how='inner')

start_time, end_time = _identify_baseline_period(merged_df, threshold_range=10_000, empty_minutes=20)

cap_baseline = _create_cap_baseline_from_cap_df(merged_df, start_time, end_time)
_save_baseline(cap_baseline)
_detect_presence_cap(merged_df, cap_baseline, occupancy_threshold=5, rolling_seconds=60, threshold_percent=0.75)


merged_df['final_left_occupied'] = merged_df['piezo_left1_presence'] + merged_df['cap_left_occupied']
merged_df['final_right_occupied'] = merged_df['piezo_right1_presence'] + merged_df['cap_right_occupied']

plot_occupancy(merged_df, start_time='02:00', end_time='17:00')
# plot_occupancy(merged_df)

sleep_analysis = _build_sleep_analysis(merged_df, max_gap_in_minutes=15)


# total = identify_sleep_intervals(sleep_analysis['right']['present_intervals'])


# TODO: Get total time with no movement per side

# endregion
# ---------------------------------------------------------------------------------------------------
# region SCRATCH PLOTTING DEBUGGING

# DEBUG
plot_df_column(merged_df, ['final_left_occupied', 'cap_left_occupied', 'piezo_left1_presence', 'left1_avg', 'left1_range', 'left_out', 'left_cen', 'left_in'], start_time='11:10', end_time='11:25')
plot_df_column(merged_df, ['final_right_occupied', 'cap_right_occupied', 'piezo_right1_presence', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='12:40', end_time='13:00')



merged_df["left_out_shift"] = merged_df.apply(lambda row: _sensor_delta(row, "left_out", cap_baseline), axis=1)
merged_df["left_cen_shift"] = merged_df.apply(lambda row: _sensor_delta(row, "left_cen", cap_baseline), axis=1)
merged_df["left_in_shift"] = merged_df.apply(lambda row: _sensor_delta(row, "left_in", cap_baseline), axis=1)


plot_cap_presence(merged_df, start_time="2025-01-24 04:00:00", end_time="2025-01-26 15:00:00", title=f'')

plot_df_column(piezo_df, ['piezo_left1_presence', 'left1_avg', 'left1_range'])
plot_df_column(piezo_df, ['piezo_left1_presence', 'left1_avg', 'left1_range'])
plot_df_column(piezo_df, ['piezo_right1_presence', 'right1_avg', 'right1_range'])
plot_df_column(merged_df, ['cap_left_occupied', 'piezo_left1_presence', 'left1_avg', 'left1_range', 'left_combined'])
plot_df_column(merged_df, ['piezo_left1_presence', 'cap_left_occupied', 'left_out_shift', 'left_cen_shift', 'left_in_shift'])
plot_df_column(merged_df, ['left1_range', 'cap_left_occupied', 'left_out', 'left_cen', 'left_in'])
plot_df_column(merged_df, ['right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='06:00', end_time='15:00')
plot_df_column(merged_df, ['final_left_occupied', 'cap_left_occupied', 'piezo_left1_presence', 'left1_avg', 'left1_range', 'left_out', 'left_cen', 'left_in'], start_time='11:10', end_time='11:25')
plot_df_column(merged_df, ['piezo_right1_presence', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='12:00', end_time='12:55')

plot_df_column(piezo_df, ['left1_range', 'left1_avg'], start_time="2025-01-23 00:00:00", end_time="2025-01-23 02:00:00")
plot_df_column(piezo_df, ['left1_range', 'left1_avg'], start_time="2025-01-23 01:00:00", end_time="2025-01-23 02:00:00")
piezo_df[pd.to_datetime("2025-01-24 12:00:00"):pd.to_datetime("2025-01-24 12:30:00")]['right1_range'].min()
piezo_df[pd.to_datetime("2025-01-24 12:00:00"):pd.to_datetime("2025-01-24 12:05:00")]['right1_range'].mean()


filtered_df = piezo_df.loc[pd.to_datetime("2025-01-24 12:00:00"):pd.to_datetime("2025-01-24 13:00:00")].query("right1_range < 10_000")
filtered_df = merged_df.loc[pd.to_datetime("2025-01-24 12:00:00"):pd.to_datetime("2025-01-24 12:30:00")].query("piezo_right1_presence == 0")


elisa = DataManager('elisa')
np_array = np.concatenate(elisa.piezo_df[pd.to_datetime(start_time):pd.to_datetime(end_time)]['left1'])

plot_df_column(elisa.piezo_df, ['left1'], start_time="2025-01-23 00:00:00", end_time="2025-01-23 02:00:00")
plot_df_column(elisa.piezo_df, ['left1'], start_time="2025-01-23 01:00:00", end_time="2025-01-23 02:00:00")

# endregion
# ---------------------------------------------------------------------------------------------------

df = load_cap_df(data, 'left')
