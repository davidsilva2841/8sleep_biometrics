import pandas as pd
import numpy as np

from plot_presence import plot_cap_presence, plot_df_column, plot_occupancy
from load_raw import load_raw_data
from data_types import Data


def load_cap_df(data: Data) -> pd.DataFrame:
    df = pd.DataFrame(data['cap_senses'])
    df['left_out'] = df['left'].apply(lambda x: x['out'])
    df['left_cen'] = df['left'].apply(lambda x: x['cen'])
    df['left_in'] = df['left'].apply(lambda x: x['in'])
    df['left_status'] = df['left'].apply(lambda x: x['status'])

    df['right_out'] = df['right'].apply(lambda x: x['out'])
    df['right_cen'] = df['right'].apply(lambda x: x['cen'])
    df['right_in'] = df['right'].apply(lambda x: x['in'])
    df['right_status'] = df['right'].apply(lambda x: x['status'])

    df.drop(columns=['left', 'right'], inplace=True)
    df.sort_values(by='ts', inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    return df


def _identify_baseline_period(piezo_df: pd.DataFrame):
    # Apply rolling window and check condition for both columns
    # Define a rolling window of 30 minutes and check condition for both columns
    valid_mask = piezo_df[['left1_range', 'right1_range']].rolling('10min').apply(
        lambda x: (x < 1000).all().all(), raw=True
    )

    # Find the first valid period
    first_valid_index = valid_mask[valid_mask.all(axis=1) == 1].index.min()

    if pd.notna(first_valid_index):
        start_time = first_valid_index
        end_time = start_time + pd.Timedelta(minutes=30)
        print(f"First valid 30-minute period: {start_time} to {end_time}")
        return (start_time, end_time)
    else:
        print("No valid 30-minute period found.")


def create_cap_baseline_from_cap_df(cap_df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp):
    # filtered_df = df[pd.to_datetime(start_time):pd.to_datetime(end_time)]
    filtered_df = cap_df[start_time:end_time]
    baseline_stats = {}
    for sensor in ["left_out", "left_cen", "left_in", "right_out", "right_cen", "right_in"]:
        baseline_stats[sensor] = {
            "mean": filtered_df[sensor].mean(),
            "std": max(filtered_df[sensor].std(), 5)
        }

    return baseline_stats


def calculate_avg(arr):
    return np.mean(arr)


def load_piezo_df(data: Data) -> pd.DataFrame:
    df = pd.DataFrame(data['piezo_dual'])
    df.sort_values(by='ts', inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    df['left1_avg'] = df['left1'].apply(calculate_avg)

    df['right1_avg'] = df['right1'].apply(calculate_avg)
    for column in ['left1_avg', 'right1_avg']:
        upper_bound = np.percentile(df[column], 99)  # 99th percentile
        lower_bound = np.percentile(df[column], 1)  # 1st percentile
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df.drop(columns=['left1', 'left2', 'right1', 'right2', 'type', 'freq', 'adc', 'gain', 'seq'], inplace=True)
    return df


def detect_presence_piezo(df: pd.DataFrame, clean=True):
    df['right1_min'] = df['right1_avg'].rolling(window='10s', center=True).min()
    df['right1_max'] = df['right1_avg'].rolling(window='10s', center=True).max()
    df['right1_range'] = abs(df['right1_max'] - df['right1_min'])

    df['left1_min'] = df['left1_avg'].rolling(window='10s', center=True).min()
    df['left1_max'] = df['left1_avg'].rolling(window='10s', center=True).max()
    df['left1_range'] = abs(df['left1_max'] - df['left1_min'])

    df['piezo_right1_presence'] = df['right1_range'].apply(lambda x: 1 if x > 50_000 else 0)
    df['piezo_left1_presence'] = df['left1_range'].apply(lambda x: 1 if x > 50_000 else 0)

    df['piezo_left1_presence'] = (
        df['piezo_left1_presence']
        .rolling('120s', min_periods=1)
        .apply(lambda x: 1 if x.sum() >= 45 else 0, raw=False)
        .astype(int)
    )

    df['piezo_right1_presence'] = (
        df['piezo_right1_presence']
        .rolling('120s', min_periods=1)
        .apply(lambda x: 1 if x.sum() >= 45 else 0, raw=False)
        .astype(int)
    )
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


# A helper function to turn the raw sensor reading into a "delta from baseline"
# You can scale by standard deviations, or by raw difference from the mean, etc.
def sensor_delta(row, sensor, cap_baseline):
    # difference from mean in terms of # of standard deviations
    return (row[sensor] - cap_baseline[sensor]["mean"]) / cap_baseline[sensor]["std"]


def detect_presence_cap(cap_df: pd.DataFrame, cap_baseline, clean=True):

    cap_df["left_combined"] = (
            cap_df.apply(lambda row: sensor_delta(row, "left_out", cap_baseline), axis=1) +
            cap_df.apply(lambda row: sensor_delta(row, "left_cen", cap_baseline), axis=1) +
            cap_df.apply(lambda row: sensor_delta(row, "left_in", cap_baseline), axis=1)
    )

    cap_df["right_combined"] = (
            cap_df.apply(lambda row: sensor_delta(row, "right_out", cap_baseline), axis=1) +
            cap_df.apply(lambda row: sensor_delta(row, "right_cen", cap_baseline), axis=1) +
            cap_df.apply(lambda row: sensor_delta(row, "right_in", cap_baseline), axis=1)
    )

    occupancy_threshold = 5

    cap_df["cap_left_occupied"] = cap_df["left_combined"].apply(lambda x: 1 if x > occupancy_threshold else 0)
    cap_df["cap_right_occupied"] = cap_df["right_combined"].apply(lambda x: 1 if x > occupancy_threshold else 0)


    # Optional: apply a rolling minimum to require that the signal stays
    # above the threshold for at least, say, 10 seconds in a row before flipping "occupied"
    # cap_df["cap_left_occupied"] = (
    #     cap_df["cap_left_occupied"]
    #     .rolling("120s", min_periods=1)
    #     .apply(lambda x: 1 if np.all(x) else 0, raw=False)
    #     .astype(int)
    # )
    cap_df["cap_left_occupied"] = (
        cap_df["cap_left_occupied"]
        .rolling("120s", min_periods=1)
        .apply(lambda x: 1 if x.sum() >= 45 else 0, raw=False)
        .astype(int)
    )

    cap_df["cap_right_occupied"] = (
        cap_df["cap_right_occupied"]
        .rolling("120s", min_periods=1)
        .apply(lambda x: 1 if x.sum() >= 45 else 0, raw=False)
        .astype(int)
    )


def get_presence_intervals(df, side='left'):
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
    if side == 'left':
        occupancy_col = 'final_left_occupied'
    elif side == 'right':
        occupancy_col = 'final_right_occupied'
    else:
        raise ValueError("Side must be either 'left' or 'right'.")

    # Ensure timestamps are sorted
    df = df.sort_index()

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


# ---------------------------------------------------------------------------------------------------
folder_path = '/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-27'
data = load_raw_data(folder_path=folder_path)

piezo_df = load_piezo_df(data)
detect_presence_piezo(piezo_df, clean=False)


cap_df = load_cap_df(data)
start_time, end_time = _identify_baseline_period(piezo_df)
cap_baseline = create_cap_baseline_from_cap_df(cap_df, start_time, end_time)
detect_presence_cap(cap_df, cap_baseline)



merged_df = piezo_df.merge(cap_df, on='ts', how='inner')
merged_df['final_left_occupied'] = merged_df['piezo_left1_presence'] + merged_df['cap_left_occupied']
merged_df['final_right_occupied'] = merged_df['piezo_right1_presence'] + merged_df['cap_right_occupied']

plot_occupancy(merged_df, start_time='02:00', end_time='17:00')

plot_cap_presence(merged_df, title=f'')



# Example usage with provided DataFrame
left_present_intervals, left_not_present_intervals = get_presence_intervals(merged_df, side='left')
right_present_intervals, right_not_present_intervals = get_presence_intervals(merged_df, side='right')

# Print the results
print("Left Side Present Intervals:", left_present_intervals)
print("Left Side Not Present Intervals:", left_not_present_intervals)

print("Right Side Present Intervals:", right_present_intervals)
for r in right_present_intervals:
    print(r)
print("Right Side Not Present Intervals:", right_not_present_intervals)

merged_df.head()

plot_cap_presence(merged_df, start_time="2025-01-24 04:00:00", end_time="2025-01-26 15:00:00", title=f'')

plot_df_column(merged_df, ['cap_left_occupied', 'piezo_left1_presence', 'left1_avg', 'left1_range', 'left_combined'])



# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------

# plot_df_column(merged_df, ['cap_left_occupied', 'piezo_left1_presence', 'left1_avg', 'left1_range'])
# plot_df_column(piezo_df, ['piezo_left1_presence'], start_time="2025-01-26 08:00:00", end_time="2025-01-26 10:00:00")
#
# plot_df_column(piezo_df, ['left1_avg', 'left1_std', 'right1_avg', 'right1_std'], start_time="2025-01-26 03:00:00", end_time="2025-01-26 10:00:00")
# plot_df_column(piezo_df, ['left1_avg', 'right1_avg'], start_time="2025-01-26 04:00:00", end_time="2025-01-26 07:30:00")
# plot_df_column(piezo_df, ['left1_avg'], start_time="2025-01-26 03:06:00", end_time="2025-01-26 03:07:00")
# plot_df_column(piezo_df, ['left1_avg', 'right1_avg'], start_time="2025-01-26 03:00:00", end_time="2025-01-26 10:00:00")
# plot_df_column(piezo_df, ['right1_avg'], start_time="2025-01-26 06:00:00", end_time="2025-01-26 16:00:00")
# piezo_df[pd.to_datetime("2025-01-26 00:00:00"):pd.to_datetime("2025-01-26 00:30:00")]['left1_range'].max()
#
#
# empty_min = piezo_df[pd.to_datetime("2025-01-26 03:05:00"):pd.to_datetime("2025-01-26 03:10:00")]['left1_avg'].min()
# empty_max = piezo_df[pd.to_datetime("2025-01-26 03:05:00"):pd.to_datetime("2025-01-26 03:10:00")]['left1_avg'].max()
# # 85,556
# empty_range = empty_max - empty_min
#
# tally_min = piezo_df[pd.to_datetime("2025-01-26 05:00:00"):pd.to_datetime("2025-01-26 05:10:00")]['left1_avg'].min()
# tally_max = piezo_df[pd.to_datetime("2025-01-26 05:00:00"):pd.to_datetime("2025-01-26 05:10:00")]['left1_avg'].max()
#
# # 1,677,570
# tally_range = tally_max - tally_min
# tally_range
#
# tally_min = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['left1_avg'].min()
# tally_max = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['left1_avg'].max()
#
# # 295,224
# tally_range = tally_max - tally_min
# tally_range
#
# piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['right1_range']
# david_min_gone = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['right1_avg'].min()
# david_max_gone = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['right1_avg'].max()
# # 79,303
# david_range_gone = david_max_gone - david_min_gone
#
# david_min_present = piezo_df[pd.to_datetime("2025-01-26 07:00:00"):pd.to_datetime("2025-01-26 07:10:00")]['right1_avg'].min()
# david_max_present = piezo_df[pd.to_datetime("2025-01-26 07:00:00"):pd.to_datetime("2025-01-26 07:10:00")]['right1_avg'].max()
# # 1,900,295
# david_range_present = david_max_present - david_min_present
