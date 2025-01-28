# Ingest capacitance data
# Maintain running averages
# Identify when presence is detected (by threshold?)

# STD of capacitance during unoccupied period is ~5
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from plot_presence import plot_cap_presence, plot_df_column
from load_raw import load_raw_data
from data_types import Data
from cleaning import interpolate_outliers_in_wave


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


def create_cap_baseline_from_cap_df(df: pd.DataFrame, start_time: str, end_time: str):
    filtered_df = df[pd.to_datetime(start_time):pd.to_datetime(end_time)]

    baseline_stats = {}
    for sensor in ["left_out", "left_cen", "left_in", "right_out", "right_cen", "right_in"]:
        baseline_stats[sensor] = {
            "mean": filtered_df[sensor].mean(),
            "std": max(filtered_df[sensor].std(), 5)
        }

    return baseline_stats


def create_piezo_baseline_from_piezo_df(df: pd.DataFrame, start_time: str, end_time: str):
    filtered_df = df[pd.to_datetime(start_time):pd.to_datetime(end_time)]
    baseline_stats = {}
    for sensor in ["left1_avg", "left2_avg", "right1_avg", "right2_avg"]:
        baseline_stats[sensor] = {
            "mean": filtered_df[sensor].mean(),
            "std": max(filtered_df[sensor].std(), 5)
        }
    return baseline_stats


def calculate_std(arr):
    return np.std(arr)


def calculate_avg(arr):
    return np.mean(arr)


def load_piezo_df(data: Data) -> pd.DataFrame:
    df = pd.DataFrame(data['piezo_dual'])
    df.sort_values(by='ts', inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)

    # df['left1_std'] = df['left1'].apply(calculate_std)
    # df['left2_std'] = df['left2'].apply(calculate_std)
    #
    # df['right1_std'] = df['right1'].apply(calculate_std)
    # df['right2_std'] = df['right2'].apply(calculate_std)

    df['left1_avg'] = df['left1'].apply(calculate_avg)
    # df['left2_avg'] = df['left2'].apply(calculate_avg)

    df['right1_avg'] = df['right1'].apply(calculate_avg)
    # df['right2_avg'] = df['right2'].apply(calculate_avg)
    for column in ['left1_avg', 'right1_avg']:
        upper_bound = np.percentile(df[column], 99)  # 99th percentile
        lower_bound = np.percentile(df[column], 1)  # 1st percentile
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df.drop(columns=['left1', 'left2', 'right1', 'right2', 'type', 'freq', 'adc', 'gain', 'seq'], inplace=True)
    return df




def smooth_outliers(df, columns):
    for col in columns:
        lower_bound = df[col].quantile(0.01)  # 1st percentile
        upper_bound = df[col].quantile(0.99)  # 99th percentile
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


# ---------------------------------------------------------------------------------------------------
# CAP SENSORS

folder_path = '/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-26'
data = load_raw_data(folder_path=folder_path)
cap_df = load_cap_df(data)
cap_baseline = create_cap_baseline_from_cap_df(cap_df, '2025-01-26 02:00:00', '2025-01-26 04:00:00')
# piezo_baseline = create_piezo_baseline_from_piezo_df(piezo_df, '2025-01-26 02:00:00', '2025-01-26 04:00:00')

# ---------------------------------------------------------------------------------------------------

piezo_df = load_piezo_df(data)
piezo_df['right1_min'] = piezo_df['right1_avg'].rolling(window='10s', center=True).min()
piezo_df['right1_max'] = piezo_df['right1_avg'].rolling(window='10s', center=True).max()
piezo_df['right1_range'] = abs(piezo_df['right1_max'] - piezo_df['right1_min'])


piezo_df['left1_min'] = piezo_df['left1_avg'].rolling(window='10s', center=True).min()
piezo_df['left1_max'] = piezo_df['left1_avg'].rolling(window='10s', center=True).max()
piezo_df['left1_range'] = abs(piezo_df['left1_max'] - piezo_df['left1_min'])


piezo_df['piezo_right1_presence'] = piezo_df['right1_range'].apply(lambda x: 1 if x > 50_000 else 0)
piezo_df['piezo_left1_presence'] = piezo_df['left1_range'].apply(lambda x: 1 if x > 50_000 else 0)

# piezo_df['piezo_left1_presence'] = (
#     piezo_df[['piezo_left1_presence']]
#     .rolling('30s', min_periods=1)
#     .max()
#     ['piezo_left1_presence']
#     .astype(int)
# )
piezo_df['piezo_left1_presence'] = (
    piezo_df['piezo_left1_presence']
    .rolling('120s', min_periods=1)  # 60s rolling window
    .apply(lambda x: 1 if x.sum() >= 45 else 0, raw=False)
    .astype(int)
)



piezo_df['piezo_right1_presence'] = (
    piezo_df['piezo_right1_presence']
    .rolling('120s', min_periods=1)  # 60s rolling window
    .apply(lambda x: 1 if x.sum() >= 45 else 0, raw=False)
    .astype(int)
)

# ---------------------------------------------------------------------------------------------------

# A helper function to turn the raw sensor reading into a "delta from baseline"
# You can scale by standard deviations, or by raw difference from the mean, etc.
def sensor_delta(row, sensor, baseline_stats):
    # difference from mean in terms of # of standard deviations
    return (row[sensor] - baseline_stats[sensor]["mean"]) / baseline_stats[sensor]["std"]

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


# ---------------------------------------------------------------------------------------------------


occupancy_threshold = 25

# cap_df["right_occupied"] = cap_df["right_combined"] > occupancy_threshold
cap_df["left_occupied"] = cap_df["left_combined"].apply(lambda x: 1 if x > occupancy_threshold else 0)
cap_df["right_occupied"] = cap_df["right_combined"].apply(lambda x: 1 if x > occupancy_threshold else 0)

# Optional: apply a rolling minimum to require that the signal stays
# above the threshold for at least, say, 10 seconds in a row before flipping "occupied"
cap_df["cap_left_occupied"] = (
    cap_df["left_occupied"]
    .rolling("120s", min_periods=1)
    .apply(lambda x: 1 if np.all(x) else 0, raw=False)
    .astype(int)
)

cap_df["cap_right_occupied"] = (
    cap_df["right_occupied"]
    .rolling("120s", min_periods=1)
    .apply(lambda x: 1 if np.all(x) else 0, raw=False)
    .astype(int)
)
# ---------------------------------------------------------------------------------------------------
merged_df = piezo_df.merge(cap_df, on='ts', how='inner')
merged_df['final_left_occupied'] = merged_df['piezo_left1_presence'] + merged_df['cap_left_occupied']
merged_df['final_right_occupied'] = merged_df['piezo_right1_presence'] + merged_df['cap_right_occupied']


plot_cap_presence(merged_df, start_time="2025-01-26 04:00:00", end_time="2025-01-26 17:00:00", title=f'occupancy_threshold: {occupancy_threshold} || ')


# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------

plot_df_column(merged_df, ['cap_left_occupied', 'piezo_left1_presence', 'left1_avg', 'left1_range'], start_time="2025-01-26 06:00:00", end_time="2025-01-26 06:30:00")
plot_df_column(piezo_df, ['piezo_left1_presence'], start_time="2025-01-26 08:00:00", end_time="2025-01-26 10:00:00")


plot_df_column(piezo_df, ['left1_avg', 'left1_std', 'right1_avg', 'right1_std'], start_time="2025-01-26 03:00:00", end_time="2025-01-26 10:00:00")
plot_df_column(piezo_df, ['left1_avg', 'right1_avg'], start_time="2025-01-26 04:00:00", end_time="2025-01-26 07:30:00")
plot_df_column(piezo_df, ['left1_avg'], start_time="2025-01-26 03:06:00", end_time="2025-01-26 03:07:00")
plot_df_column(piezo_df, ['left1_avg', 'right1_avg'], start_time="2025-01-26 03:00:00", end_time="2025-01-26 10:00:00")
plot_df_column(piezo_df, ['right1_avg'], start_time="2025-01-26 06:00:00", end_time="2025-01-26 16:00:00")



empty_min = piezo_df[pd.to_datetime("2025-01-26 03:05:00"):pd.to_datetime("2025-01-26 03:10:00")]['left1_avg'].min()
empty_max = piezo_df[pd.to_datetime("2025-01-26 03:05:00"):pd.to_datetime("2025-01-26 03:10:00")]['left1_avg'].max()
# 85,556
empty_range = empty_max - empty_min


tally_min = piezo_df[pd.to_datetime("2025-01-26 05:00:00"):pd.to_datetime("2025-01-26 05:10:00")]['left1_avg'].min()
tally_max = piezo_df[pd.to_datetime("2025-01-26 05:00:00"):pd.to_datetime("2025-01-26 05:10:00")]['left1_avg'].max()

# 1,677,570
tally_range = tally_max - tally_min
tally_range



tally_min = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['left1_avg'].min()
tally_max = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['left1_avg'].max()

# 295,224
tally_range = tally_max - tally_min
tally_range


piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['right1_range']
david_min_gone = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['right1_avg'].min()
david_max_gone = piezo_df[pd.to_datetime("2025-01-26 06:28:00"):pd.to_datetime("2025-01-26 06:30:00")]['right1_avg'].max()
# 79,303
david_range_gone = david_max_gone - david_min_gone



david_min_present = piezo_df[pd.to_datetime("2025-01-26 07:00:00"):pd.to_datetime("2025-01-26 07:10:00")]['right1_avg'].min()
david_max_present = piezo_df[pd.to_datetime("2025-01-26 07:00:00"):pd.to_datetime("2025-01-26 07:10:00")]['right1_avg'].max()
# 1,900,295
david_range_present = david_max_present - david_min_present




