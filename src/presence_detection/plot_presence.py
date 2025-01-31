# scp -r -P 8822 'pod1:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/no_presence
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from load_raw import load_raw_data
from data_types import Data
import matplotlib.ticker as ticker


def plot_cap_presence(df: pd.DataFrame, title: str = '', start_time: str = None, end_time: str = None):
    if start_time:
        title_start_time = start_time
    else:
        title_start_time = df.index[0]

    if end_time:
        title_end_time = end_time
    else:
        title_end_time = df.index[-1]

    title = f'{title} {title_start_time} -> {title_end_time}'

    # Ensure DataFrame index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Filter dataframe based on provided time range
    if start_time:
        start_time = pd.to_datetime(start_time)
        df = df[df.index >= start_time]

    if end_time:
        end_time = pd.to_datetime(end_time)
        df = df[df.index <= end_time]

    # Select only numeric columns to avoid errors during resampling
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_resampled = df[numeric_cols].resample('30s').mean()

    # Convert final_left_occupied and final_right_occupied to binary presence values
    df_resampled['left_present'] = (df_resampled['final_left_occupied'] == 2).astype(int)
    df_resampled['right_present'] = (df_resampled['final_right_occupied'] == 2).astype(int)

    # Define sensor columns
    left_sensors = ['left_out', 'left_cen', 'left_in']
    right_sensors = ['right_out', 'right_cen', 'right_in']

    fig, axs = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

    # Plot left side capacitance values (1st plot)
    for col in left_sensors:
        axs[0].plot(df_resampled.index, df_resampled[col], label=col)
    axs[0].set_ylabel('Left Side Capacitance')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, linestyle='--', linewidth=0.75)

    # Plot left side occupancy status (2nd plot)
    axs[1].plot(df_resampled.index, df_resampled['left_present'], label='Left Present', color='purple', linewidth=4)
    axs[1].set_ylabel('Left Occupancy')
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['Not Present', 'Present'])
    axs[1].legend(loc='upper left')
    axs[1].grid(True, linestyle='--', linewidth=0.75)

    # Plot right side capacitance values (3rd plot)
    for col in right_sensors:
        axs[2].plot(df_resampled.index, df_resampled[col], label=col)
    axs[2].set_ylabel('Right Side Capacitance')
    axs[2].legend(loc='upper left')
    axs[2].grid(True, linestyle='--', linewidth=0.75)

    # Plot right side occupancy status (4th plot)
    axs[3].plot(df_resampled.index, df_resampled['right_present'], label='Right Present', color='purple', linewidth=4)
    axs[3].set_ylabel('Right Occupancy')
    axs[3].set_yticks([0, 1])
    axs[3].set_yticklabels(['Not Present', 'Present'])
    axs[3].legend(loc='upper left')
    axs[3].grid(True, linestyle='--', linewidth=0.75)

    axs[3].set_xlabel('Timestamp')

    # Format x-axis timestamps
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=45)  # Rotate timestamps for better readability

    plt.suptitle(title)

    plt.tight_layout()
    plt.show()



def plot_occupancy(df: pd.DataFrame, title: str = '', start_time: str = None, end_time: str = None):



    # Ensure DataFrame index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Convert index to time-only format for filtering
    df['time'] = df.index.strftime('%H:%M')

    # Filter dataframe based on provided time range (ignoring date)
    if start_time:
        df = df[df['time'] >= start_time]

    if end_time:
        df = df[df['time'] <= end_time]

    # Select numeric columns to avoid errors during resampling
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_resampled = df[numeric_cols].resample('30S').mean()

    # Convert occupancy columns to binary presence values
    df_resampled['final_left_present'] = (df_resampled['final_left_occupied'] == 2).astype(int)
    df_resampled['final_right_present'] = (df_resampled['final_right_occupied'] == 2).astype(int)
    df_resampled['cap_left_present'] = (df_resampled['cap_left_occupied'] == 1).astype(int)
    df_resampled['piezo_left_present'] = (df_resampled['piezo_left1_presence'] == 1).astype(int)
    df_resampled['cap_right_present'] = (df_resampled['cap_right_occupied'] == 1).astype(int)
    df_resampled['piezo_right_present'] = (df_resampled['piezo_right1_presence'] == 1).astype(int)

    fig, axs = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

    # 1st chart: final_left_present
    axs[0].plot(df_resampled.index, df_resampled['final_left_present'], label='Final Left Present (2=Present)', color='purple', linewidth=3)
    axs[0].set_ylabel('Final Left Presence')
    axs[0].legend(loc='upper left')
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(['Not Present', 'Present'])
    axs[0].grid(True, linestyle='--', linewidth=0.75)

    # 2nd chart: cap_left_occupied & piezo_left1_presence
    axs[1].plot(df_resampled.index, df_resampled['cap_left_present'], label='Cap Left Occupied (1=Present)', color='orange', linewidth=4)
    axs[1].plot(df_resampled.index, df_resampled['piezo_left_present'], label='Piezo Left Present (1=Present)', color='brown', linewidth=2)
    axs[1].set_ylabel('Left Sensors')
    axs[1].legend(loc='upper left')
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(['Not Present', 'Present'])
    axs[1].grid(True, linestyle='--', linewidth=0.75)

    # 3rd chart: final_right_present
    axs[2].plot(df_resampled.index, df_resampled['final_right_present'], label='Final Right Present (2=Present)', color='purple', linewidth=3)
    axs[2].set_ylabel('Final Right Presence')
    axs[2].legend(loc='upper left')
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['Not Present', 'Present'])
    axs[2].grid(True, linestyle='--', linewidth=0.75)

    # 4th chart: cap_right_occupied & piezo_right1_presence
    axs[3].plot(df_resampled.index, df_resampled['cap_right_present'], label='Cap Right Occupied (1=Present)', color='orange', linewidth=4)
    axs[3].plot(df_resampled.index, df_resampled['piezo_right_present'], label='Piezo Right Present (1=Present)', color='brown', linewidth=2)
    axs[3].set_ylabel('Right Sensors')
    axs[3].legend(loc='upper left')
    axs[3].set_yticks([0, 1])
    axs[3].set_yticklabels(['Not Present', 'Present'])
    axs[3].grid(True, linestyle='--', linewidth=0.75)

    axs[3].set_xlabel('Timestamp')

    # Format x-axis timestamps
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    title_start_time = df.index[0].strftime('%Y-%m-%d %H:%M')
    title_end_time = df.index[-1].strftime('%Y-%m-%d %H:%M')

    title = f'{title} {title_start_time} -> {title_end_time}'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_df_column(df: pd.DataFrame, columns: List[str], start_time: str = None, end_time: str = None):
    if start_time:
        title_start_time = start_time
    else:
        title_start_time = df.index[0]

    if end_time:
        title_end_time = end_time
    else:
        title_end_time = df.index[-1]

    # Filter dataframe based on provided time range
    if start_time:
        start_time = pd.to_datetime(start_time)
        df = df[df.index >= start_time]

    if end_time:
        end_time = pd.to_datetime(end_time)
        df = df[df.index <= end_time]

    # Drop top and bottom 1% outliers for each column
    df_filtered = df.copy()

    # Create subplots for each column
    fig, axes = plt.subplots(nrows=len(columns), ncols=1, figsize=(12, 6 * len(columns)), sharex=True)

    # Ensure axes is always iterable
    if len(columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, columns):
        ax.plot(df_filtered.index, df_filtered[column], label=column, color='b')
        ax.set_ylabel(column)
        ax.legend()
        ax.grid(True, linestyle='--')

        # Format x-axis to display time as HH:mm
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # Format y-axis to avoid scientific notation
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

    plt.xlabel('Time')
    plt.suptitle(f'{title_start_time} -> {title_end_time}')
    plt.xticks(rotation=45)  # Rotate timestamps for better readability
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to fit title
    plt.show()



