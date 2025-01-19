from datetime import datetime, timedelta
import heartpy as hp
import json
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error





# Validation data
heart_rate_df = pd.read_csv('./heart_rate.csv')
respiratory_rate_df = pd.read_csv('./respiratory_rate.csv')
sleep_stage_df = pd.read_csv('./sleep_stage.csv')
hrv_df = pd.read_csv('./hrv.csv')

# Raw sensor data
sensor_df = pd.read_pickle('./sensor_data.pkl')
sensor_df.sort_values(by='timestamp', ascending=True, inplace=True)
sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
sensor_df.set_index('timestamp', inplace=True)

# Time period of sleep analysis
start_time = '2025-01-15 06:30:00'
end_time = '2025-01-15 14:00:00'
start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')


# Iterate over the timespan
window_seconds = 10     # Window we'll be calculating
slide_by_seconds = 5    # Iteration interval

window_time = timedelta(seconds=window_seconds)
slide_by_time = timedelta(seconds=slide_by_seconds)

start_interval = start_dt
end_interval = start_interval + slide_by_time


# Predicting heart rates
measurements = []
while start_interval <= end_dt:
    try:

        # Create a single dimensional np array for the interval, each row is a second of data with 500 measurements, so 10 seconds would be 5,000 measurements
        #
        # NOTE: THERE IS A SECOND SENSOR `right_sensor_2`, we could update this to calculate the HR against both sensors and determine the average
        # I wanted to keep this example simple though
        np_array = np.concatenate(sensor_df[start_interval:end_interval]['right_sensor_1'])
        data = hp.filter_signal(np_array, [0.05, 15], 500, filtertype='bandpass')
        data = hp.scale_data(data)
        working_data, measurement = hp.process(
            data,
            500,
            freq_method='fft',
            breathing_method='fft',
            bpmmin=40,
            hampel_correct=False,  # KEEP FALSE - Takes too long
            bpmmax=90,
            reject_segmentwise=False,  # KEEP FALSE - Less accurate
            windowsize=0.5,
            clipping_scale=False,  # KEEP FALSE - Did not change reading
            clean_rr=True,  # KEEP TRUE - More accurate
            clean_rr_method='quotient-filter',  # z-score is worse
        )
        measurements.append({
            'start_time': start_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'heart_rate': measurement['bpm'],
            'hrv': measurement['sdnn'],
            'breathing_rate': measurement['breathingrate'] * 60,
        })
    except Exception as e:
        pass

    start_interval += slide_by_time
    end_interval += slide_by_time


# Create predictions dataframe
df_pred = pd.DataFrame(measurements)


# ---------------------------------------------------------------------------------------------------
# Analyzing results

def analyze_predictions(df_pred: pd.DataFrame) -> (pd.DataFrame, dict):
    df_pred.sort_values(by=['start_time'], inplace=True)
    df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])
    df_pred['end_time'] = pd.to_datetime(df_pred['end_time'])
    # Compute the midpoint of start_time and end_time
    df_pred['mid_time'] = df_pred['start_time'] + (df_pred['end_time'] - df_pred['start_time']) / 2

    heart_rate_df['start_time'] = pd.to_datetime(heart_rate_df['start_time'])

    df_combined = pd.merge_asof(
        heart_rate_df,
        df_pred,
        left_on='start_time',
        right_on='mid_time',
        direction='nearest',
        tolerance=pd.Timedelta('60s')
    )
    column = 'heart_rate'
    df_combined.dropna(subset=['heart_rate'], inplace=True)
    results = {
        'mean': round(df_combined[column].mean(), 2),
        'std': round(df_combined[column].std(), 2),
        'min': round(df_combined[column].min(), 2),
        'max': round(df_combined[column].max(), 2),
        'corr': f'{round(df_combined['actual_heart_rate'].corr(df_combined['heart_rate']), 2) * 100:.2f}%',
        'mae': round(mean_absolute_error(df_combined['actual_heart_rate'], df_combined['heart_rate']), 2),
        'mse': round(mean_squared_error(df_combined['actual_heart_rate'], df_combined['heart_rate']), 2),
        'mape': f"{round(np.mean(np.abs((df_combined['actual_heart_rate'] - df_combined['heart_rate']) / df_combined['actual_heart_rate'])) * 100, 2)}%",
        'rmse': round(np.sqrt(mean_squared_error(df_combined['actual_heart_rate'], df_combined['heart_rate'])), 2),
    }
    print(json.dumps(results, indent=4))
    return (df_combined, results)


# ---------------------------------------------------------------------------------------------------
# Plotting results


def plot_validation_data(
        start_time: str,
        end_time: str,
        df_pred: pd.DataFrame = None,
):
    # -------------------------------------------------------------------------
    # 1. Helper function to parse times and resample data at 10-second intervals
    # -------------------------------------------------------------------------
    def resample_df(df: pd.DataFrame, time_col, freq='10s'):
        _df = df.copy()
        # Convert the time column to datetime
        _df[time_col] = pd.to_datetime(_df[time_col]).dt.tz_localize(None)
        # Set the time column as the index
        _df.set_index(time_col, inplace=True)
        # Ensure the index is sorted
        _df.sort_index(inplace=True)
        # Remove duplicate index entries
        _df = _df[~_df.index.duplicated(keep='first')]
        # Select only numeric columns
        numeric_columns = _df.select_dtypes(include=['number'])
        # Resample and aggregate using mean
        resampled_df = numeric_columns.resample(freq).mean()
        # Interpolate to fill missing values
        resampled_df = resampled_df.interpolate(method='linear')
        return resampled_df

    def resample_df_sleep(df, time_col, freq='10S'):
        _df = df.copy()
        # Make sure we convert the time column to datetime
        _df[time_col] = pd.to_datetime(_df[time_col]).dt.tz_localize(None)
        # Set the index to time_col
        _df.set_index(time_col, inplace=True)
        # Sort by the index so resample can work properly
        _df.sort_index(inplace=True)
        _df = _df[~_df.index.duplicated(keep='first')]
        # Resample and forward fill
        _df = _df.resample(freq).ffill()
        return _df

    (df_combined, results) = analyze_predictions(df_pred)

    # -------------------------------------------------------------------------
    # 2. Resample each DataFrame (breath_rate, hr, hrv, sleep)
    # -------------------------------------------------------------------------
    respiratory_rate_df_resampled = resample_df(respiratory_rate_df, time_col='start_time')
    hr_resampled = resample_df(heart_rate_df, time_col='start_time')
    hrv_resampled = resample_df(hrv_df, time_col='start_time')

    # Sleep
    sleep_stage_map = {
        'awake': 0,
        'asleepCore': 1,
        'asleepREM': 2,
    }
    sleep_df_copy = sleep_stage_df.copy()
    sleep_df_copy['sleep_stage'] = sleep_df_copy['sleep_stage'].map(sleep_stage_map)
    sleep_resampled = resample_df_sleep(
        sleep_df_copy,
        time_col='start_time',
    )

    # --------------------------------------- ----------------------------------
    # 3. Clip/respect the specified start/end times
    # -------------------------------------------------------------------------
    pd_start_time = pd.to_datetime(start_time)
    pd_end_time = pd.to_datetime(end_time)

    respiratory_rate_df_resampled = respiratory_rate_df_resampled.loc[
        (respiratory_rate_df_resampled.index >= pd_start_time) & (respiratory_rate_df_resampled.index <= pd_end_time)
        ]
    hr_resampled = hr_resampled.loc[
        (hr_resampled.index >= pd_start_time) & (hr_resampled.index <= pd_end_time)

        ]
    hrv_resampled = hrv_resampled.loc[
        (hrv_resampled.index >= pd_start_time) & (hrv_resampled.index <= pd_end_time)
        ]
    sleep_resampled = sleep_resampled.loc[
        (sleep_resampled.index >= pd_start_time) & (sleep_resampled.index <= pd_end_time)
        ]

    # -------------------------------------------------------------------------
    # 4. If an estimated_df is provided, resample and clip it as well
    # -------------------------------------------------------------------------
    if df_pred is not None:
        df_pred['start_time'] = pd.to_datetime(df_pred['start_time']).dt.tz_localize(None)
        estimated_df_resampled = resample_df(df_pred, time_col='start_time')
        estimated_df_resampled = estimated_df_resampled.loc[
            (estimated_df_resampled.index >= pd_start_time) &
            (estimated_df_resampled.index <= pd_end_time)
            ]
    else:
        estimated_df_resampled = None

    # -------------------------------------------------------------------------
    # 5. Plot the data in 4 subplots with a shared x-axis
    # -------------------------------------------------------------------------
    # Increase the figure width to create more whitespace on the right side.
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 15), sharex=True)

    # Adjust the right border to leave extra whitespace (0.6 means 60% width for plots, 40% whitespace).
    fig.subplots_adjust(right=0.6)

    # Plot 1: Breath Rate
    axes[0].plot(
        respiratory_rate_df_resampled.index,
        respiratory_rate_df_resampled['respiratory_rate'],
        color='#f44336',
        label='Breathing Rate',
        linewidth=4
    )
    # If estimated breath_rate is provided, overlay it
    if estimated_df_resampled is not None and 'breathing_rate' in estimated_df_resampled.columns:
        axes[0].plot(
            estimated_df_resampled.index,
            estimated_df_resampled['breathing_rate'],
            color='#757ce8',
            label='Estimated Breathing Rate',
            linewidth=2
        )
    axes[0].set_ylabel("Breath Rate (count/min)")
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    # Plot 2: Heart Rate
    axes[1].plot(
        hr_resampled.index,
        hr_resampled['actual_heart_rate'],
        color='#f44336',
        label='Heart Rate',
        linewidth=4
    )
    # If estimated heart rate columns are provided, overlay them
    if estimated_df_resampled is not None:
        if 'heart_rate' in estimated_df_resampled.columns and not estimated_df_resampled['heart_rate'].isnull().all():
            axes[1].plot(
                estimated_df_resampled.index,
                estimated_df_resampled['heart_rate'],
                color='#757ce8',
                label='Estimated Heart Rate',
                linewidth=2
            )
    axes[1].set_ylabel("Heart Rate (count/min)")
    axes[1].legend(loc='upper left')
    axes[1].grid(True)

    # Plot 3: HRV
    axes[2].plot(
        hrv_resampled.index,
        hrv_resampled['value'],
        color='#f44336',
        label='HRV',
        linewidth=4
    )
    if estimated_df_resampled is not None:
        if 'hrv' in estimated_df_resampled.columns and not estimated_df_resampled['hrv'].isnull().all():
            axes[2].plot(
                estimated_df_resampled.index,
                estimated_df_resampled['hrv'],
                color='#757ce8',
                label='Estimated HRV',
                linewidth=2
            )
    axes[2].set_ylabel("HRV (ms)")
    axes[2].legend(loc='upper left')
    axes[2].grid(True)

    # Plot 4: Sleep
    axes[3].step(
        sleep_resampled.index,
        sleep_resampled['sleep_stage'],
        where='post',
        color='purple',
        label='Sleep Stage',
        linewidth=4
    )
    axes[3].set_ylabel("Sleep Stage\n(0=awake,1=core,2=rem)")
    axes[3].legend(loc='upper left')
    axes[3].grid(True)
    axes[3].set_ylim(0, 3)
    axes[3].yaxis.set_major_locator(MaxNLocator(integer=True))
    sleep_stage_labels = ['Awake', 'Core', 'REM', '']
    axes[3].set_yticks(range(len(sleep_stage_labels)))
    axes[3].set_yticklabels(sleep_stage_labels)

    # Set the x-limits to the specified range
    axes[-1].set_xlim([pd_start_time, pd_end_time])

    # Format the x-axis time
    time_format = DateFormatter("%H:%M:%S")
    for ax in axes:
        ax.xaxis.set_major_formatter(time_format)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)  # Leave space at the bottom

    # Add results at the bottom of plot
    if results is not None:
        info_text = "\n".join(f"{key}: {value}" for key, value in results.items())
        fig.text(
            0.25, 0.16, # Position: x, y (centered horizontally below the chart)
            info_text,
            ha='left', va='top', fontsize=12, family='monospace'
        )
    plt.show(block=False)

# ---------------------------------------------------------------------------------------------------
# Viewing results


df_pred_copy = df_pred.copy()
df_pred = df_pred_copy.copy()
df_pred = df_pred[(df_pred['heart_rate'] <= 90) & (df_pred['heart_rate'] >= 40)]
df_pred.dropna(subset=['heart_rate'], inplace=True)


df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=5, min_periods=5).mean()

plot_validation_data(start_time, end_time, df_pred)

# {
#     "mean": 67.3,
#     "std": 4.45,
#     "min": 55.17,
#     "max": 78.7,
#     "corr": "41.00%",
#     "mae": 4.45,
#     "mse": 36.83,
#     "mape": "7.14%",
#     "rmse": 6.07
# }
