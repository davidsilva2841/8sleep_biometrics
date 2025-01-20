import json
import gc
from calculations import RunData
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import PROJECT_FOLDER_PATH

from data_manager import DataManager


# ---------------------------------------------------------------------------------------------------

# region PLOTTING

def _plot_validation_data(
        start_time: str,
        end_time: str,
        data: DataManager,
        estimated_df: pd.DataFrame = None,
        info: dict = None,
        stats_json: dict=None,
):
    try:
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
            # _df = _df[~_df.index.duplicated(keep='first')]

            # Select only numeric columns
            # numeric_columns = _df.select_dtypes(include=['number'])

            # Resample and aggregate using mean
            # resampled_df = numeric_columns.resample(freq).mean()

            # Interpolate to fill missing values
            # resampled_df = resampled_df.interpolate(method='linear')

            # return resampled_df
            return _df

        def resample_df_sleep(df, time_col, freq='10S'):
            _df = df.copy()
            # Make sure we convert the time column to datetime
            _df[time_col] = pd.to_datetime(_df[time_col]).dt.tz_localize(None)
            # Set the index to time_col
            _df.set_index(time_col, inplace=True)
            # Sort by the index so resample can work properly
            _df.sort_index(inplace=True)
            # _df = _df[~_df.index.duplicated(keep='first')]
            # Resample and forward fill
            # _df = _df.resample(freq).ffill()
            return _df

        # -------------------------------------------------------------------------
        # 2. Resample each DataFrame (breath_rate, hr, hrv, sleep)
        # -------------------------------------------------------------------------
        breath_rate_resampled = resample_df(data.breath_rate_df, time_col='start_time')
        hr_resampled = resample_df(data.heart_rate_df, time_col='start_time')
        hrv_resampled = resample_df(data.hrv_df, time_col='start_time')

        # Sleep
        sleep_stage_map = {
            'awake': 0,
            'asleepCore': 1,
            'asleepREM': 2,
        }
        sleep_df_copy = data.sleep_df.copy()
        sleep_df_copy['sleep_stage_actual'] = sleep_df_copy['sleep_stage_actual'].map(sleep_stage_map)
        sleep_resampled = resample_df_sleep(
            sleep_df_copy,
            time_col='start_time',
        )

        # --------------------------------------- ----------------------------------
        # 3. Clip/respect the specified start/end times
        # -------------------------------------------------------------------------
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)

        breath_rate_resampled = breath_rate_resampled.loc[
            (breath_rate_resampled.index >= start_dt) & (breath_rate_resampled.index <= end_dt)
            ]
        hr_resampled = hr_resampled.loc[
            (hr_resampled.index >= start_dt) & (hr_resampled.index <= end_dt)

            ]
        hrv_resampled = hrv_resampled.loc[
            (hrv_resampled.index >= start_dt) & (hrv_resampled.index <= end_dt)
            ]
        sleep_resampled = sleep_resampled.loc[
            (sleep_resampled.index >= start_dt) & (sleep_resampled.index <= end_dt)
            ]

        # -------------------------------------------------------------------------
        # 4. If an estimated_df is provided, resample and clip it as well
        # -------------------------------------------------------------------------
        if estimated_df is not None:
            estimated_df['start_time'] = pd.to_datetime(estimated_df['start_time']).dt.tz_localize(None)
            estimated_df_resampled = resample_df(estimated_df, time_col='start_time')
            estimated_df_resampled = estimated_df_resampled.loc[
                (estimated_df_resampled.index >= start_dt) &
                (estimated_df_resampled.index <= end_dt)
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
            breath_rate_resampled.index,
            breath_rate_resampled['breath_rate_actual'],
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
            hr_resampled['heart_rate_actual'],
            color='#f44336',
            label='Heart Rate',
            linewidth=2
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
        df = data.heart_rate_df.copy()
        # Convert the time column to datetime
        df['start_time'] = pd.to_datetime(df['start_time']).dt.tz_localize(None)
        # Set the time column as the index
        df.set_index('start_time', inplace=True)
        # Ensure the index is sorted
        df.sort_index(inplace=True)
        axes[1].scatter(
            df.index,
            df['heart_rate_actual'],
            color='#f44336',
            label='Heart Rate',
            s=30,
        )
        axes[1].set_ylim(45, 90)
        axes[1].set_ylabel("Heart Rate (count/min)")
        axes[1].legend(loc='upper left')
        axes[1].grid(True)

        # Plot 3: HRV
        axes[2].plot(
            hrv_resampled.index,
            hrv_resampled['hrv_actual'],
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
            sleep_resampled['sleep_stage_actual'],
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
        axes[-1].set_xlim([start_dt, end_dt])

        # Format the x-axis time
        time_format = DateFormatter("%H:%M:%S")
        for ax in axes:
            ax.xaxis.set_major_formatter(time_format)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=30)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.20)  # Leave space at the bottom

        # Add JSON stats at the bottom
        if stats_json is not None:
            stats_text = "\n".join(f"{key}: {value}" for key, value in stats_json.items())
            fig.text(
                0.5, 0.15, # Position: x, y (centered horizontally below the chart)
                stats_text,
                ha='left', va='top', fontsize=12, family='monospace'
            )
        if info is not None:
            info_text = "\n".join(f"{key}: {value}" for key, value in info.items())
            fig.text(
                0.25, 0.15, # Position: x, y (centered horizontally below the chart)
                info_text,
                ha='left', va='top', fontsize=12, family='monospace'
            )
        file_name = f'{end_time[:10]}_{info["name"]}_mae_{stats_json["mae"]}_corr_{stats_json["corr"]}_mape_{stats_json["mape"]}.png'
        save_path = f'{PROJECT_FOLDER_PATH}raw/plots/dump/{file_name}'
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show(block=False)
    except Exception as e:
        print('ERROR -----------------------------------------------------------------------------------------------------')
        print(e)
        traceback.print_exc()

# endregion

# ---------------------------------------------------------------------------------------------------
# region ANALYZE

def split_df_by_source(df_pred: pd.DataFrame, column: str):
    unique_values = df_pred[column].unique()
    df_dict = {value: df_pred[df_pred[column] == value].copy() for value in unique_values}
    return df_dict


def analyze_predictions(data: DataManager, df_pred: pd.DataFrame, run_data: RunData, plot=True):
    df_pred.sort_values(by=['source', 'start_time'], inplace=True)
    df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])
    df_pred['end_time'] = pd.to_datetime(df_pred['end_time'])
    df_pred['mid_time'] = df_pred['start_time']
    df_dict = split_df_by_source(df_pred, 'source')

    data.heart_rate_df['start_time'] = pd.to_datetime(data.heart_rate_df['start_time'])
    column = 'heart_rate'

    # Compute the midpoint of start_time and end_time
    results = {}
    for source, split_df in df_dict.items():
        split_df.dropna(subset=[column], inplace=True)
        split_df_merged = pd.merge_asof(
            data.heart_rate_df,
            split_df,
            left_on='start_time',
            right_on='mid_time',
            direction='nearest',
            tolerance=pd.Timedelta('30s')
        )
        split_df_merged.dropna(subset=[column], inplace=True)

        pre_merge_count = split_df.shape[0]
        post_merge_count = split_df_merged.shape[0]

        results[source] = {
            'mean': round(split_df_merged[column].mean(), 2),
            'std': round(split_df_merged[column].std(), 2),
            'min': round(split_df_merged[column].min(), 2),
            'max': round(split_df_merged[column].max(), 2),
            'corr': f'{round(split_df_merged['heart_rate_actual'].corr(split_df_merged['heart_rate']), 2) * 100:.2f}%',
            'mae': round(mean_absolute_error(split_df_merged['heart_rate_actual'], split_df_merged['heart_rate']), 2),
            'mse': round(mean_squared_error(split_df_merged['heart_rate_actual'], split_df_merged['heart_rate']), 2),
            'mape': f"{round(np.mean(np.abs((split_df_merged['heart_rate_actual'] - split_df_merged['heart_rate']) / split_df_merged['heart_rate_actual'])) * 100, 2)}%",
            'rmse': round(np.sqrt(mean_squared_error(split_df_merged['heart_rate_actual'], split_df_merged['heart_rate'])), 2),
            'pre_merge_count': pre_merge_count,
            'post_merge_count': post_merge_count,
        }
        if plot:
            print(json.dumps(results, indent=4))
            _plot_validation_data(run_data.start_time, run_data.end_time, data, split_df, run_data.chart_info, results[source])

        del df_dict
        gc.collect()
        return results[source]


# endregion
