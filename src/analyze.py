"""
- Plots predictions
- Analyzes prediction accuracy for heart rate
"""
import json
import gc
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
import traceback
from typing import Union, TypedDict, Literal, List, Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import PROJECT_FOLDER_PATH

from data_manager import DataManager
from run_data import ChartInfo

# ---------------------------------------------------------------------------------------------------
# region Types

EvaluationMetric = Literal['heart_rate', 'hrv', 'sleep_stage', 'breathing_rate']

class AccuracyMetrics(TypedDict):
    corr: str
    mae: float
    mse: float
    mape: str
    rmse: float


class StatisticalMetrics(TypedDict):
    mean: float
    std: float
    min: float
    max: float


class EvaluationMetrics(TypedDict):
    accuracy: AccuracyMetrics
    statistics: StatisticalMetrics


class Results(TypedDict):
    heart_rate: Optional[EvaluationMetrics]
    hrv: Optional[EvaluationMetrics]
    sleep_stage: Optional[EvaluationMetrics]
    breathing_rate: Optional[EvaluationMetrics]

# endregion


# ---------------------------------------------------------------------------------------------------
# region PLOTTING

def _plot_validation_data(
        start_time: str,
        end_time: str,
        data: DataManager,
        df_pred: pd.DataFrame = None,
        chart_info: ChartInfo = None,
        results: Results=None,
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
            _df = _df[~_df.index.duplicated(keep='first')]
            # Resample and forward fill
            _df = _df.resample(freq).ffill()
            return _df

        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)
        # -------------------------------------------------------------------------
        # 2. Resample each DataFrame (breath_rate, hr, hrv, sleep)
        # -------------------------------------------------------------------------
        if not data.breath_rate_df.empty:
            breath_rate_resampled = resample_df(data.breath_rate_df, time_col='start_time')
            breath_rate_resampled = breath_rate_resampled.loc[
                (breath_rate_resampled.index >= start_dt) & (breath_rate_resampled.index <= end_dt)
                ]
        hr_resampled = resample_df(data.heart_rate_df, time_col='start_time')
        hrv_resampled = resample_df(data.hrv_df, time_col='start_time')

        if not data.sleep_df.empty:
            # Sleep
            sleep_stage_map = {
                'awake': 0,
                'asleepCore': 1,
                'asleepDeep': 2,
                'asleepREM': 3,
            }
            sleep_df_copy = data.sleep_df.copy()
            sleep_df_copy['sleep_stage_actual'] = sleep_df_copy['sleep_stage_actual'].map(sleep_stage_map)
            sleep_resampled = resample_df_sleep(sleep_df_copy, time_col='start_time')
            sleep_resampled = sleep_resampled.loc[
                (sleep_resampled.index >= start_dt) & (sleep_resampled.index <= end_dt)
                ]
        # --------------------------------------- ----------------------------------
        # 3. Clip/respect the specified start/end times
        # -------------------------------------------------------------------------

        hr_resampled = hr_resampled.loc[
            (hr_resampled.index >= start_dt) & (hr_resampled.index <= end_dt)

            ]
        hrv_resampled = hrv_resampled.loc[
            (hrv_resampled.index >= start_dt) & (hrv_resampled.index <= end_dt)
            ]


        # -------------------------------------------------------------------------
        # 4. If an estimated_df is provided, resample and clip it as well
        # -------------------------------------------------------------------------
        if df_pred is not None:
            df_pred['start_time'] = pd.to_datetime(df_pred['start_time']).dt.tz_localize(None)
            estimated_df_resampled = resample_df(df_pred, time_col='start_time')
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

        if not data.breath_rate_df.empty:
            # Plot 1: Breath Rate
            axes[0].plot(
                breath_rate_resampled.index,
                breath_rate_resampled['breathing_rate_actual'],
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

        axes[1].set_ylim(estimated_df_resampled['heart_rate'].min() - 2, estimated_df_resampled['heart_rate'].max() + 2)
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

        if not data.sleep_df.empty:
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
            axes[3].set_ylim(0, 4)
            axes[3].yaxis.set_major_locator(MaxNLocator(integer=True))
            sleep_stage_labels = ['Awake', 'Core', 'Deep', 'REM', '']
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

        if chart_info is not None:
            if 'labels' in chart_info:
                labels = "INFO\n" + "\n".join(f"{key.ljust(11)}: {value}" for key, value in chart_info['labels'].items())
                fig.text(
                    0.01, 0.15, # Position: x, y (centered horizontally below the chart)
                    labels,
                    ha='left', va='top', fontsize=16, family='monospace'
                )
            if 'runtime_params' in chart_info:
                labels = "RUNTIME PARAMETERS\n" + "\n".join(f"{key.ljust(17)}: {value}" for key, value in chart_info['runtime_params'].items())
                fig.text(
                    0.30, 0.15, # Position: x, y (centered horizontally below the chart)
                    labels,
                    ha='left', va='top', fontsize=16, family='monospace'
                )
        if results is not None:
            accuracy = results['heart_rate']['accuracy']
            stats_text = "HEART RATE ACCURACY\n" + "\n".join(f"{key.ljust(6)}: {value}" for key, value in accuracy.items())
            fig.text(
                0.57, 0.15, # Position: x, y (centered horizontally below the chart)
                stats_text,
                ha='left', va='top', fontsize=16, family='monospace'
            )

        rmse = results["heart_rate"]['accuracy']["rmse"]
        corr = results["heart_rate"]['accuracy']["corr"]
        file_name = f'{data.name}_{end_time[:10]}_rmse_{rmse}_corr_{corr}.png'
        if chart_info['labels']['label']:
            file_name = chart_info['labels']['label'] + '_' + file_name
        save_path = f'{PROJECT_FOLDER_PATH}tmp/plots/{file_name}'
        print(f'Saving plot to: {save_path}')
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show(block=False)
    except Exception as e:
        print('ERROR -----------------------------------------------------------------------------------------------------')
        print(e)
        traceback.print_exc()





# endregion

# ---------------------------------------------------------------------------------------------------
# region ANALYZE





def _calculate_accuracy(data: DataManager, df_pred: pd.DataFrame, column: EvaluationMetric) -> Union[EvaluationMetrics, {}]:
    if column == 'heart_rate':
        df_merged = pd.merge_asof(
            data.heart_rate_df,
            df_pred,
            left_on='start_time',
            right_on='start_time',
            direction='nearest',
            tolerance=pd.Timedelta('30s')
        )
    elif column == 'breathing_rate':
        df_merged = pd.merge_asof(
            data.breath_rate_df,
            df_pred,
            left_on='start_time',
            right_on='start_time',
            direction='nearest',
            tolerance=pd.Timedelta('30s')
        )
    elif column == 'hrv':
        df_merged = pd.merge_asof(
            data.hrv_df,
            df_pred,
            left_on='start_time',
            right_on='start_time',
            direction='nearest',
            tolerance=pd.Timedelta('30s')
        )
    else:
        return
    actual_column_name = f'{column}_actual'
    df_merged.dropna(subset=[column], inplace=True)
    if df_merged.shape[0] == 0:
        return {}
    res = {
        'statistics': {
            'mean': round(df_merged[column].mean(), 2),
            'std': round(df_merged[column].std(), 2),
            'min': round(df_merged[column].min(), 2),
            'max': round(df_merged[column].max(), 2),
        },
        'accuracy': {
            'corr': f'{round(df_merged[actual_column_name].corr(df_merged[column]), 2) * 100:.2f}%',
            'mape': f"{round(np.mean(np.abs((df_merged[actual_column_name] - df_merged[column]) / df_merged[actual_column_name])) * 100, 2)}%",
            'mae': round(mean_absolute_error(df_merged[actual_column_name], df_merged[column]), 2),
            'rmse': round(np.sqrt(mean_squared_error(df_merged[actual_column_name], df_merged[column])), 2),
            'mse': round(mean_squared_error(df_merged[actual_column_name], df_merged[column]), 2),
        }
    }
    del df_merged
    gc.collect()
    return res



def analyze_predictions(
        data: DataManager,
        df_pred: pd.DataFrame,
        start_time: str,
        end_time: str,
        chart_info: ChartInfo = None,
        plot=True
) -> Results:
    df_pred.sort_values(by=['start_time'], inplace=True)
    df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])

    # Only concerned with heart rate now
    # columns: List[EvaluationMetric] = ['heart_rate', 'hrv', 'sleep_stage', 'breathing_rate']
    columns: List[EvaluationMetric] = ['heart_rate']

    results: Results = {}
    for column in columns:
        results[column] = _calculate_accuracy(data, df_pred, column)

    if plot:
        print('-----------------------------------------------------------------------------------------------------')
        _plot_validation_data(start_time, end_time, data, df_pred, chart_info, results=results)
        print(json.dumps(results, indent=4))


    gc.collect()
    return results


# endregion
