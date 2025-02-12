"""
This module provides functions for processing 8 sleep data and calculating biometrics

The module includes functionalities for:
- Cleaning and preprocessing raw sensor data.
- Running sliding window calculations to estimate heart rate trends.
- Handling and processing RunData objects for comprehensive heart rate interval estimation.

Key Functions:
- `clean_df_pred`: Cleans predicted breathing rate and HRV data, filling missing values and smoothing data.
- `_calculate`: Processes raw sensor data to derive heart rate, HRV, and breathing rate.
- `estimate_heart_rate_intervals`: Runs heart rate estimation over intervals and stores results.
"""
import gc
import numpy as np
from vitals.run_data import RunData
import traceback
import pandas as pd

import sys
import os
import platform
import warnings

if platform.system().lower() == 'linux':
    sys.path.append('/home/dac/python_packages/')
    sys.path.append('/home/dac/free-sleep/biometrics/')

sys.path.append(os.getcwd())
from data_types import *
from vitals.cleaning import interpolate_outliers_in_wave
from heart.filtering import filter_signal, remove_baseline_wander
from heart.preprocessing import scale_data, enhance_ecg_peaks
from heart.heartpy import process
from get_logger import get_logger

logger = get_logger()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# ---------------------------------------------------------------------------------------------------
# region CLEAN DF

def clean_df_pred(df_pred: pd.DataFrame) -> pd.DataFrame:
    breathing_lower_threshold = 10
    breathing_upper_threshold = 23

    # Replace values outside the threshold with NaN
    df_pred['breathing_rate'] = df_pred['breathing_rate'].where(
        (df_pred['breathing_rate'] >= breathing_lower_threshold) & (df_pred['breathing_rate'] <= breathing_upper_threshold), np.nan
    )

    # Fill NaN with the last valid value (forward fill)
    # df_pred['breathing_rate'] = df_pred['breathing_rate'].ffill()
    df_pred['breathing_rate'] = df_pred['breathing_rate'].interpolate(method='linear')

    # Fill any remaining NaN with a rolling mean
    window_size = 3
    df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=window_size, min_periods=1).mean()

    hrv_lower_threshold = 10
    hrv_upper_threshold = 100

    # Replace values outside the threshold with NaN
    df_pred['hrv'] = df_pred['hrv'].where(
        (df_pred['hrv'] >= hrv_lower_threshold) & (df_pred['hrv'] <= hrv_upper_threshold), np.nan
    )

    # Fill NaN with the last valid value (forward fill)
    # df_pred['hrv'] = df_pred['hrv'].ffill()
    df_pred['hrv'] = df_pred['hrv'].interpolate(method='linear')

    # Fill any remaining NaN with a rolling mean
    window_size = 30
    df_pred['hrv'] = df_pred['hrv'].rolling(window=window_size, min_periods=10).mean()
    return df_pred


# endregion


# ---------------------------------------------------------------------------------------------------
# region CALCULATIONS


def _calculate(run_data: RunData, side: str):
    # Get the signal
    np_array = np.concatenate(run_data.piezo_df[run_data.start_interval:run_data.end_interval][side])

    # Remove outliers from signal
    data = interpolate_outliers_in_wave(
        np_array,
        lower_percentile=run_data.signal_percentile[0],
        upper_percentile=run_data.signal_percentile[1]
    )

    data = scale_data(data, lower=0, upper=1024)
    data = remove_baseline_wander(data, sample_rate=500.0, cutoff=0.05)

    data = filter_signal(
        data,
        cutoff=[0.5, 20.0],
        sample_rate=500.0,
        order=2,
        filtertype='bandpass'
    )

    working_data, measurement = process(
        data,
        500,
        breathing_method='fft',
        bpmmin=40,
        bpmmax=90,
        reject_segmentwise=False,  # KEEP FALSE - Less accurate
        windowsize=run_data.window_size,
        clean_rr_method='quotient-filter',
        calculate_breathing=True,
    )
    if run_data.is_valid(measurement):
        return {
            'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'heart_rate': measurement['bpm'],
            'hrv': measurement['sdnn'],
            'breathing_rate': measurement['breathingrate'] * 60,
            # 'breathing_rate': 0,
        }
    return None


# WARNING: ERRORS HERE FAIL SILENTLY - PASS debug=True in order to see errors
def estimate_heart_rate_intervals(run_data: RunData, debug=False):
    """
    Estimates heart rate intervals using the given RunData object.

    Parameters:
    -----------
    run_data : RunData
        The data structure containing sleep data, sensor readings, and runtime parameters.

    Returns:
    --------
    None
        Results are stored in `run_data.df_pred`.

    Example:
    --------
    >>> estimate_heart_rate_intervals(run_data)
    """
    if not debug and run_data.log:
        logger.warning('debug=False, errors will fail SILENTLY, pass debug=True in order to see errors')

    if run_data.log:
        print('-----------------------------------------------------------------------------------------------------')
        print(f'Estimating heart rate for {run_data.name} {run_data.start_time} -> {run_data.end_time}')

    run_data.start_timer()
    while run_data.start_interval <= run_data.end_datetime:
        measurement_1 = None
        measurement_2 = None
        try:
            measurement_1 = _calculate(run_data, run_data.side_1)
        except Exception as e:
            run_data.sensor_1_error_count += 1
            if debug:
                traceback.print_exc()

        if run_data.senor_count == 2:
            try:
                measurement_2 = _calculate(run_data, run_data.side_2)
            except Exception as e:
                run_data.sensor_2_error_count += 1

        if measurement_1 is not None and measurement_2 is not None:
            run_data.measurements_side_1.append(measurement_1)
            run_data.measurements_side_2.append(measurement_2)

            m1_heart_rate = measurement_1['heart_rate']
            m2_heart_rate = measurement_2['heart_rate']
            if run_data.hr_moving_avg is not None:
                heart_rate = (((m1_heart_rate + m2_heart_rate) / 2) + run_data.hr_moving_avg) / 2
            else:
                heart_rate = (m1_heart_rate + m2_heart_rate) / 2

            if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                if heart_rate < run_data.hr_moving_avg:
                    heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                else:
                    heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

            run_data.heart_rates.append(heart_rate)

            run_data.combined_measurements.append({
                'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
                'heart_rate': heart_rate,
                'hrv': (measurement_1['hrv'] + measurement_2['hrv']) / 2,
                'breathing_rate': (measurement_1['breathing_rate'] + measurement_2['breathing_rate']) / 2 * 60,
            })

        elif measurement_1 is not None:
            run_data.measurements_side_1.append(measurement_1)
            m1_heart_rate = measurement_1['heart_rate']

            # If the HR differs by more than the allowable movement
            if run_data.hr_moving_avg is not None and abs(m1_heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                if m1_heart_rate < run_data.hr_moving_avg:
                    m1_heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                else:
                    m1_heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

            run_data.heart_rates.append(m1_heart_rate)

            measurement_1['heart_rate'] = m1_heart_rate
            run_data.combined_measurements.append(measurement_1)

        elif measurement_2 is not None:
            run_data.sensor_1_drop_count += 1
            m2_heart_rate = measurement_2['heart_rate']

            if run_data.hr_moving_avg is not None:
                heart_rate = (m2_heart_rate + run_data.hr_moving_avg) / 2
            else:
                heart_rate = m2_heart_rate

            if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                if heart_rate < run_data.hr_moving_avg:
                    heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                else:
                    heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

            run_data.heart_rates.append(heart_rate)

            measurement_2['heart_rate'] = heart_rate
            run_data.combined_measurements.append(measurement_2)
            run_data.measurements_side_2.append(measurement_2)


        run_data.next()

    run_data.stop_timer()
    run_data.print_results()
    run_data.combine_results()
    gc.collect()

# endregion
