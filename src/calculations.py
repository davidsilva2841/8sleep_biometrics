import gc
from run_data import RunData
import numpy as np
# import heartpy as hp
from data_types import *
from heart.filtering import filter_signal, remove_baseline_wander
from heart.preprocessing import scale_data
from heart.heartpy import process, process_segmentwise
from cleaning import *

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


def get_hr_via_fft(signal, sample_rate, min_bpm=40, max_bpm=90):
    """
    Returns the heart rate (bpm) by searching for the largest peak in the
    frequency range [min_bpm..max_bpm].
    """
    # 1) Do an FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spectrum = np.fft.rfft(signal)
    magnitudes = np.abs(spectrum)

    # 2) Convert the BPM range to freq range
    min_freq = min_bpm / 60.0
    max_freq = max_bpm / 60.0

    # 3) Only look in [min_freq..max_freq] region
    valid_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    if len(valid_idx) == 0:
        return None

    # 4) Find the peak
    peak_idx = valid_idx[np.argmax(magnitudes[valid_idx])]
    peak_freq = freqs[peak_idx]
    # convert to BPM
    return peak_freq * 60.0


def sliding_fft_bpm(signal, sample_rate, window_size_sec=10, step_sec=5):
    window_size = int(window_size_sec * sample_rate)
    step = int(step_sec * sample_rate)
    results = []
    start = 0
    while start + window_size <= len(signal):
        chunk = signal[start:start+window_size]
        hr = get_hr_via_fft(chunk, sample_rate)
        results.append((start, start+window_size, hr))
        start += step
    return results


def _calculate(run_data: RunData, side: str):
    np_array = np.concatenate(run_data.piezo_df[run_data.start_interval:run_data.end_interval][side])

    data = interpolate_outliers_in_wave(np_array, 2)
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
        freq_method='fft',
        breathing_method='fft',
        bpmmin=40,
        bpmmax=90,
        hampel_correct=False,     # KEEP FALSE - Takes too long
        reject_segmentwise=False, # KEEP FALSE - Less accurate
        windowsize=0.50,
        clipping_scale=False,     # KEEP FALSE - Did not change reading
        clean_rr=True,            # KEEP TRUE - More accurate
        clean_rr_method='quotient-filter', # z-score is worse
    )
    if run_data.is_valid(measurement):
        return {
            'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'heart_rate': measurement['bpm'],
            'hrv': measurement['sdnn'],
            'breathing_rate': measurement['breathingrate'] * 60,
        }
    return None




def estimate_heart_rate_intervals(run_data: RunData):
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
            # traceback.print_exc()

        # if run_data.senor_count == 2:
        #     try:
        #         measurement_2 = _calculate(run_data, run_data.side_2)
        #     except Exception as e:
        #         run_data.sensor_2_error_count += 1

        # if measurement_1 is not None and measurement_2 is not None:
        #     run_data.measurements_side_1.append(measurement_1)
        #     run_data.measurements_side_2.append(measurement_2)
        #
        #     m1_heart_rate = measurement_1['heart_rate']
        #     m2_heart_rate = measurement_2['heart_rate']
        #     if run_data.hr_moving_avg is not None:
        #         heart_rate = (((m1_heart_rate + m2_heart_rate) / 2) + run_data.hr_moving_avg) / 2
        #     else:
        #         heart_rate = (m1_heart_rate + m2_heart_rate) / 2
        #
        #     if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
        #         if heart_rate < run_data.hr_moving_avg:
        #             heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
        #         else:
        #             heart_rate = run_data.hr_moving_avg + run_data.hr_std_2
        #
        #         run_data.heart_rates.append(heart_rate)
        #
        #     run_data.combined_measurements.append({
        #         'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
        #         'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
        #         'heart_rate': heart_rate,
        #         'hrv': (measurement_1['hrv'] + measurement_2['hrv']) / 2,
        #         'breathing_rate': (measurement_1['breathing_rate'] + measurement_2['breathing_rate']) / 2 * 60,
        #     })

        # elif measurement_1 is not None:
        if measurement_1 is not None:
            run_data.measurements_side_1.append(measurement_1)
            run_data.sensor_2_drop_count += 1
            m1_heart_rate = measurement_1['heart_rate']

            if run_data.hr_moving_avg is not None:
                heart_rate = (m1_heart_rate + run_data.hr_moving_avg) / 2
            else:
                heart_rate = m1_heart_rate

            if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                if heart_rate < run_data.hr_moving_avg:
                    heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                else:
                    heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

            run_data.heart_rates.append(heart_rate)

            measurement_1['heart_rate'] = heart_rate
            run_data.combined_measurements.append(measurement_1)

        # elif measurement_2 is not None:
        #     run_data.measurements_side_2.append(measurement_2)
        #     run_data.sensor_1_drop_count += 1
        #     m2_heart_rate = measurement_2['heart_rate']
        #
        #     if run_data.hr_moving_avg is not None:
        #         heart_rate = (m2_heart_rate + run_data.hr_moving_avg) / 2
        #     else:
        #         heart_rate = m2_heart_rate
        #
        #     if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
        #         if heart_rate < run_data.hr_moving_avg:
        #             heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
        #         else:
        #             heart_rate = run_data.hr_moving_avg + run_data.hr_std_2
        #
        #         run_data.heart_rates.append(heart_rate)
        #
        #     measurement_2['heart_rate'] = heart_rate
        #     run_data.combined_measurements.append(measurement_2)

        run_data.next()


    run_data.stop_timer()
    run_data.print_results()
    run_data.combine_results()
    gc.collect()




# endregion
