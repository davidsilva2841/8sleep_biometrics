import gc
from run_data import RunData

# import heartpy as hp
from data_types import *
from heart.filtering import filter_signal
from heart.preprocessing import scale_data
from heart.heartpy import process

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
    np_array = np.concatenate(run_data.piezo_df[run_data.start_interval:run_data.end_interval][side])

    data = filter_signal(np_array, [0.05, 15], 500, filtertype='bandpass')
    data = scale_data(data)
    working_data, measurement = process(
        data,
        500,
        freq_method='fft',
        breathing_method='fft',
        bpmmin=40,
        bpmmax=90,
        hampel_correct=False,     # KEEP FALSE - Takes too long
        reject_segmentwise=False, # KEEP FALSE - Less accurate
        windowsize=0.5,
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

        run_data.next()


    run_data.stop_timer()
    run_data.print_results()
    run_data.combine_results()
    gc.collect()




# endregion
