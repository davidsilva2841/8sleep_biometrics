# scp -r -P 8822 'pod2:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/david/raw/load
# scp -r -P 8822 'pod1:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/tally/raw/load

from src.data_manager import DataManager
from src.calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from src.analyze import analyze_predictions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd


import heartpy as hp

def main():
    tally = DataManager('tally', load=True)
    david = DataManager('david', load=True)
    df = tally.piezo_df
    val = df.iloc[0]['right1']
    df['Age'] = df['Age'].apply(lambda x: x + 5)

    data = hp.filter_signal(val, [0.05, 15], 500, filtertype='bandpass')
    data = tally
    tally.update_piezo_df()
    tally.piezo_df.tail()
    period = data.sleep_periods[-1]
    for period in data.sleep_periods:
    # for period in data.sleep_periods:
        start_time = period['start_time']
        end_time = period['end_time']

        run_data = RunData(
            data.piezo_df,
            start_time,
            end_time,
            slide_by=1,
            window=10,
            hr_std_range=(1,12),
            percentile=(1,99),
            moving_avg_size=60,
            name=data.name,
            side=period['side']
        )

        estimate_heart_rate_intervals(run_data)


        df_pred = run_data.df_pred.copy()
        df_pred = clean_df_pred(df_pred)

        r_window_avg = 25
        r_min_periods = 25

        df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()

        df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=40, min_periods=10).mean()
        df_pred['hrv'] = df_pred['hrv'].rolling(window=40, min_periods=10).mean()

        run_data.chart_info['r_window_avg'] = r_window_avg
        run_data.chart_info['r_window_avg'] = r_window_avg
        run_data.chart_info['label'] = 'TEST'

        run_data.print_results()
        analyze_predictions(data, df_pred, start_time, end_time, run_data.chart_info)


