# scp -r -P 8822 'pod2:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/david/raw/load
# scp -r -P 8822 'pod1:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/tally/raw/load
import json
from data_manager import DataManager
from calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from analyze import analyze_predictions
import warnings

from run_data import RuntimeParams

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import pandas as pd

def main():
    tally = DataManager('tally', load=True)
    david = DataManager('david', load=True)
    david.load_new_raw_data()
    david.load_new_validation_data()

    trinity = DataManager('trinity', load=True)
    david = DataManager('david', load=True)
    data = david
    period = data.sleep_periods[-1]
    for period in data.sleep_periods:
        start_time = period['start_time']
        end_time = period['end_time']

        runtime_params: RuntimeParams = {
            'window': 10,
            'slide_by': 1,
            'moving_avg_size': 60,
            'hr_std_range': (1,6),
            'percentile': (15, 85),
        }
        run_data = RunData(
            data.piezo_df,
            start_time,
            end_time,
            runtime_params=runtime_params,
            name=data.name,
            side=period['side'],
            sensor_count=data.sensor_count,
            log=True
        )

        estimate_heart_rate_intervals(run_data)

        df_pred = run_data.df_pred.copy()
        df_pred = clean_df_pred(df_pred)

        r_window_avg = 20
        r_min_periods = 20

        df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()

        df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=40, min_periods=10).mean()
        df_pred['hrv'] = df_pred['hrv'].rolling(window=40, min_periods=10).mean()

        # run_data.chart_info['r_window_avg'] = r_window_avg
        # run_data.chart_info['r_min_periods'] = r_window_avg
        # run_data.chart_info['label'] = 'new'

        run_data.print_results()
        results = analyze_predictions(data, df_pred, run_data, plot=True)


        df = pd.read_pickle('/Users/ds/main/8sleep_biometrics/data/people/den/raw/load/2025-01-20.pkl.zip')
