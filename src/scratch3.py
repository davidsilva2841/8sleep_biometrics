# scp -r -P 8822 'root@192.168.1.50:/persistent/*.RAW' /Users/david/8sleep/raw
import gc

import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pd.set_option('display.width', 300)

from analyze import analyze_predictions
from vitals.calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from vitals.run_data import RuntimeParams
from data_manager import DataManager


if __name__ == "__main__":
    data = DataManager('trinity', load=True)
    period = data.sleep_periods[0]

    start_time = period['start_time']
    end_time = period['end_time']

    runtime_params: RuntimeParams = {
        'window': 10,
        'slide_by': 1,
        'moving_avg_size': 120,
        'hr_std_range': (1, 20),
        'hr_percentile': (20, 75),
        'signal_percentile': (0.5, 99.5),
    }
    run_data = RunData(
        data.piezo_df,
        start_time,
        end_time,
        runtime_params=runtime_params,
        name=data.name,
        side=period['side'],
        sensor_count=data.sensor_count,
        label='COMPARE_NUMPY',
        log=True
    )

    estimate_heart_rate_intervals(run_data, debug=True)

    df_pred = run_data.df_pred.copy()

    df_pred = clean_df_pred(df_pred)
    r_window_avg = 15
    r_min_periods = 5

    df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()

    df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=40, min_periods=10).mean()
    df_pred['hrv'] = df_pred['hrv'].rolling(window=40, min_periods=10).mean()
    # df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])
    # df_pred = (
    #     df_pred.groupby(pd.Grouper(key='start_time', freq='40s'))
    #     .agg({'heart_rate': 'mean'})
    #     .reset_index()
    # )
    file_name = f'{data.name}_{period["start_time"][:10]}_'

    results = analyze_predictions(data, df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=True)
    print('-----------------------------------------------------------------------------------------------------')
    print(results)
    print('-----------------------------------------------------------------------------------------------------')

