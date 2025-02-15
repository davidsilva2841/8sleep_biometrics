import gc

import pandas as pd
import warnings
import requests
from scipy.signal import resample
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pd.set_option('display.width', 300)

pd.options.compute.use_bottleneck = True  # Uses Bottleneck for faster aggregation
pd.options.compute.use_numexpr = True    # Uses Numexpr for faster arithmetic

from toolkit import tools
from analyze import analyze_predictions
from vitals.calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from vitals.run_data import RuntimeParams

from data_manager import DataManager

def main():
    data = DataManager('david', load=True)
    data._update_piezo_df()
    period = data.sleep_periods[-1]
    start_time = period['start_time']
    end_time = period['end_time']

    resp = requests.get('http://192.168.50.231:3000/api/metrics/vitals?side=right')
    resp.json()
    df_pred = pd.DataFrame(resp.json())

    df_pred.dropna(subset=['heart_rate'], inplace=True)
    df_pred['start_time'] = df_pred['period_start']
    data.heart_rate_df.dtypes
    df_pred["start_time"] = pd.to_datetime(df_pred["start_time"], utc=True)
    # Convert to naive datetime (remove timezone)
    df_pred["start_time"] = df_pred["start_time"].dt.tz_localize(None)
    df_pred = df_pred[df_pred['start_time'] > '2025-02-15']
    df_pred = df_pred[df_pred['start_time'] > start_time]
    df_pred.sort_values(by='start_time', inplace=True)
    results = analyze_predictions(data, df_pred, start_time, end_time, chart_info={'labels': {'start_time': start_time, 'end_time': end_time}}, plot=True)
    df = data.piezo_df[data.piezo_df.index  > '2025-02-14']
    data.piezo_df.tail()
    data.load_new_raw_data()



    # for name in ['alina', 'den', 'elisa', 'trinity', 'tally', 'david']:
    for name in ['trinity']:
        data = DataManager(name, load=True)

        for period in data.sleep_periods:
            start_time = period['start_time']
            end_time = period['end_time']

            runtime_params: RuntimeParams = {
                'window': 3,
                'slide_by': 1,
                'moving_avg_size': 100,
                'hr_std_range': (1, 10),
                'hr_percentile': (20, 75),
                'signal_percentile': (0.2, 99.8),
                'window_size': 0.65,
            }
            run_data = RunData(
                data.piezo_df,
                start_time,
                end_time,
                runtime_params=runtime_params,
                name=data.name,
                side=period['side'],
                sensor_count=data.sensor_count,
                label='BEST_02_11',
                log=True
            )

            estimate_heart_rate_intervals(run_data, debug=False)

            df_pred = run_data.df_pred.copy()

            df_pred = clean_df_pred(df_pred)
            r_window_avg = 20
            r_min_periods = 20
            df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()

            df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=40, min_periods=10).mean()
            df_pred['hrv'] = df_pred['hrv'].rolling(window=40, min_periods=10).mean()
            df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])
            # df_pred = (
            #     df_pred.groupby(pd.Grouper(key='start_time', freq='40s'))
            #     .agg({'heart_rate': 'mean'})
            #     .reset_index()
            # )

            results = analyze_predictions(data, df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=True)

            file_name = f'{data.name}_{period["start_time"][:10]}_'
            tools.write_json_to_file(f'/Users/ds/main/8sleep_biometrics/predictions/davids/{file_name}.json', results)
            run_data.print_results()
            del run_data
            gc.collect()
            # ---------------------------------------------------------------------------------------------------

            df = run_data.df_pred.copy()
            df['start_time'] = pd.to_datetime(df['start_time'])

            # Floor timestamps to the nearest 5-minute interval
            df['period_start'] = df['start_time'].dt.floor('5min')
            df['period_end'] = df['period_start'] + pd.Timedelta(minutes=5)

            # Convert timestamps to Unix epoch (for Prisma)
            df['period_start'] = df['period_start'].astype('int64') // 10 ** 9
            df['period_end'] = df['period_end'].astype('int64') // 10 ** 9

            # Group by 5-minute periods and compute averages
            df_aggregated = df.groupby(['period_start', 'period_end']).agg({
                'heart_rate': 'mean',
                'hrv': 'mean',
                'breathing_rate': 'mean'
            }).reset_index()
            df_aggregated['ts_start'] = pd.to_datetime(df_aggregated['period_start'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            df_aggregated['ts_end'] = pd.to_datetime(df_aggregated['period_end'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')


        del data
        gc.collect()



