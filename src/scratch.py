# scp -r -P 8822 'root@192.168.1.50:/persistent/*.RAW' /Users/david/8sleep/raw
import gc

import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pd.set_option('display.width', 300)

from toolkit import tools
from analyze import analyze_predictions
from vitals.calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from vitals.run_data import RuntimeParams
from data_manager import DataManager
from heartpy.exceptions import BadSignalWarning




def main():
    data = DataManager('trinity', load=True)
    period = data.sleep_periods[0]


    for name in ['alina', 'elisa', 'den']:
        data = DataManager(name, load=True)

        for period in data.sleep_periods:
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

            estimate_heart_rate_intervals(run_data, debug=False)

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



