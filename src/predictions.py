# scp -r -P 8822 'pod2:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/david/raw/load
import json
from data_manager import DataManager
from calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from analyze import analyze_predictions
import warnings

from run_data import RuntimeParams
import pandas as pd

pd.set_option('display.width', 300)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# scp /Users/ds/main/8sleep_biometrics/data/people/tally/raw/tally_piezo_df.feather ds@192.168.50.167:/home/ds/main/8sleep_biometrics/data/people/tally/raw
# ssh-copy-id -i ~/.ssh/id_rsa.pub ds@192.168.50.167




def main():

    den = DataManager('den')
    den.heart_rate_df
    david = DataManager('david')
    david.load_new_raw_data()
    david.load_new_validation_data()
    tally = DataManager('tally')
    trinity.load_new_raw_data()
    data_managers = [trinity, tally, david]
    trinity = DataManager('trinity')
    data = tally
    period = data.sleep_periods[-1]
    for data in data_managers:
        for period in data.sleep_periods:
            start_time = period['start_time']
            end_time = period['end_time']

            runtime_params: RuntimeParams = {
                'window': 4,
                'slide_by': 1,
                'moving_avg_size': 120,
                'hr_std_range': (1,10),
                'percentile': (17.5, 82.5),
            }
            run_data = RunData(
                data.piezo_df,
                start_time,
                end_time,
                runtime_params=runtime_params,
                name=data.name,
                side=period['side'],
                sensor_count=1,
                label='TALLY_NEW',
                log=True
            )

            estimate_heart_rate_intervals(run_data)

            df_pred = run_data.df_pred.copy()
            df_pred = clean_df_pred(df_pred)

            r_window_avg = 10
            r_min_periods = 5

            df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()

            df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=40, min_periods=10).mean()
            df_pred['hrv'] = df_pred['hrv'].rolling(window=40, min_periods=10).mean()

            run_data.print_results()
            results = analyze_predictions(data, df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=True)


        df = pd.read_pickle('/Users/ds/main/8sleep_biometrics/data/people/den/raw/load/2025-01-20.pkl.zip')

# den = DataManager('den')
# df = den.heart_rate_df.copy()
# # df['start_time'] = pd.to_datetime(df['start_time'])
# #
# # # Shift 'start_time' by 1 hour
# # df['start_time'] = df['start_time'] + pd.to_timedelta(1, unit='h')
# #
# #
# # df.head()
# lower_bound = df['heart_rate_actual'].quantile(0.00)
# upper_bound = df['heart_rate_actual'].quantile(0.96)
# # Filter the DataFrame to remove outliers
# df_filtered = df[(df['heart_rate_actual'] >= lower_bound) & (df['heart_rate_actual'] <= upper_bound)]
#
# # Resample data to 10-second intervals without setting index
# df_resampled = (
#     df_filtered.groupby(pd.Grouper(key='start_time', freq='120S'))
#     .agg({'heart_rate_actual': 'mean'})
#     .reset_index()
# )
# df_resampled.dropna(inplace=True)
# df_resampled.to_csv(den.heart_rate_file_path, index=False)
# data.heart_rate_df = df_resampled
#
