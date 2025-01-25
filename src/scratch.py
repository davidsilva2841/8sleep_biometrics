import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.set_option('display.width', 300)

from analyze import analyze_predictions
from calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from data_manager import DataManager
from run_data import RuntimeParams




def main():
    david = DataManager('david', load=True)
    david.load_new_validation_data()
    tally = DataManager('tally', load=False)
    tally.load_new_raw_data()
    data_managers = [david]
    data = david
    david.piezo_df.tail()
    david.load_new_raw_data()
    period = data.sleep_periods[-1]
    start_time = period['start_time']
    end_time = period['end_time']
    data.piezo_df
    david = DataManager('david', load=True)
    # den = DataManager('den', load=True)
    tally = DataManager('tally', load=True)
    trinity = DataManager('trinity', load=True)


    data_managers = [david, tally, trinity]
    for data in data_managers:
        for period in data.sleep_periods:
            start_time = period['start_time']
            end_time = period['end_time']

            runtime_params: RuntimeParams = {
                'window': 6,
                'slide_by': 1,
                'moving_avg_size': 120,
                'hr_std_range': (1, 10),
                'hr_percentile': (20, 75),
                'signal_percentile': (1,99),
            }
            run_data = RunData(
                data.piezo_df,
                start_time,
                end_time,
                runtime_params=runtime_params,
                name=data.name,
                side=period['side'],
                sensor_count=1,
                label='BEST',
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
            df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])
            df_pred = (
                df_pred.groupby(pd.Grouper(key='start_time', freq='40s'))
                .agg({'heart_rate': 'mean'})
                .reset_index()
            )
            results = analyze_predictions(data, df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=True)
            run_data.print_results()


