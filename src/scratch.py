# scp -r -P 8822 'root@192.168.1.50:/persistent/*.RAW' /Users/david/8sleep/raw
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
    alina = DataManager('alina')
    alina._update_piezo_df()
    alina.load_new_validation_data()
    elisa.load_new_raw_data()
    den = DataManager('den')
    alina.load_new_raw_data()
    alina.load_new_validation_data()
    tally = DataManager('tally', load=False)
    tally.load_new_raw_data()
    david._update_piezo_df()
    david.load_new_validation_data()
    tally = DataManager('tally', load=False)
    tally.load_new_raw_data()
    tally.load_new_validation_data()
    data_managers = [david]
    data = david
    david.piezo_df.tail()
    david.load_new_raw_data()
    david.load_new_validation_data()
    period = data.sleep_periods[-1]
    start_time = period['start_time']
    end_time = period['end_time']
    tally = DataManager('tally', load=True)
    tally.load_new_raw_data()
    david = DataManager('david', load=True)
    trinity = DataManager('trinity', load=True)
    den.load_new_raw_data()
    data_managers = [trinity, david]

    elisa = DataManager('elisa', load=True)
    data = alina
    period = data.sleep_periods[-1]

    # fft_vals = np.fft.rfft(data.piezo_df.iloc[0]['right1'])
    # mags = np.abs(fft_vals)
    # freq_resolution = 500 / 500
    # freqs = np.fft.rfftfreq(fft_vals, d=1.0 / 500)
    for data in data_managers:
        for period in data.sleep_periods:
            start_time = period['start_time']
            end_time = period['end_time']

            runtime_params: RuntimeParams = {
                'window': 10,
                'slide_by': 1,
                'moving_avg_size': 120,
                'hr_std_range': (1, 20),
                'hr_percentile': (25, 75),
                'signal_percentile': (1, 99),
            }
            run_data = RunData(
                data.piezo_df,
                start_time,
                end_time,
                runtime_params=runtime_params,
                name=data.name,
                side=period['side'],
                sensor_count=data.sensor_count,
                label='NEW_BEST',
                log=True
            )

            estimate_heart_rate_intervals(run_data)

            df_pred = run_data.df_pred.copy()
            df_pred = run_data.df_pred_side_1.copy()
            df_pred = run_data.df_pred_side_2.copy()

            df_pred = clean_df_pred(df_pred)
            r_window_avg = 30
            r_min_periods = 30

            df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()

            df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=40, min_periods=10).mean()
            df_pred['hrv'] = df_pred['hrv'].rolling(window=40, min_periods=10).mean()
            # df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])
            # df_pred = (
            #     df_pred.groupby(pd.Grouper(key='start_time', freq='40s'))
            #     .agg({'heart_rate': 'mean'})
            #     .reset_index()
            # )
            results = analyze_predictions(data, df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=True)
            run_data.print_results()


