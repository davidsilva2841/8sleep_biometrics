import gc
import os.path

import pandas as pd
import subprocess

from presence_detection.presence_types import SleepRecord
from run_data import ChartInfo
from toolkit import tools
from src.data_manager import *
from src.analyze import analyze_predictions




def run_sleep_decoder(sleep_period: TimePeriod, name: str, file_name: str):

    command = [
        "cargo", "run", "--bin", "sleep-decoder", "--release",
        f"/Users/ds/main/8sleep_biometrics/data/people/{name}/raw/{name}_piezo_df.feather",
        f"--start-time", sleep_period['start_time'][:-3],
        f"--end-time", sleep_period['end_time'][:-3],
        "--csv-output", f"/Users/ds/main/8sleep_biometrics/predictions/{file_name}/",
        "--hr-window-seconds", "10.0",
        "--hr-window-overlap", "0.67",
        "--br-window-seconds=120.0",
        "--br-window-overlap=0.0",
        "--harmonic-penalty-close=0.7",
        "--harmonic-penalty-far=0.3",
        "--hr-smoothing-window=75",
        "--hr-smoothing-strength=0.25",
        "--hr-history-window=180",
        "--hr-outlier-percentile=0.0075",
        "--feather-input"
    ]

    print(' '.join(command))
    print(*command, sep='\n')
    print('-----------------------------------------------------------------------------------------------------')

    result = subprocess.run(
        command,
        cwd="/Users/ds/main/sleep-decoder/",
        capture_output=True,
        text=True
    )

    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

# if __name__ == "__main__":
#     run_sleep_decoder()


def main():
    for name in ['alina', 'david', 'den', 'elisa', 'tally', 'trinity']:
        data = DataManager(name, load=True)
        for period in data.sleep_periods:
            print('-----------------------------------------------------------------------------------------------------')

            print(f'{name} {period}')
            start_time = period['start_time']
            end_time = period['end_time']
            side = period['side']
            file_name = f'{name}_{period["start_time"][:10]}_'
            run_sleep_decoder(period, name, file_name)

            df_pred_file_path = f'/Users/ds/main/8sleep_biometrics/predictions/{file_name}_{side}_combined_period_0.csv'
            df_pred = pd.read_csv(df_pred_file_path)

            df_pred.rename({
                'fft_hr_smoothed': 'heart_rate',
                'timestamp': 'start_time',
            }, axis=1, inplace=True)

            chart_info: ChartInfo = {
                'labels': {
                    'name': name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': 'overlord'
                }
            }
            results = analyze_predictions(data, df_pred, start_time, end_time, chart_info, plot=True)
            tools.write_json_to_file(f'/Users/ds/main/8sleep_biometrics/predictions/results/{file_name}.json', results)
        del data
        gc.collect()



    files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/predictions/results/', full_path=True)
    results = [{**tools.read_json_from_file(file)['heart_rate']['accuracy'], 'file': os.path.basename(file)} for file in files]
    overlord_df = pd.DataFrame(results)
    overlord_df['corr'] = overlord_df['corr'].str.rstrip('%').astype(float) / 100
    overlord_df['rmse'].mean()
    overlord_df['corr'].mean()
    overlord_df.describe()
    files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/predictions/davids/', full_path=True)
    results = [{**tools.read_json_from_file(file)['heart_rate']['accuracy'], 'file': os.path.basename(file)} for file in files]
    davids_df = pd.DataFrame(results)
    davids_df['corr'] = davids_df['corr'].str.rstrip('%').astype(float) / 100
    davids_df['corr'].mean()
    davids_df['rmse'].mean()
    davids_df.describe()

