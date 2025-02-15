import traceback
import pandas as pd
import numpy as np
from biometrics.stream.biometric_processor import BiometricProcessor
from biometrics.load_raw_files import load_raw_files
from load_raw import load_raw_data
from datetime import datetime
from analyze import analyze_predictions
from data_manager import DataManager
from toolkit import tools

data = DataManager('david', load=True)

period = data.sleep_periods[-1]
start_time = period['start_time']
end_time = period['end_time']
raw_data = load_raw_data('/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-02-15')
# raw_data = load_raw_files(
#     '/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-02-15',
#     datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"),
#     datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S"),
#     'right',
#     sensor_count=2,
#     raw_data_types=['piezo-dual']
# )

df = pd.DataFrame(raw_data['piezo_dual'])
df.sort_values(by=['ts'], inplace=True)
df = df[(df['ts'] >= start_time) & (df['ts'] <= end_time)]


processor = BiometricProcessor(side='right', sensor_count=2, insertion_frequency=1, insert_to_sql=False)



bar = tools.progress_bar(df.shape[0])
for i in range(len(df) - 2):
    bar.update()
    window = df.iloc[i:i+3]
    signal1 = np.concatenate(window['right1'].values)
    signal2 = np.concatenate(window['right2'].values)
    processor.calculate_vitals(df.iloc[i]['ts'], signal1, signal2)

df_pred = pd.DataFrame(processor.combined_measurements)

r_window_avg = 20
r_min_periods = 20
df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()
df_pred.rename({
    'period_start': 'start_time',
}, axis=1, inplace=True)

df_pred = pd.DataFrame(processor.smoothed_measurements)

results = analyze_predictions(data, df_pred, start_time, end_time, plot=True)


