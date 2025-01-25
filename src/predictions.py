import json
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


# den = DataManager('den', load=True)
# tally = DataManager('tally', load=True)
# trinity = DataManager('trinity', load=True)

david = DataManager('david', load=True)

print('-----------------------------------------------------------------------------------------------------')
print('Raw sensor data:')
print(david.piezo_df.head())

"""
ts                  type        freq  adc  gain  left1                         left2                         right1                        right2                         seq
2025-01-10 08:00:00  piezo-dual  500   1    400   [-216455, -217550, ...]       [-232650, -234649, ...]        [-1170737, -1165373, ...]      [-1011911, -1014120, ...]      1573995
"""
print('-----------------------------------------------------------------------------------------------------')
print('Sleep period:')
print(json.dumps(david.sleep_periods[-1], indent=4))


data = david
period = data.sleep_periods[-1]


runtime_params: RuntimeParams = {
    'window': 6,                    # Time window in seconds for HR estimation
    'slide_by': 1,                  # Step size for moving the window
    'moving_avg_size': 120,         # Moving average window size
    'hr_std_range': (1,10),         # Allowable HR deviation range, any measurements less or more than this are constrained to the min/max
    'hr_percentile': (20, 75),      # Percentile for HR
    'signal_percentile': (1, 99),   # Percentile for signal
}

run_data = RunData(
    data.piezo_df,
    period['start_time'],
    period['end_time'],
    runtime_params=runtime_params,
    name=data.name,
    side=period['side'],
    # sensor_count=data.sensor_count,
    sensor_count=1, # Readings tend to be more accurate when we only use sensor 1, calculations.py is not setup to use both sensors anymore
    label='',
    log=True        # Log the progress/results
)

# Results are saved to run_data.df_pred
estimate_heart_rate_intervals(run_data)
print(run_data.df_pred)

# Make a copy for tinkering with result
df_pred = run_data.df_pred.copy()
df_pred = clean_df_pred(df_pred)

r_window_avg = 20
r_min_periods = 20
df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=r_window_avg, min_periods=r_min_periods).mean()
df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=40, min_periods=10).mean()
df_pred['hrv'] = df_pred['hrv'].rolling(window=40, min_periods=10).mean()

# Calculates accuracy and plots results
results = analyze_predictions(data, df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=True)
run_data.print_results()

