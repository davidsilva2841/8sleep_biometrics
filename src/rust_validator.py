import pandas as pd

from src.data_manager import DataManager
from src.analyze import analyze_predictions


# Period 1
start_time = '2025-01-12 08:00'
end_time = '2025-01-12 14:00'

# Period 2
start_time = '2025-01-12 08:00'
end_time = '2025-01-12 15:00'

# Period 3 - tossed around a lot this night
start_time = '2025-01-13 07:30'
end_time = '2025-01-13 013:25'

# Period 4
start_time = '2025-01-14 06:30'
end_time = '2025-01-14 14:00'

tally =
"""
HR_WINDOW_SECONDS=10
HR_WINDOW_OVERLAP_PERCENT=0.1
BR_WINDOW_SECONDS=120
BR_WINDOW_OVERLAP_PERCENT=0

cargo run --release /Users/ds/main/8sleep_biometrics/data/people/tally/raw/loaded/2025-01-19 \
  --csv-output=/Users/ds/main/8sleep_biometrics/raw/predictions/2025-01-19/tally/ \
  --split-sensors=true

"""

# SEGMENT_WIDTH_SECONDS=10 SEGMENT_OVERLAP_PERCENT=0.1 CSV_OUTPUT=/Users/ds/main/Google_Drive/Projects/testing/8sleep/raw cargo run --release /Users/ds/main/Google_Drive/Projects/testing/8sleep/raw/accurate/data

# Period 1
start_time = '2025-01-10 08:00'
end_time = '2025-01-10 14:00'

# Period 2
start_time = '2025-01-12 08:00'
end_time = '2025-01-12 15:00'

# Period 3 - tossed around a lot this night
start_time = '2025-01-13 07:30'
end_time = '2025-01-13 13:25'

# Period 4
start_time = '2025-01-14 06:30'
end_time = '2025-01-14 14:00'

start_time = '2025-01-15 06:30'
end_time = '2025-01-15 14:00'
start_time = '2025-01-16 07:00'
end_time = '2025-01-16 14:40'



def main():

    tally = DataManager('tally')
    data = tally
    period = data.sleep_periods[-1]
    start_time = period['start_time']
    end_time = period['end_time']

    file_path = '/Users/ds/main/8sleep_biometrics/raw/predictions/2025-01-19/tally_right_combined_period_0.csv'
    df_pred = pd.read_csv(file_path)

    df_pred['end_time'] = df_pred['timestamp']
    df_pred['source'] = 'overlord'
    df_pred.rename({
        'fft_hr_smoothed': 'heart_rate',
        'timestamp': 'start_time',
    }, axis=1, inplace=True)

    start_time = '2025-01-17 07:00'
    end_time = '2025-01-17 12:45'

    info = {
        'source': 'overlord',
        'start_time': start_time,
        'end_time': end_time,
    }

    data = DataManager('david')
    analyze_predictions(data, df_pred, start_time, end_time, info)


