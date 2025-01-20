# scp -r -P 8822 'pod2:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/david/raw/load
# scp -r -P 8822 'pod1:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/tally/raw/load
import json
from data_manager import DataManager
from calculations import clean_df_pred, estimate_heart_rate_intervals, RunData
from analyze import analyze_predictions
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import pandas as pd


def main():
    trinity = DataManager('trinity', load=True)
    trinity.update_piezo_df()
    trinity.load_new_validation_data()
    tally = DataManager('tally', load=True)
    tally.load_new_validation_data()
    tally.load_new_raw_data()
    david.load_new_raw_data()
    david = DataManager('david', load=True)
    david.load_new_validation_data()
    den = DataManager('den', load=False)
    den.load_new_validation_data()

    fp = '/Users/ds/main/8sleep_biometrics/data/people/david/validation/david_breath_rate.csv'
    df = pd.read_csv(fp)
    df.rename({
        'startDate': 'start_time',
        'endDate': 'end_time',
        'value': 'breath_rate'
    }, axis=1, inplace=True)

    df.rename({
        'startDate': 'start_time',
        'endDate': 'end_time',
        'value': 'heart_rate'
    }, axis=1, inplace=True)
