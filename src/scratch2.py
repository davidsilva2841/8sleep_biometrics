# scp -r -P 8822 'root@192.168.1.50:/persistent/*.RAW' /Users/david/8sleep/raw
import gc

import pandas as pd
import warnings

from datetime import datetime, timezone, timedelta
from data_manager import DataManager

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

pd.set_option('display.width', 300)

from biometrics.sleep_detection.sleep_detector import detect_sleep

data = DataManager('david', load=False)
period = data.sleep_periods[0]
for period in data.sleep_periods:
    start_time = datetime.strptime(period['start_time'], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(period['end_time'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=4)
    folder_name = start_time.strftime('%Y-%m-%d')
    folder_path = f'/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/{folder_name}/'
    detect_sleep(
        'right',
        start_time - timedelta(hours=5),
        end_time,
        folder_path,
    )





