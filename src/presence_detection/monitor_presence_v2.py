# python3 analyze_sleep.py --side=right --start_time="2025-01-31 23:00:00" --end_time="2025-02-01 15:00:00"
# python3 analyze_sleep.py --side=left --start_time="2025-01-31 23:00:00" --end_time="2025-02-01 15:30:00"
# python3 analyze_sleep.py --side=left --start_time="2025-02-01 14:00:00" --end_time="2025-02-01 14:01:00"
import json
import gc
import pandas as pd
import os
import sys
import argparse
import traceback
from datetime import datetime, timezone, timedelta
from db import *
from presence_detection.cap_data import save_baseline

sys.path.append(os.getcwd())
from load_raw_file import load_raw_files
from piezo_data import load_piezo_df, detect_presence_piezo, identify_baseline_period
from cap_data import *
from sleep_detector import *
from logger import get_logger
from presence_types import *
from plot_presence import *

logger = get_logger()

#
# def main(folder_path: str, start_time: datetime, end_time: datetime, side: Side, clean: bool=True):
def main():
    # folder_path = '/Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-29/'
    # folder_path = '/Users/ds/main/8sleep_biometrics/data/recent/'
    folder_path = '/Users/ds/main/8sleep_biometrics/data/people/trinity/raw/loaded/2025-01-20/'
    start_time = datetime.strptime('2025-01-20 00:00:00', "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_time =   datetime.strptime('2025-01-20 07:00:00', "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    side: Side = 'left'
    clean = False
    create_db_and_table()
    data = load_raw_files(folder_path, start_time - timedelta(hours=12), end_time + timedelta(hours=2), side)


    piezo_df = load_piezo_df(data, side, lower_percentile=2, upper_percentile=98)
    detect_presence_piezo(
        piezo_df,
        side,
        rolling_seconds=10,
        threshold_percent=0.70,
        range_threshold=100_000,
        range_rolling_seconds=10,
        clean=clean
    )
    cap_df = load_cap_df(data, side)

    # Cleanup data
    # del data
    # gc.collect()

    merged_df = piezo_df.join(cap_df, how='inner')
    merged_df.drop_duplicates(inplace=True)
    # Free up memory from old dfs
    piezo_df.drop(piezo_df.index, inplace=True)
    cap_df.drop(cap_df.index, inplace=True)
    # del piezo_df
    # del cap_df
    # gc.collect()
    piezo_df.index.is_unique
    cap_df.index.is_unique
    merged_df.index.is_unique
    # cap_baseline = load_baseline()

    baseline_start_time, baseline_end_time = identify_baseline_period(merged_df, side, threshold_range=10_000, empty_minutes=10)
    cap_baseline = create_cap_baseline_from_cap_df(merged_df, baseline_start_time, baseline_end_time, side, min_std=5)

    # save_baseline(side, cap_baseline)
    # cap_baseline = load_baseline(side)

    detect_presence_cap(
        merged_df,
        cap_baseline,
        side,
        occupancy_threshold=5,
        rolling_seconds=10,
        threshold_percent=0.90,
        clean=clean
    )

    merged_df[f'final_{side}_occupied'] = merged_df[f'piezo_{side}1_presence'] + merged_df[f'cap_{side}_occupied']
    sleep_records = build_sleep_records(merged_df, side, max_gap_in_minutes=15)
    print(json.dumps(sleep_records, default=custom_serializer, indent=4))
    plot_occupancy_one_side(merged_df, side)

    plot_df_column(merged_df,['piezo_right1_presence', 'cap_right_occupied', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'])
    plot_df_column(merged_df,['piezo_right1_presence', 'cap_right_occupied', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='00:00:07', end_time='03:58:07')
    plot_df_column(merged_df,['piezo_left1_presence', 'cap_left_occupied', 'left1_avg', 'left1_range', 'left_out', 'left_cen', 'left_in'], start_time='00:00:07', end_time='03:58:07')
    plot_df_column(merged_df,['piezo_right1_presence', 'cap_right_occupied', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='07:34:15', end_time='07:37:45')
    plot_df_column(merged_df,['piezo_right1_presence', 'cap_right_occupied', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='06:30:15', end_time='06:45:45')
    merged_df[pd.to_datetime("2025-02-05 07:35:00"):pd.to_datetime("2025-02-05 07:35:30")]


    merged_df[pd.to_datetime("2025-02-03 06:30:00"):pd.to_datetime("2025-02-03 06:45:00")].head(50)
    piezo_df[pd.to_datetime("2025-02-03 06:30:00"):pd.to_datetime("2025-02-03 06:45:00")][['right1_range', 'piezo_right1_presence']].head(50)
    merged_df[pd.to_datetime("2025-02-03 06:30:00"):pd.to_datetime("2025-02-03 06:45:00")]['piezo_right1_presence']
    merged_df[pd.to_datetime("2025-02-03 06:30:00"):pd.to_datetime("2025-02-03 06:45:00")]['right1_range'].min()
    plot_df_column(merged_df,['piezo_right1_presence', 'cap_right_occupied', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'])





    insert_sleep_records(sleep_records)
    plot_df_column(merged_df, ['final_left_occupied', 'cap_left_occupied', 'piezo_left1_presence', 'left1_avg', 'left1_range', 'left_out', 'left_cen',
                               'left_in'], start_time='11:10', end_time='11:25')
    plot_df_column(merged_df,['final_right_occupied', 'cap_right_occupied', 'piezo_right1_presence', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='00:00', end_time='23:59')


    plot_df_column(merged_df,['piezo_right1_presence', 'cap_right_occupied', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='05:00:15', end_time='07:00:45')
    plot_df_column(merged_df,['piezo_right1_presence', 'cap_right_occupied', 'right1_avg', 'right1_range', 'right_out', 'right_cen', 'right_in'], start_time='05:40:15', end_time='05:50:45')

    return merged_df, sleep_records




