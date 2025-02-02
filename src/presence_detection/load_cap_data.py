import os
import json
import math
import pandas as pd

from presence_types import *
from logger import get_logger

logger = get_logger()
CAP_BASE_LINE_FILE_PATH = './cap_baseline.json'


def _create_cap_baseline_from_cap_df(merged_df: pd.DataFrame, side: Side, min_std: int = 5) -> CapBaseline:
    print(f'Creating baseline for capacitance sensors...')
    baseline_stats = {}
    for sensor in [f'{side}_out', f'{side}_cen', f'{side}_in']:
        baseline_stats[sensor] = {
            "mean": merged_df[sensor].mean(),
            "std": max(merged_df[sensor].std(), min_std)
        }

    print(json.dumps(baseline_stats, indent=4))
    return baseline_stats


def load_baseline():
    if os.path.isfile(CAP_BASE_LINE_FILE_PATH):
        with open(CAP_BASE_LINE_FILE_PATH, 'r') as baseline:
            return json.load(baseline)


def load_cap_df(data: Data, side: Side) -> pd.DataFrame:
    logger.debug('Loading cap df...')
    df = pd.DataFrame(data['cap_senses'], columns=['ts', side])

    df[f'{side}_out'] = df[side].str['out']
    df[f'{side}_cen'] = df[side].str['cen']
    df[f'{side}_in'] = df[side].str['in']

    df.drop(columns=[side], inplace=True)

    # Sort, parse, set index in one pass
    df.sort_values('ts', inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
    df.set_index('ts', inplace=True)
    return df


def detect_presence_cap(
        merged_df: pd.DataFrame,
        cap_baseline,
        side: Side,
        occupancy_threshold: int = 50,
        rolling_seconds=120,
        threshold_percent=0.75,
        clean=True
) -> pd.DataFrame:
    logger.debug('Detecting cap presence...')
    # Vectorized sensor deltas (removes the need for _sensor_delta row-wise function):
    merged_df[f'{side}_combined'] = (
            (merged_df[f'{side}_out'] - cap_baseline[f'{side}_out']['mean']) / cap_baseline[f'{side}_out']['std']
            + (merged_df[f'{side}_cen'] - cap_baseline[f'{side}_cen']['mean']) / cap_baseline[f'{side}_cen']['std']
            + (merged_df[f'{side}_in'] - cap_baseline[f'{side}_in']['mean']) / cap_baseline[f'{side}_in']['std']
    )

    merged_df[f'cap_{side}_occupied'] = (merged_df[f'{side}_combined'] > occupancy_threshold).astype(int)

    threshold_count = math.ceil(threshold_percent * rolling_seconds)

    # Rolling presence detection
    merged_df[f'cap_{side}_occupied'] = (
            merged_df[f'cap_{side}_occupied']
            .rolling(rolling_seconds, min_periods=1)
            .sum()
            >= threshold_count
    ).astype(int)

    if clean:
        merged_df.drop(columns=[f'{side}_combined', f'{side}_out', f'{side}_cen', f'{side}_in'], inplace=True)

    return merged_df
