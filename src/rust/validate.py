# ls -1 . | wc -l
import gc
import random
from time import daylight
from typing import Dict, Any

import numpy as np
import multiprocessing
import json
import itertools
from calculations import estimate_heart_rate_intervals, RunData
from analyze import analyze_predictions
import tools
import hashlib
from test.globals import data_managers
import traceback
import subprocess
from config import PROJECT_FOLDER_PATH
import warnings
import textwrap
import shlex

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# SLEEP_DECODER_PATH = '/Users/ds/main/sleep-decoder'
SLEEP_DECODER_PATH = '/home/ds/main/sleep-decoder'


def build_command(params: Dict[str, Any], feather_path: str, output_path: str, start_time: str, end_time: str) -> str:
    cmd = [
        SLEEP_DECODER_PATH,
        feather_path,
        f"--csv-output={output_path}",
        f"--start-time=\"{start_time}\"",
        f"--end-time=\"{end_time}\"",
        f"--hr-window-seconds={params['hr_window_seconds']}",
        f"--hr-window-overlap={params['hr_window_overlap']}",
        f"--br-window-seconds={params['br_window_seconds']}",
        f"--br-window-overlap={params['br_window_overlap']}",
        f"--harmonic-penalty-close={params['harmonic_penalty_close']}",
        f"--harmonic-penalty-far={params['harmonic_penalty_far']}",
        f"--hr-smoothing-window={params['hr_smoothing_window']}",
        f"--hr-smoothing-strength={params['hr_smoothing_strength']}",
        f"--hr-history-window={params['hr_history_window']}",
        f"--hr-outlier-percentile={params['hr_outlier_percentile']}"
        f"--feather-input"
    ]
    return " ".join(str(x) for x in cmd)


def hash_dict(result):
    # Convert dictionary to a JSON string with sorted keys for consistency
    json_str = json.dumps(result, sort_keys=True)

    # Generate SHA-128 hash (SHA-256 recommended if security is a concern)
    hash_obj = hashlib.sha1(json_str.encode())  # SHA-1 (160-bit) or change to sha256
    return hash_obj.hexdigest()


def _clean_time(time: str):
    return time.replace(':', '.').replace(' ', '_')


def test_params():
    data = data_managers[0]
    period = data.sleep_periods[0]
    for data in data_managers:
        for period in data.sleep_periods:
            start_time = period['start_time'][:-3]
            end_time = period['end_time'][:-3]
            folder_name = f'{data.name}___{_clean_time(start_time)}___{_clean_time(end_time)}'
            output_path = f'{PROJECT_FOLDER_PATH}tmp/rust/{folder_name}/a'
            tools.create_folder(output_path)
            params = {
                "hr_window_seconds": 10.0,
                "hr_window_overlap": 0.67,
                "br_window_seconds": 120.0,
                "br_window_overlap": 0.0,
                "harmonic_penalty_close": 0.7,
                "harmonic_penalty_far": 0.3,
                "hr_smoothing_window": 60,
                "hr_smoothing_strength": 0.25,
                "hr_history_window": 180,
                "hr_outlier_percentile": 0.01
            }
            command = build_command(params, data.piezo_df_file_path, output_path, start_time, end_time)

            result = subprocess.run(
                command,
                shell=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )

