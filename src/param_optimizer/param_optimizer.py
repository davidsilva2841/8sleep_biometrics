"""
This script performs parallelized hyperparameter optimization to find the best combination of parameters
for identifying heart rate (HR) from 8 Sleep device data.

## Overview
- The script uses multiprocessing to distribute parameter combinations across multiple processes.
- Heart rate predictions are estimated for different time periods and analyzed for accuracy.
- Results are saved as JSON files for further analysis.

## Key Features
1. **Parameter Combination Generation:**
   - Generates all possible parameter combinations for HR analysis.
   - Uses itertools to create parameter combinations based on defined ranges.

2. **Parallel Processing:**
   - Leverages multiprocessing to distribute the workload across multiple CPU cores.
   - `parallel_predictions()` function splits parameter sets and processes them concurrently.

3. **Heart Rate Estimation Workflow:**
   - Runs heart rate interval estimations using the `estimate_heart_rate_intervals()` function.
   - Applies rolling average smoothing to predictions.
   - Evaluates results using the `analyze_predictions()` function.

4. **Result Storage and Hashing:**
   - Results are hashed using SHA-1 for consistency and uniqueness.
   - Each result is saved as a JSON file in the specified project folder.

## Usage
To run the script, execute it as the main module:
    python src/param_optimizer/param_optimizer.py.py


## Configuration
The script defines parameter ranges within the `param_grid` dictionary. Users can modify this to experiment
with different parameter values:

    param_grid = {
        "slide_by": [1],
        "window": [6],
        "hr_std_range": [(1,8), (1,10), (1,12)],
        "percentile": [(20, 75), (15,80)],
        "moving_avg_size": [120, 130, 140]
    }

## Output
- Processed results are saved as JSON files in the `src/test/results/` directory.
- Printed logs provide feedback on progress and potential errors.

## Error Handling
- Any exceptions during processing are caught and logged using traceback for debugging.

"""


import gc
import hashlib
import itertools
import json
import multiprocessing
import numpy as np
import random
import tools
import traceback
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from analyze import analyze_predictions
from calculations import estimate_heart_rate_intervals, RunData
from config import PROJECT_FOLDER_PATH
from globals import data_managers




def hash_dict(result):
    # Convert dictionary to a JSON string with sorted keys for consistency
    json_str = json.dumps(result, sort_keys=True)

    # Generate SHA-128 hash (SHA-256 recommended if security is a concern)
    hash_obj = hashlib.sha1(json_str.encode())  # SHA-1 (160-bit) or change to sha256
    return hash_obj.hexdigest()



def run_prediction(param_groups):
    rolling_combinations = [
        {'r_window_avg': 2, 'r_min_periods': 2},
        {'r_window_avg': 5, 'r_min_periods': 2},
        {'r_window_avg': 5, 'r_min_periods': 5},
        {'r_window_avg': 10, 'r_min_periods': 5},
        {'r_window_avg': 15, 'r_min_periods': 5},
        {'r_window_avg': 15, 'r_min_periods': 10},
        {'r_window_avg': 20, 'r_min_periods': 15},
        {'r_window_avg': 20, 'r_min_periods': 20},
        {'r_window_avg': 25, 'r_min_periods': 15},
        {'r_window_avg': 25, 'r_min_periods': 25},
        {'r_window_avg': 30, 'r_min_periods': 20},
        {'r_window_avg': 30, 'r_min_periods': 30},
    ]
    for params in param_groups:
        if params['slide_by'] >= params['window']:
            continue

        for data in data_managers:
            for period in data.sleep_periods:
                start_time = period['start_time']
                end_time = period['end_time']
                run_data = RunData(
                    data.piezo_df,
                    start_time,
                    end_time,
                    runtime_params=params,
                    name=data.name,
                    side=period['side'],
                    log=False
                )
                estimate_heart_rate_intervals(run_data)
                gc.collect()
                for combo in rolling_combinations:
                    try:
                        if run_data.df_pred.empty:
                            break
                        df_pred = run_data.df_pred.copy()

                        df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=combo['r_window_avg'], min_periods=combo['r_min_periods']).mean()

                        iter_params = {**params, **combo}
                        analyzed_results = analyze_predictions(data, df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=False)
                        result = {
                            **run_data.chart_info['labels'],
                            **run_data.chart_info['runtime_params'],
                            **analyzed_results['heart_rate']['accuracy'],
                            **iter_params,
                        }

                        params_hash = hash_dict(iter_params)
                        result['params_hash'] = params_hash

                        json_file_name = hash_dict(result)
                        file_path = f'{PROJECT_FOLDER_PATH}src/test/results/{json_file_name}.json'
                        print(f'Saving result to json file: {file_path}')
                        tools.write_json_to_file(file_path, result)
                        print('DONE FOR -----------------------------------------------------------------------------------------------------')
                        print(json.dumps(result, indent=4))
                        df_pred.drop(df_pred.index, inplace=True)
                        del df_pred
                        gc.collect()
                    except Exception as e:
                        print(e)
                        traceback.print_exc()

                del run_data
                gc.collect()
        print('-----------------------------------------------------------------------------------------------------')
        gc.collect()
        print('Finished worker for:')
        print(json.dumps(params, indent=4))


# ---------------------------------------------------------------------------------------------------
def parallel_predictions(param_combinations):
    processes = 34

    random.shuffle(param_combinations)
    param_groups = np.array_split(param_combinations, processes)

    with multiprocessing.Pool(processes=processes, maxtasksperchild=10) as pool:
        r = pool.map(run_prediction, param_groups)
    return r


if __name__ == "__main__":
    # MUST MATCH RuntimeParams from src/run_data.py
    param_grid = {
        "slide_by": [1],
        "window": [6],
        "hr_std_range": [(1,8), (1, 10), (1,12), (1, 15), (1, 20)],
        "percentile": [(20, 75), (20,70), (15,80)],
        "moving_avg_size": [120, 130, 140]
    }

    # Generate all combinations of parameters with named keys
    # param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

    param_combinations.reverse()
    r = parallel_predictions(param_combinations)
