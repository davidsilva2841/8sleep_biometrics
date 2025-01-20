# ls -1 . | wc -l
import gc
import random
import numpy as np
import multiprocessing
import json
import itertools
from calculations import estimate_heart_rate_intervals, RunData
from analyze import analyze_predictions
import tools
import hashlib
from globals import data_managers

from config import PROJECT_FOLDER_PATH
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
# Define parameter ranges


# ---------------------------------------------------------------------------------------------------
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

        # tally = DataManager('tally', load=True)
        # david = DataManager('david', load=True)

        for data in data_managers:
            for period in data.sleep_periods:
                start_time = period['start_time']
                end_time = period['end_time']
                run_data = RunData(
                    data.piezo_df,
                    start_time,
                    end_time,
                    slide_by=params['slide_by'],
                    window=params['window'],
                    hr_std_range=params['hr_std_range'],
                    percentile=params['percentile'],
                    moving_avg_size=params['moving_avg_size'],
                    name=data.name,
                    side=period['side'],
                    log=False
                )
                estimate_heart_rate_intervals(run_data)
                gc.collect()
                for combo in rolling_combinations:
                    df_pred = run_data.df_pred.copy()
                    df_pred['heart_rate'] = df_pred['heart_rate'].rolling(window=combo['r_window_avg'], min_periods=combo['r_min_periods']).mean()

                    results = analyze_predictions(data, df_pred, run_data, plot=False)
                    iter_params = {**params, **combo}
                    result = {**run_data.chart_info, **results, **iter_params}

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

                del run_data
                gc.collect()
        print('-----------------------------------------------------------------------------------------------------')
        gc.collect()
        print('Finished worker for:')
        print(json.dumps(params, indent=4))


# ---------------------------------------------------------------------------------------------------
def parallel_predictions(param_combinations):
    processes = 30

    random.shuffle(param_combinations)
    param_groups = np.array_split(param_combinations, processes)

    with multiprocessing.Pool(processes=processes, maxtasksperchild=10) as pool:
        r = pool.map(run_prediction, param_groups)
    return r


if __name__ == "__main__":
    param_grid = {
        "slide_by": [1],
        "window": [10, 15, 30, 60],
        "hr_std_range": [(1, 6), (1,8), (1, 10), (1, 15)],
        "percentile": [(2, 98), (3,97), (5, 95), (7.5, 92.5),  (10, 90), (15, 85)],
        "moving_avg_size": [10, 30, 60]
    }

    # Generate all combinations of parameters with named keys
    # param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    param_combinations = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

    param_combinations.reverse()
    r = parallel_predictions(param_combinations)
#
#
