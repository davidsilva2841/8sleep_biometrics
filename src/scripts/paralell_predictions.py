from typing import Tuple

from param_optimizer.globals import data_managers
import json
from data_manager import DataManager, TimePeriod
from calculations import estimate_heart_rate_intervals, RunData
from analyze import analyze_predictions
import multiprocessing

from run_data import RuntimeParams, PostRuntimeParams


RUNTIME_PARAMS: RuntimeParams = {
    'window': 8,
    'slide_by': 1,
    'moving_avg_size': 100,
    'hr_std_range': (1,12),
    'percentile': (25, 75),
}

POST_RUNTIME_PARAMS: PostRuntimeParams = {
    'r_window_avg': 20,
    'r_min_periods': 15,
}

LABEL = ''

def run_prediction(job: Tuple[DataManager, TimePeriod]):
    data, period = job
    run_data = RunData(
        data.piezo_df,
        period['start_time'],
        period['end_time'],
        runtime_params=RUNTIME_PARAMS,
        name=data.name,
        side=period['side'],
        sensor_count=1,
        label=LABEL,
        log=False
    )
    estimate_heart_rate_intervals(run_data)
    run_data.df_pred['heart_rate'] = run_data.df_pred['heart_rate'].rolling(
        window=POST_RUNTIME_PARAMS['r_window_avg'],
        min_periods=POST_RUNTIME_PARAMS['r_min_periods']
    ).mean()
    results = analyze_predictions(data, run_data.df_pred, run_data.start_time, run_data.end_time, run_data.chart_info, plot=True)
    result = {
        **run_data.chart_info['labels'],
        **run_data.chart_info['runtime_params'],
        **results['heart_rate']['accuracy'],
    }
    print('-----------------------------------------------------------------------------------------------------')
    print(json.dumps(result, indent=4))



def parallel_predictions():
    jobs = []
    processes = 2
    for data in data_managers:
        for period in data.sleep_periods:
            jobs.append((data, period))
    with multiprocessing.Pool(processes=processes, maxtasksperchild=10) as pool:
        r = pool.map(run_prediction, jobs)
    return r


if __name__ == "__main__":
    r = parallel_predictions()

