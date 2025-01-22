import gc
import math
import time
from datetime import datetime, timedelta
import numpy as np
from typing import Union, Tuple, TypedDict
import heartpy as hp
import statistics
from data_types import *
import tools
from heart.filtering import filter_signal


class Measurement(TypedDict):
    start_time: str
    end_time: str
    heart_rate: float
    hrv: float
    breathing_rate: float
    source: str


class Measurements(TypedDict):
    combined: List[Measurement]



class ChartLabels(TypedDict, total=False):
    name: str
    start_time: str
    end_time: str
    label: str
    elapsed: float


class RuntimeParams(TypedDict):
    window: int
    slide_by: int
    moving_avg_size: int
    hr_std_range: Tuple[int, int]
    percentile: Tuple[int, int]


class Accuracy(TypedDict):
    window: int
    slide_by: int
    moving_avg_size: int
    hr_std_range: Tuple[int, int]
    percentile: Tuple[int, int]


class ChartInfo(TypedDict, total=False):
    labels: ChartLabels
    runtime_params: RuntimeParams
    accuracy: Accuracy


class RunData:
    df_pred: Union[None, pd.DataFrame]
    df_pred_side_1: Union[None, pd.DataFrame]
    df_pred_side_2: Union[None, pd.DataFrame]
    chart_info: ChartInfo

    def _load_piezo_df(self, piezo_df: pd.DataFrame):
        # Convert start_time and end_time to datetime
        start_time_dt = pd.to_datetime(self.start_time)
        end_time_dt = pd.to_datetime(self.end_time)
        # Load only the rows we need
        self.piezo_df: pd.DataFrame = piezo_df.loc[start_time_dt:end_time_dt]
        # self.piezo_df.loc[:, self.side_1] = self.piezo_df[self.side_1].apply(lambda x: hp.filter_signal(x, [0.05, 15], 500, filtertype='bandpass'))
        # self.piezo_df.loc[:, self.side_2] = self.piezo_df[self.side_2].apply(lambda x: hp.filter_signal(x, [0.05, 15], 500, filtertype='bandpass'))
        # self.piezo_df[self.side_1] = self.piezo_df[self.side_1].apply(lambda x: hp.filter_signal(x, [0.05, 15], 500, filtertype='bandpass'))
        # self.piezo_df[self.side_2] = self.piezo_df[self.side_2].apply(lambda x: hp.filter_signal(x, [0.05, 15], 500, filtertype='bandpass'))


    def __init__(
            self,
            piezo_df: pd.DataFrame,
            start_time: str,
            end_time: str,
            runtime_params: RuntimeParams,
            name: str = '',
            side: str = 'right',
            sensor_count=2,
            log=True,
            label=''
    ):

        # Runtime parameters
        self.slide_by: int = runtime_params['slide_by']                       # Sliding window step size in seconds
        self.window: int = runtime_params['window']                           # Window size in seconds
        self.hr_std_range: Tuple[int, int] = runtime_params['hr_std_range']   # Heart rate standard deviation range (lower, upper)
        self.percentile: Tuple[int, int] = runtime_params['percentile']       # Percentile range (lower, upper)
        self.moving_avg_size: int = runtime_params['moving_avg_size']         # Moving average window size in seconds

        self.name: str = name                               # Name
        self.side: str = side                               # Side of the bed (e.g., 'left', 'right')
        self.side_1: str = f'{side}1'
        self.side_2: str = f'{side}2'
        self.start_time: str = start_time                   # Start time in 'YYYY-MM-DD HH:MM:SS' format
        self.end_time: str = end_time                       # End time in 'YYYY-MM-DD HH:MM:SS' format
        self.log = log
        self.senor_count = sensor_count

        self._load_piezo_df(piezo_df)

        # Running metrics
        self.heart_rates: List[float] = []                  # Store average heart rates
        self.last_heart_rates: List[float] = []             # Store last <moving_avg_size> heart rates
        self.lower_bound: Union[float, None] = None         # Lower bound of HR
        self.upper_bound: Union[float, None] = None         # Upper bound of HR
        self.hr_moving_avg: Union[float, None] = None       # Current moving average heart rate
        self.hr_std_2: Union[float, None] = None            # Standard deviation of heart rate

        self.i: int = 0                                     # Iteration counter
        self.sensor_1_drop_count: int = 0
        self.sensor_2_drop_count: int = 0
        self.sensor_1_error_count: int = 0
        self.sensor_2_error_count: int = 0
        self.dropped_from_percentile: int = 0

        # Define the interval
        self.start_interval = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        self.end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

        window_time = timedelta(seconds=self.window)
        self.end_interval = self.start_interval + window_time
        self.slide_by_time = timedelta(seconds=self.slide_by)

        total_seconds = (self.end_datetime - self.start_interval).total_seconds()
        if total_seconds < 0:
            raise Exception(f'end_time is before start_time: {start_time} -> {end_time}')

        self.total_intervals = math.ceil(total_seconds / self.slide_by)
        self.progress_bar_update_interval = 100
        self.bar = tools.progress_bar(math.ceil(self.total_intervals / self.progress_bar_update_interval))

        self.measurements_side_1 = []
        self.measurements_side_2 = []
        self.combined_measurements = []
        self.chart_info = {
            'labels': {
                'name': self.name,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'label': label,
                'elapsed': 0,
            },
            'runtime_params': {
                'window': self.window,
                'slide_by': self.slide_by,
                'moving_avg_size': self.moving_avg_size,
                'hr_std_range': self.hr_std_range,
                'percentile': self.percentile,
            },
            'accuracy': None
        }

        self.timer_start = None
        self.timer_end = None
        self.elapsed_time = None

    def start_timer(self):
        """Start the timer."""
        self.timer_start = time.time()
        print("Timer started...")

    def stop_timer(self):
        """Stop the timer and return elapsed time."""
        if self.timer_start is None:
            raise Exception("Timer was not started.")
        self.timer_end = time.time()
        self.elapsed_time = self.timer_end - self.timer_start
        self.chart_info['labels']['elapsed'] = self.elapsed_time
        print(f"Timer stopped. Elapsed time: {self.elapsed_time:.2f} seconds.")



    def next(self):
        # Calculate the heart rate STD
        if len(self.heart_rates) >= self.moving_avg_size:
            self.last_heart_rates = self.heart_rates[-self.moving_avg_size:]
            self.hr_moving_avg = statistics.mean(self.last_heart_rates)

            self.lower_bound = np.percentile(self.last_heart_rates, self.percentile[0])
            self.upper_bound = np.percentile(self.last_heart_rates, self.percentile[1])

            if self.upper_bound - self.lower_bound < 25:
                self.upper_bound = self.hr_moving_avg + 12.5
                self.lower_bound = self.hr_moving_avg - 12.5

            self.hr_std_2 = statistics.stdev(self.last_heart_rates) * 2
            if self.hr_std_2 < self.hr_std_range[0]:
                self.hr_std_2 = self.hr_std_range[0]
            elif self.hr_std_2 > self.hr_std_range[1]:
                self.hr_std_2 = self.hr_std_range[1]

        self.i += 1
        if self.log:
            if self.i % self.progress_bar_update_interval == 0:
                self.bar.update()

        self.start_interval += self.slide_by_time
        self.end_interval += self.slide_by_time


    def is_valid(self, measurement) -> bool:
        if np.isnan(measurement['bpm']):
            return False

        if measurement['bpm'] > 90:
            return False
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound < measurement['bpm'] < self.upper_bound:
                return True
            else:
                self.dropped_from_percentile += 1
                # print(f"calculations.py:75 DROP lower_bound: {self.lower_bound} <-> upper_bound: {self.upper_bound} | bpm: {measurement['bpm'] }")
                return False
        return True


    def print_results(self):
        if self.log:
            print('-----------------------------------------------------------------------------------------------------')
            print(f'Estimated heart rate for {self.name} {self.start_time} -> {self.end_time}')
            print(f"Elapsed time: {self.elapsed_time:.2f} seconds.")
            print(f"Sensor 1 - Dropped    {self.sensor_1_drop_count:,}/{self.total_intervals:,}  ({(self.sensor_1_drop_count / self.total_intervals) * 100:.2f}%)")
            print(f"Sensor 1 - Errors     {self.sensor_1_error_count:,}/{self.total_intervals:,}  ({(self.sensor_1_error_count / self.total_intervals) * 100:.2f}%)")
            sensor_1_predicted = self.total_intervals - self.sensor_1_error_count - self.sensor_1_drop_count
            print(f"Sensor 1 - Predicted  {sensor_1_predicted:,}/{self.total_intervals:,}  ({(sensor_1_predicted / self.total_intervals) * 100:.2f}%)")

            print(f"Sensor 2 - Dropped: {self.sensor_2_drop_count:,}/{self.total_intervals:,}  ({(self.sensor_2_drop_count / self.total_intervals) * 100:.2f}%)")
            print(f"Sensor 2 - Errors: {self.sensor_2_error_count:,}/{self.total_intervals:,}  ({(self.sensor_2_error_count / self.total_intervals) * 100:.2f}%)")
            sensor_2_predicted = self.total_intervals - self.sensor_2_error_count - self.sensor_2_drop_count
            print(f"Sensor 2 - Predicted  {sensor_2_predicted:,}/{self.total_intervals:,}  ({(sensor_2_predicted / self.total_intervals) * 100:.2f}%)")

            print(f"Dropped b/ of percentile: {self.dropped_from_percentile:,}/{self.total_intervals:,}  ({(self.dropped_from_percentile / self.total_intervals) * 100:.2f}%)")
            predicted_inverval_count = len(self.combined_measurements)
            print(f"Total predicted: {predicted_inverval_count:,}/{self.total_intervals:,}  ({(predicted_inverval_count / self.total_intervals) * 100:.2f}%)")


    def combine_results(self):
        self.df_pred = pd.DataFrame(self.combined_measurements)
        if not self.df_pred.empty:
            self.df_pred.dropna(subset=['heart_rate'], inplace=True)
            self.df_pred.sort_values(by='start_time', inplace=True)
        else:
            print(f'EMPTY DATAFRAME {self.name} {self.start_time} -> {self.end_time}')
        # self.df_pred_side_1 = pd.DataFrame(self.measurements_side_1)
        # self.df_pred_side_1.dropna(subset=['heart_rate'], inplace=True)
        # self.df_pred_side_1.sort_values(by='start_time', inplace=True)
        #
        # self.df_pred_side_2 = pd.DataFrame(self.measurements_side_2)
        # self.df_pred_side_2.dropna(subset=['heart_rate'], inplace=True)
        # self.df_pred_side_2.sort_values(by='start_time', inplace=True)

