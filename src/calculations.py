import math
import time
from datetime import datetime, timedelta
from typing import Union, Tuple
import heartpy as hp
import statistics
from src.data_types import *
from toolkit import tools
import traceback

class Measurement(TypedDict):
    start_time: str
    end_time: str
    heart_rate: float
    hrv: float
    breathing_rate: float
    source: str


class Measurements(TypedDict):
    combined: List[Measurement]


class RunData:
    df_pred: Union[None, pd.DataFrame]
    df_pred_side_1: Union[None, pd.DataFrame]
    df_pred_side_2: Union[None, pd.DataFrame]
    chart_info: dict

    # def _load_piezo_df(self, piezo_df: pd.DataFrame):


    def __init__(
            self,
            piezo_df: pd.DataFrame,
            start_time: str,
            end_time: str,
            slide_by: int = 1,
            window: int = 10,
            hr_std_range: Tuple[int, int] = (1,8),
            percentile: Tuple[int, int] = (1,99),
            moving_avg_size: int = 60,
            name: str = '',
            side: str = 'right'
    ):
        # Convert start_time and end_time to datetime
        start_time_dt = pd.to_datetime(start_time)
        end_time_dt = pd.to_datetime(end_time)
        # Load only the rows we need
        self.piezo_df: pd.DataFrame = piezo_df.loc[start_time_dt:end_time_dt]

        # Parameters
        self.slide_by: int = slide_by                       # Sliding window step size in seconds
        self.window: int = window                           # Window size in seconds
        self.hr_std_range: Tuple[int, int] = hr_std_range   # Heart rate standard deviation range (lower, upper)
        self.percentile: Tuple[int, int] = percentile       # Percentile range (lower, upper)
        self.moving_avg_size: int = moving_avg_size         # Moving average window size in seconds
        self.name: str = name                               # Name
        self.side: str = side                               # Side of the bed (e.g., 'left', 'right')
        self.side_1: str = f'{side}1'
        self.side_2: str = f'{side}2'
        self.start_time: str = start_time                   # Start time in 'YYYY-MM-DD HH:MM:SS' format
        self.end_time: str = end_time                       # End time in 'YYYY-MM-DD HH:MM:SS' format


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
        self.end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')

        window_time = timedelta(seconds=window)
        self.end_interval = self.start_interval + window_time
        self.slide_by_time = timedelta(seconds=slide_by)

        total_seconds = (self.end_time - self.start_interval).total_seconds()
        if total_seconds < 0:
            raise Exception(f'end_time is before start_time: {start_time} -> {end_time}')

        self.total_intervals = math.ceil(total_seconds / slide_by)
        self.progress_bar_update_interval = 100
        self.bar = tools.progress_bar(math.ceil(self.total_intervals / self.progress_bar_update_interval))

        self.measurements_side_1 = []
        self.measurements_side_2 = []
        self.combined_measurements = []
        self.chart_info = {
            'window': self.window,
            'slide_by': self.slide_by,
            'moving_avg_size': self.moving_avg_size,
            'hr_std_range': self.hr_std_range,
            'percentile': self.percentile,
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
        self.df_pred.dropna(subset=['heart_rate'], inplace=True)
        self.df_pred.sort_values(by='start_time', inplace=True)

        # self.df_pred_side_1 = pd.DataFrame(self.measurements_side_1)
        # self.df_pred_side_1.dropna(subset=['heart_rate'], inplace=True)
        # self.df_pred_side_1.sort_values(by='start_time', inplace=True)
        #
        # self.df_pred_side_2 = pd.DataFrame(self.measurements_side_2)
        # self.df_pred_side_2.dropna(subset=['heart_rate'], inplace=True)
        # self.df_pred_side_2.sort_values(by='start_time', inplace=True)


# ---------------------------------------------------------------------------------------------------
# region CLEAN DF

def clean_df_pred(df_pred: pd.DataFrame) -> pd.DataFrame:

    breathing_lower_threshold = 10
    breathing_upper_threshold = 23

    # Replace values outside the threshold with NaN
    df_pred['breathing_rate'] = df_pred['breathing_rate'].where(
        (df_pred['breathing_rate'] >= breathing_lower_threshold) & (df_pred['breathing_rate'] <= breathing_upper_threshold), np.nan
    )

    # Fill NaN with the last valid value (forward fill)
    # df_pred['breathing_rate'] = df_pred['breathing_rate'].ffill()
    df_pred['breathing_rate'] = df_pred['breathing_rate'].interpolate(method='linear')

    # Fill any remaining NaN with a rolling mean
    window_size = 3
    df_pred['breathing_rate'] = df_pred['breathing_rate'].rolling(window=window_size, min_periods=1).mean()

    hrv_lower_threshold = 10
    hrv_upper_threshold = 100

    # Replace values outside the threshold with NaN
    df_pred['hrv'] = df_pred['hrv'].where(
        (df_pred['hrv'] >= hrv_lower_threshold) & (df_pred['hrv'] <= hrv_upper_threshold), np.nan
    )

    # Fill NaN with the last valid value (forward fill)
    # df_pred['hrv'] = df_pred['hrv'].ffill()
    df_pred['hrv'] = df_pred['hrv'].interpolate(method='linear')

    # Fill any remaining NaN with a rolling mean
    window_size = 30
    df_pred['hrv'] = df_pred['hrv'].rolling(window=window_size, min_periods=10).mean()
    return df_pred


# endregion


# ---------------------------------------------------------------------------------------------------
# region CALCULATIONS




def _calculate(run_data: RunData, side: str):
    np_array = np.concatenate(run_data.piezo_df[run_data.start_interval:run_data.end_interval][side])

    data = hp.filter_signal(np_array, [0.05, 15], 500, filtertype='bandpass')
    data = hp.scale_data(data)
    working_data, measurement = hp.process(
        data,
        500,
        freq_method='fft',
        breathing_method='fft',
        bpmmin=40,
        hampel_correct=False,     # KEEP FALSE - Takes too long
        bpmmax=90,
        reject_segmentwise=False, # KEEP FALSE - Less accurate
        windowsize=0.5,
        clipping_scale=False,     # KEEP FALSE - Did not change reading
        clean_rr=True,            # KEEP TRUE - More accurate
        clean_rr_method='quotient-filter', # z-score is worse
    )
    if run_data.is_valid(measurement):
        return {
            'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
            'heart_rate': measurement['bpm'],
            'hrv': measurement['sdnn'],
            'breathing_rate': measurement['breathingrate'] * 60,
            'source': 'combined',
        }
    return None




def estimate_heart_rate_intervals(run_data: RunData):
    print('-----------------------------------------------------------------------------------------------------')
    print(f'Estimating heart rate for {run_data.name} {run_data.start_time} -> {run_data.end_time}')
    # Convert strings to datetime objects
    run_data.start_timer()
    while run_data.start_interval <= run_data.end_time:
        measurement_1 = None
        measurement_2 = None
        try:
            measurement_1 = _calculate(run_data, run_data.side_1)
        except Exception as e:
            run_data.sensor_1_error_count += 1
            # traceback.print_exc()

        try:
            measurement_1 = _calculate(run_data, run_data.side_2)
        except Exception as e:
            run_data.sensor_2_error_count += 1

        if measurement_1 is not None and measurement_2 is not None:
            run_data.measurements_side_1.append(measurement_1)
            run_data.measurements_side_2.append(measurement_2)

            if np.isnan(m1_heart_rate) or np.isnan(m2_heart_rate):
                print('-----------------------------------------------------------------------------------------------------')
                print('DETECTED NAN!!!!')
            m1_heart_rate = measurement_1['heart_rate']
            m2_heart_rate = measurement_2['heart_rate']
            if run_data.hr_moving_avg is not None and not np.isnan(m1_heart_rate) and not np.isnan(m2_heart_rate):
                heart_rate = (((m1_heart_rate + m2_heart_rate) / 2) + run_data.hr_moving_avg) / 2
            elif not np.isnan(m1_heart_rate) and not np.isnan(m2_heart_rate):
                heart_rate = (m1_heart_rate + m2_heart_rate) / 2
            elif not np.isnan(m1_heart_rate):
                if run_data.hr_moving_avg is not None:
                    heart_rate = (m1_heart_rate + run_data.hr_moving_avg) / 2
                else:
                    heart_rate = m1_heart_rate
            elif not np.isnan(m2_heart_rate):
                if run_data.hr_moving_avg is not None:
                    heart_rate = (m2_heart_rate + run_data.hr_moving_avg) / 2
                else:
                    heart_rate = m2_heart_rate
            else:
                heart_rate = m1_heart_rate

            if not np.isnan(heart_rate):
                if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                    if heart_rate < run_data.hr_moving_avg:
                        heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                    else:
                        heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

                run_data.heart_rates.append(heart_rate)

            run_data.combined_measurements.append({
                'start_time': run_data.start_interval.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': run_data.end_interval.strftime('%Y-%m-%d %H:%M:%S'),
                'heart_rate': heart_rate,
                'hrv': (measurement_1['hrv'] + measurement_2['hrv']) / 2,
                'breathing_rate': (measurement_1['breathing_rate'] + measurement_2['breathing_rate']) / 2 * 60,
                'source': 'combined',
            })

        elif measurement_1 is not None:
            run_data.sensor_2_drop_count += 1
            m1_heart_rate = measurement_1['heart_rate']

            if run_data.hr_moving_avg is not None and not np.isnan(m1_heart_rate):
                heart_rate = (m1_heart_rate + run_data.hr_moving_avg) / 2
            else:
                heart_rate = m1_heart_rate


            if not np.isnan(heart_rate):
                if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                    if heart_rate < run_data.hr_moving_avg:
                        heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                    else:
                        heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

                run_data.heart_rates.append(heart_rate)

            measurement_1['heart_rate'] = heart_rate
            run_data.combined_measurements.append(measurement_1)

        elif measurement_2 is not None:
            run_data.sensor_1_drop_count += 1
            m2_heart_rate = measurement_2['heart_rate']

            if run_data.hr_moving_avg is not None and not np.isnan(m2_heart_rate):
                heart_rate = (m2_heart_rate + run_data.hr_moving_avg) / 2
            else:
                heart_rate = m2_heart_rate

            if not np.isnan(heart_rate):
                if run_data.hr_moving_avg is not None and abs(heart_rate - run_data.hr_moving_avg) > run_data.hr_std_2:
                    if heart_rate < run_data.hr_moving_avg:
                        heart_rate = run_data.hr_moving_avg - run_data.hr_std_2
                    else:
                        heart_rate = run_data.hr_moving_avg + run_data.hr_std_2

                run_data.heart_rates.append(heart_rate)

            measurement_2['heart_rate'] = heart_rate
            run_data.combined_measurements.append(measurement_2)

        run_data.next()


    run_data.stop_timer()
    run_data.print_results()
    run_data.combine_results()



# endregion
