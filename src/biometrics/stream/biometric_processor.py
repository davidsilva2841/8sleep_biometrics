import gc
from typing import Union, Tuple, TypedDict, List, Optional
import traceback
import numpy as np
import json

from get_logger import get_logger
from vitals.run_data import RuntimeParams
from vitals.cleaning import interpolate_outliers_in_wave
from heart.preprocessing import scale_data
from heart.filtering import filter_signal, remove_baseline_wander
from heart.heartpy import process
from db import insert_vitals
from data_types import *
logger = get_logger()






class BiometricProcessor:
    heart_rates: List[float]        # Store average heart rates
    last_heart_rates: List[float]   # Store last <moving_avg_size> heart rates
    lower_bound: Optional[np.floating]    # Lower bound of HR (None if not set)
    upper_bound: Optional[np.floating]    # Upper bound of HR (None if not set)
    hr_moving_avg: Optional[np.floating]  # Current moving average heart rate
    hr_std_2: Optional[float]       # Standard deviation of heart rate
    def __init__(
            self,
            side: str = 'left',
            sensor_count=1,
            runtime_params: RuntimeParams = None,
            insertion_frequency=60,
            insert_to_sql=True,
    ):
        self.present = False
        self.side = side
        self.sensor_count = sensor_count
        self.insertion_frequency = insertion_frequency
        self.insert_to_sql = insert_to_sql
        self.iteration_count = 0
        if runtime_params is None:
            runtime_params: RuntimeParams = {
                'window': 3,
                'slide_by': 1,
                'moving_avg_size': 100,
                'hr_std_range': (1, 10),
                'hr_percentile': (20, 75),
                'signal_percentile': (0.2, 99.8),
                'window_size': 0.65,
            }

        self.slide_by = runtime_params['slide_by']  # Sliding window step size in seconds
        self.window = runtime_params['window']  # Window size in seconds
        self.hr_std_range = runtime_params['hr_std_range']  # Heart rate standard deviation range (lower, upper)
        self.hr_percentile = runtime_params['hr_percentile']  # Accepted percentile range for heart rate (lower, upper)
        self.moving_avg_size = runtime_params['moving_avg_size']  # Moving average window size in seconds
        self.signal_percentile = runtime_params['signal_percentile']  # Percent of outliers from raw signal to replace
        self.window_size = runtime_params['window_size']
        self.init_tracking()
        self.no_presence_tolerance = 10
        self.not_present_for = 0
        self.combined_measurements = []
        self.smoothed_measurements = []

    def init_tracking(self):
        # Running metrics
        self.heart_rates = []
        self.last_heart_rates = []
        self.lower_bound = None
        self.upper_bound = None
        self.hr_moving_avg = None
        self.hr_std_2 = None

    def reset(self):
        self.iteration_count = 0
        self.init_tracking()


    def detect_presence(self, signal: np.ndarray):
        signal_range = np.ptp(signal)
        if signal_range > 200_000:
            self.not_present_for = 0
            self.present = True
            # logger.debug(f'{self.side} side is present! | Signal range: {signal_range:,}')
        else:
            # logger.debug(f'{self.side} side is not present')
            self.not_present_for += 1
            if self.not_present_for == self.no_presence_tolerance:
                logger.debug(f'User not detected for {self.no_presence_tolerance} seconds on {self.side} side, resetting...')
                self.present = False
                self.reset()


    def _calculate_vitals(self, signal: np.ndarray, epoch: int):
        # Remove outliers from signal
        data = interpolate_outliers_in_wave(
            signal,
            lower_percentile=self.signal_percentile[0],
            upper_percentile=self.signal_percentile[1],
        )

        data = scale_data(data, lower=0, upper=1024)
        data = remove_baseline_wander(data, sample_rate=500.0, cutoff=0.05)

        data = filter_signal(
            data,
            cutoff=[0.5, 20.0],
            sample_rate=500.0,
            order=2,
            filtertype='bandpass'
        )

        working_data, measurement = process(
            data,
            500,
            breathing_method='fft',
            bpmmin=40,
            bpmmax=90,
            reject_segmentwise=False,
            windowsize=self.window_size,
            clean_rr_method='quotient-filter',
            calculate_breathing=True,
        )
        if self.is_valid(measurement):
            return {
                'side': self.side,
                'period_start': epoch,
                'heart_rate': measurement['bpm'],
                'hrv': measurement['sdnn'],
                'breathing_rate': measurement['breathingrate'] * 60,
            }
        return None


    def calculate_vitals(self, epoch: int, signal1: np.ndarray, signal2: Union[None, np.ndarray] = None):
        measurement_1 = None
        measurement_2 = None
        try:
            measurement_1 = self._calculate_vitals(signal1, epoch)
        except Exception as e:
            pass
            # traceback.print_exc()

        if signal2 is not None:
            try:
                measurement_2 = self._calculate_vitals(signal2, epoch)
            except Exception as e:
                pass
                # traceback.print_exc()

        if measurement_1 is not None and measurement_2 is not None:
            m1_heart_rate = measurement_1['heart_rate']
            m2_heart_rate = measurement_2['heart_rate']
            if self.hr_moving_avg is not None:
                heart_rate = (((m1_heart_rate + m2_heart_rate) / 2) + self.hr_moving_avg) / 2
            else:
                heart_rate = (m1_heart_rate + m2_heart_rate) / 2

            if self.hr_moving_avg is not None and abs(heart_rate - self.hr_moving_avg) > self.hr_std_2:
                if heart_rate < self.hr_moving_avg:
                    heart_rate = self.hr_moving_avg - self.hr_std_2
                else:
                    heart_rate = self.hr_moving_avg + self.hr_std_2

            self.heart_rates.append(heart_rate)

            self.combined_measurements.append({
                'side': self.side,
                'period_start': epoch,
                'heart_rate': heart_rate,
                'hrv': (measurement_1['hrv'] + measurement_2['hrv']) / 2,
                'breathing_rate': (measurement_1['breathing_rate'] + measurement_2['breathing_rate']) / 2 * 60,
                # 'moving_average': self.hr_moving_avg,
                # 'hr_std_2': self.hr_std_2,
            })

        elif measurement_1 is not None:
            m1_heart_rate = measurement_1['heart_rate']

            # If the HR differs by more than the allowable movement
            if self.hr_moving_avg is not None and abs(m1_heart_rate - self.hr_moving_avg) > self.hr_std_2:
                if m1_heart_rate < self.hr_moving_avg:
                    m1_heart_rate = self.hr_moving_avg - self.hr_std_2
                else:
                    m1_heart_rate = self.hr_moving_avg + self.hr_std_2

            self.heart_rates.append(m1_heart_rate)

            measurement_1['heart_rate'] = m1_heart_rate
            # measurement_1['moving_average'] = self.hr_moving_avg
            # measurement_1['hr_std_2'] = self.hr_std_2
            self.combined_measurements.append(measurement_1)

        elif measurement_2 is not None:
            m2_heart_rate = measurement_2['heart_rate']

            if self.hr_moving_avg is not None:
                heart_rate = (m2_heart_rate + self.hr_moving_avg) / 2
            else:
                heart_rate = m2_heart_rate

            if self.hr_moving_avg is not None and abs(heart_rate - self.hr_moving_avg) > self.hr_std_2:
                if heart_rate < self.hr_moving_avg:
                    heart_rate = self.hr_moving_avg - self.hr_std_2
                else:
                    heart_rate = self.hr_moving_avg + self.hr_std_2

            self.heart_rates.append(heart_rate)

            measurement_2['heart_rate'] = heart_rate
            # measurement_2['moving_average'] = self.hr_moving_avg
            # measurement_2['hr_std_2'] = self.hr_std_2
            self.combined_measurements.append(measurement_2)

        self.next()


    def is_valid(self, measurement) -> bool:
        if np.isnan(measurement['bpm']):
            return False

        if measurement['bpm'] > 90:
            return False
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound < measurement['bpm'] < self.upper_bound:
                return True
            else:
                return False
        return True


    def next(self):
        self.iteration_count += 1
        if self.iteration_count % self.insertion_frequency == 0 and len(self.combined_measurements) > 0:
            if self.insert_to_sql:
                # Convert last heart rate to average
                self.combined_measurements[-1]['heart_rate'] = np.mean(self.heart_rates[-20:])
                insert_vitals(self.combined_measurements[-1])

                del self.combined_measurements
                gc.collect()
                self.combined_measurements = []
            else:
                smoothed_measurement = {**self.combined_measurements[-1], 'heart_rate': np.mean(self.heart_rates[-20:])}
                self.smoothed_measurements.append(smoothed_measurement)


        if len(self.heart_rates) >= self.moving_avg_size:
            if len(self.heart_rates) > self.moving_avg_size:
                self.heart_rates.pop(0)
            # TODO: Remove last heart rates and use heart_rates, just keep heart_rates to most recent X
            self.last_heart_rates = self.heart_rates[-self.moving_avg_size:]
            self.hr_moving_avg = np.mean(self.last_heart_rates)

            self.lower_bound = np.percentile(self.last_heart_rates, self.hr_percentile[0])
            self.upper_bound = np.percentile(self.last_heart_rates, self.hr_percentile[1])

            if self.upper_bound - self.lower_bound < 25:
                self.upper_bound = self.hr_moving_avg + 12.5
                self.lower_bound = self.hr_moving_avg - 12.5

            self.hr_std_2 = np.std(self.last_heart_rates) * 2
            if self.hr_std_2 < self.hr_std_range[0]:
                self.hr_std_2 = self.hr_std_range[0]
            elif self.hr_std_2 > self.hr_std_range[1]:
                self.hr_std_2 = self.hr_std_range[1]
