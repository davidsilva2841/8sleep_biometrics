from typing import Union, Tuple, TypedDict, List, Optional

import math
from get_logger import get_logger
from biometric_processor import BiometricProcessor
from data_types import *
logger = get_logger()


class StreamProcessor:
    def __init__(self, piezo_record, buffer_size=10, baseline_buffer=60):
        if 'left2' in piezo_record:
            self.sensor_count = 2
        else:
            self.sensor_count = 1
        self.buffer_size = buffer_size
        self.piezo_buffer: List[PiezoDualData] = []
        self.baseline_buffer = baseline_buffer

        self.left_side_present = False
        self.right_side_present = False
        self.left_processor = BiometricProcessor(side='left', sensor_count=self.sensor_count, insertion_frequency=60)
        self.right_processor = BiometricProcessor(side='right', sensor_count=self.sensor_count, insertion_frequency=60)
        self.midpoint = math.floor(buffer_size / 2)
        self.iteration_count = 0

    def check_presence(self, left1_signal: np.ndarray, right1_signal: np.ndarray):
        self.left_processor.detect_presence(left1_signal)
        self.right_processor.detect_presence(right1_signal)



    def process_piezo_record(self, piezo_record: PiezoDualData):
        self.piezo_buffer.append(piezo_record)
        if len(self.piezo_buffer) == self.buffer_size:
            self.piezo_buffer.pop(0)
            left1_signal = np.hstack([entry["left1"] for entry in self.piezo_buffer]).ravel()
            right1_signal = np.hstack([entry["right1"] for entry in self.piezo_buffer]).ravel()

            self.iteration_count += 1
            log = self.iteration_count % 60 == 0

            self.check_presence(left1_signal, right1_signal)
            epoch = self.piezo_buffer[self.midpoint]['ts']
            time = datetime.fromtimestamp(epoch)
            if self.left_processor.present:
                if log:
                    logger.debug(f'Presence detected for left side @ {time.isoformat()}')
                if self.sensor_count == 2:
                    left2_signal = np.hstack([entry["left2"] for entry in self.piezo_buffer]).ravel()
                else:
                    left2_signal = None

                self.left_processor.calculate_vitals(epoch, left1_signal, left2_signal)

            if self.right_processor.present:
                if log:
                    logger.debug(f'Presence detected for right side @ {time.isoformat()}')
                if self.sensor_count == 2:
                    right2_signal = np.hstack([entry["right2"] for entry in self.piezo_buffer]).ravel()
                else:
                    right2_signal = None
                self.right_processor.calculate_vitals(epoch, right1_signal, right2_signal)




