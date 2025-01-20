from typing import TypedDict, List, Literal, Dict


class TimePeriod(TypedDict):
    start_time: str
    end_time: str
    side: Literal['left', 'right']

ValidationFormat = Literal['apple_watch', 'polar']

class SleepEntry(TypedDict):
    sleep_periods: List[TimePeriod]
    sensor_count: Literal[1, 2]
    validation_format: ValidationFormat


SleepDataType = Dict[str, SleepEntry]

SLEEP_DATA: SleepDataType = {
    'david': {
        'validation_format': 'apple_watch',
        'sensor_count': 2,
        'sleep_periods': [
            {'start_time': '2025-01-10 08:00:00', 'end_time': '2025-01-10 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-12 08:00:00', 'end_time': '2025-01-12 15:00:00', 'side': 'right'},
            {'start_time': '2025-01-13 07:30:00', 'end_time': '2025-01-13 13:25:00', 'side': 'right'},
            {'start_time': '2025-01-14 06:30:00', 'end_time': '2025-01-14 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-15 06:30:00', 'end_time': '2025-01-15 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-16 07:00:00', 'end_time': '2025-01-16 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-17 07:00:00', 'end_time': '2025-01-17 12:45:00', 'side': 'right'},
            {'start_time': '2025-01-18 07:10:00', 'end_time': '2025-01-18 14:20:00', 'side': 'right'},
            {'start_time': '2025-01-19 08:06:00', 'end_time': '2025-01-19 15:44:00', 'side': 'right'},
            {'start_time': '2025-01-20 07:36:00', 'end_time': '2025-01-20 14:50:00', 'side': 'right'},
        ]
    },
    'tally': {
        'validation_format': 'apple_watch',
        'sensor_count': 2,
        'sleep_periods': [
            {'start_time': '2025-01-14 06:30:00', 'end_time': '2025-01-14 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-18 06:30:00', 'end_time': '2025-01-18 13:30:00', 'side': 'right'},
            {'start_time': '2025-01-19 06:30:00', 'end_time': '2025-01-19 13:30:00', 'side': 'right'},
        ]
    },
    'trinity': {
        'validation_format': 'apple_watch',
        'sensor_count': 1,
        'sleep_periods': [
            {'start_time': '2025-01-20 03:06:00', 'end_time': '2025-01-20 08:51:00', 'side': 'left'},
        ]
    },
    'den': {
        'validation_format': 'polar',
        'sensor_count': 1,
        'sleep_periods': [
            {'start_time': '2025-01-20 00:36:00', 'end_time': '2025-01-20 08:49:00', 'side': 'right'},
        ]
    }
}

