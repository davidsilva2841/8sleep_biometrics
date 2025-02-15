"""
Defines sleep data structures and sleep periods for different individuals

This is loaded by DataManager
"""
from typing import TypedDict, List, Literal, Dict


class TimePeriod(TypedDict):
    start_time: str
    end_time: str
    side: Literal['left', 'right']

Name = Literal['alina', 'david', 'den', 'elisa', 'tally', 'trinity']
ValidationFormat = Literal['apple_watch', 'polar']
RawFormat = Literal['raw', 'pkl']

class SleepEntry(TypedDict):
    sleep_periods: List[TimePeriod]
    sensor_count: Literal[1, 2]
    validation_format: ValidationFormat
    raw_format: RawFormat


SleepDataType = Dict[Name, SleepEntry]

SLEEP_DATA: SleepDataType = {
    'alina': {
        'validation_format': 'apple_watch',
        'sensor_count': 1,
        'raw_format': 'pkl',
        'sleep_periods': [
            {'start_time': '2025-01-26 00:58:20', 'end_time': '2025-01-26 08:47:18', 'side': 'left'},
            {'start_time': '2025-01-29 23:16:28', 'end_time': '2025-01-30 06:23:28', 'side': 'left'},
        ]
    },
    'david': {
        'validation_format': 'apple_watch',
        'sensor_count': 1,
        'raw_format': 'raw',
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
            {'start_time': '2025-01-21 06:00:00', 'end_time': '2025-01-21 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-22 06:30:00', 'end_time': '2025-01-22 13:30:00', 'side': 'right'},
            {'start_time': '2025-01-23 06:30:00', 'end_time': '2025-01-23 14:30:00', 'side': 'right'},
            {'start_time': '2025-01-24 07:00:00', 'end_time': '2025-01-24 14:45:00', 'side': 'right'},
            {'start_time': '2025-01-25 08:00:00', 'end_time': '2025-01-25 14:43:00', 'side': 'right'},
            {'start_time': '2025-01-26 07:55:00', 'end_time': '2025-01-26 14:43:00', 'side': 'right'},
            {'start_time': '2025-01-27 06:15:00', 'end_time': '2025-01-27 14:53:00', 'side': 'right'},
            {'start_time': '2025-02-14 06:35:00', 'end_time': '2025-02-14 14:29:00', 'side': 'right'},
            {'start_time': '2025-02-15 07:01:00', 'end_time': '2025-02-15 14:44:00', 'side': 'right'},
        ]
    },
    'den': {
        'validation_format': 'polar',
        'sensor_count': 1,
        'raw_format': 'pkl',
        'sleep_periods': [
            {'start_time': '2025-01-21 02:47:00', 'end_time': '2025-01-21 09:38:00', 'side': 'right'},
        ]
    },
    'elisa': {
        'validation_format': 'apple_watch',
        'raw_format': 'raw',
        'sensor_count': 2,
        'sleep_periods': [
            {'start_time': '2025-01-22 22:30:00', 'end_time': '2025-01-23 06:00:00', 'side': 'left'},
        ]
    },
    'tally': {
        'validation_format': 'apple_watch',
        'sensor_count': 2,
        'raw_format': 'raw',
        'sleep_periods': [
            {'start_time': '2025-01-14 06:30:00', 'end_time': '2025-01-14 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-18 06:30:00', 'end_time': '2025-01-18 13:30:00', 'side': 'right'},
            {'start_time': '2025-01-19 06:30:00', 'end_time': '2025-01-19 13:30:00', 'side': 'right'},
            {'start_time': '2025-01-20 06:00:00', 'end_time': '2025-01-20 14:00:00', 'side': 'right'},
            {'start_time': '2025-01-21 05:30:00', 'end_time': '2025-01-21 14:08:00', 'side': 'left'},
            {'start_time': '2025-01-22 05:30:00', 'end_time': '2025-01-22 13:50:00', 'side': 'left'},
            {'start_time': '2025-01-23 06:00:00', 'end_time': '2025-01-23 13:45:00', 'side': 'left'},
            {'start_time': '2025-01-24 05:35:00', 'end_time': '2025-01-24 14:00:00', 'side': 'left'},
            {'start_time': '2025-01-25 07:00:00', 'end_time': '2025-01-25 14:30:00', 'side': 'left'},
            {'start_time': '2025-01-26 05:00:00', 'end_time': '2025-01-26 15:15:00', 'side': 'left'},

        ]
    },
    'trinity': {
        'validation_format': 'apple_watch',
        'sensor_count': 1,
        'raw_format': 'raw',
        'sleep_periods': [
            {'start_time': '2025-01-20 03:06:00', 'end_time': '2025-01-20 08:51:00', 'side': 'left'},
            {'start_time': '2025-01-22 00:06:00', 'end_time': '2025-01-22 06:42:00', 'side': 'left'},
        ]
    },
}

