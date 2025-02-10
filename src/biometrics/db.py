import math
import os.path

import sys

import numpy as np

sys.path.append('/home/dac/python_packages/')
import pandas as pd

import platform
from datetime import datetime
import sqlite3
import json
from typing import List

from data_types import *
from get_logger import *

logger = get_logger()

DB_FILE_PATH = f'{logger.folder_path}free-sleep.db'


def custom_serializer(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()  # Convert to ISO 8601 format
    raise TypeError(f"Type {type(obj)} not serializable")


def convert_timestamps(data: List[SleepRecord]) -> List[SleepRecord]:
    formatted_data = []
    for entry in data:
        formatted_entry: SleepRecord = {
            "side": entry["side"],
            "entered_bed_at": datetime.fromisoformat(entry["entered_bed_at"]),
            "left_bed_at": datetime.fromisoformat(entry["left_bed_at"]),
            "sleep_period_seconds": entry["sleep_period_seconds"],
            "times_exited_bed": entry["times_exited_bed"],
            "present_intervals": [
                (datetime.fromisoformat(start), datetime.fromisoformat(end))
                for start, end in entry["present_intervals"]
            ],
            "not_present_intervals": [
                (datetime.fromisoformat(start), datetime.fromisoformat(end))
                for start, end in entry["not_present_intervals"]
            ]
        }
        formatted_data.append(formatted_entry)
    return formatted_data

# Create a persistent connection
conn = sqlite3.connect(DB_FILE_PATH, isolation_level=None, check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")  # Enable WAL mode
conn.execute("PRAGMA busy_timeout=5000;")  # Wait up to 5 seconds if locked
cursor = conn.cursor()

def insert_vitals(data: dict):
    """
    Inserts a record into the 'vitals' table. If a conflict occurs, it skips the insertion.
    """
    # conn = sqlite3.connect(DB_FILE_PATH)
    # cursor = conn.cursor()

    sql = """
    INSERT INTO vitals (side, period_start, heart_rate, hrv, breathing_rate)
    VALUES (:side, :period_start, :heart_rate, :hrv, :breathing_rate)
    ON CONFLICT(side, period_start) DO NOTHING;
    """
    if np.isnan(data['hrv']):
        data['hrv'] = 0
    else:
        data['hrv'] = math.floor(data['hrv'])

    if np.isnan(data['breathing_rate']):
        data['breathing_rate'] = 0
    else:
        data['breathing_rate'] = math.floor(data['breathing_rate'])

    data['heart_rate'] = math.floor(data['heart_rate'])
    print('Inserting row...')
    print(json.dumps(data, indent=4))

    try:
        cursor.execute(sql, data)
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")



def fetch_sleep_records_between(start_time: datetime, end_time: datetime, side: Side) -> List[SleepRecord]:
    """
    Fetch all sleep records where entered_bed_at is between start_time and end_time.

    :param start_time: Start datetime (inclusive).
    :param end_time: End datetime (inclusive).
    :return: List of SleepRecord objects with parsed intervals.
    """
    logger.debug(f"Fetching sleep records between {start_time} and {end_time} from {DB_FILE_PATH}...")

    conn = sqlite3.connect(DB_FILE_PATH)
    cur = conn.cursor()

    # SQL Query to select sleep records within the given range
    query = """
    SELECT *
    FROM sleep_records
    WHERE side = ? AND entered_bed_at BETWEEN ? AND ?
    ORDER BY entered_bed_at ASC;
    """

    # Execute query with formatted timestamps
    cur.execute(query, (side, start_time.isoformat() + 'Z', end_time.isoformat() + 'Z'))

    # Fetch all results
    rows = cur.fetchall()

    # Get column names for constructing dictionaries
    columns = [desc[0] for desc in cur.description]

    cur.close()
    conn.close()

    # Convert each row to a dictionary and parse JSON intervals
    sleep_records = []
    for row in rows:
        record = dict(zip(columns, row))

        # Convert entered_bed_at and left_bed_at back to datetime
        record["entered_bed_at"] = datetime.fromisoformat(record["entered_bed_at"].rstrip('Z'))
        record["left_bed_at"] = datetime.fromisoformat(record["left_bed_at"].rstrip('Z')) if record["left_bed_at"] else None

        # Decode present_intervals
        if record["present_intervals"]:
            record["present_intervals"] = [
                (datetime.fromisoformat(start.rstrip('Z')), datetime.fromisoformat(end.rstrip('Z')))
                for start, end in json.loads(record["present_intervals"])
            ]
        else:
            record["present_intervals"] = []

        # Decode not_present_intervals
        if record["not_present_intervals"]:
            record["not_present_intervals"] = [
                (datetime.fromisoformat(start.rstrip('Z')), datetime.fromisoformat(end.rstrip('Z')))
                for start, end in json.loads(record["not_present_intervals"])
            ]
        else:
            record["not_present_intervals"] = []

        sleep_records.append(record)

    logger.debug(f"Fetched {len(sleep_records)} record(s) from 'sleep_records'.")
    return sleep_records



def insert_sleep_records(sleep_records: List[SleepRecord]):
    """
    Inserts a list of records into the sleep_records table in the given database.
    Each record is expected to have:
      - side (str)
      - entered_bed_at (datetime)
      - left_bed_at (datetime, optional)
      - sleep_period_seconds (int)
      - times_exited_bed (int)
      - present_intervals (list of [start, end] datetime pairs)
      - not_present_intervals (list of [start, end] datetime pairs)
    """
    logger.debug(f'Inserting sleep records into {DB_FILE_PATH}...')

    conn = sqlite3.connect(DB_FILE_PATH)
    cur = conn.cursor()
    insert_query = """
    INSERT OR IGNORE INTO sleep_records (
        side,
        entered_bed_at,
        left_bed_at,
        sleep_period_seconds,
        times_exited_bed,
        present_intervals,
        not_present_intervals
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """

    # Convert records to tuples for insertion
    values_to_insert = []
    for sleep_record in sleep_records:
        side = sleep_record['side']
        entered_bed_at = int(sleep_record['entered_bed_at'].timestamp())
        left_bed_at = int(sleep_record.get('left_bed_at').timestamp())
        sleep_period_seconds = sleep_record.get('sleep_period_seconds', 0)
        times_exited_bed = sleep_record.get('times_exited_bed', 0)

        # Encode intervals as JSON strings
        present_intervals_str = json.dumps([
            [int(start.timestamp()), int(end.timestamp())] for start, end in sleep_record.get('present_intervals', [])
        ])
        not_present_intervals_str = json.dumps([
            [int(start.timestamp()), int(end.timestamp())] for start, end in sleep_record.get('not_present_intervals', [])
        ])

        # Prepare the data tuple
        row_tuple = (
            side,
            entered_bed_at,
            left_bed_at,
            sleep_period_seconds,
            times_exited_bed,
            present_intervals_str,
            not_present_intervals_str
        )
        values_to_insert.append(row_tuple)

    cur.executemany(insert_query, values_to_insert)
    conn.commit()
    cur.close()
    conn.close()
    logger.debug(f"Inserted {len(sleep_records)} record(s) into 'sleep_records' (ignoring duplicates).")

import atexit
atexit.register(lambda: conn.close())


