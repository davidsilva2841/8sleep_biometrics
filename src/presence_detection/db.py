import os.path
import sys
sys.path.append('/home/dac/python_packages/')
import pandas as pd

import platform
from datetime import datetime
import sqlite3
import json
from typing import List

from presence_types import *
from logger import *

logger = get_logger()

if platform.system().lower() == 'linux':
    DB_FILE_PATH = '/persistent/free-sleep-data/free-sleep.db'
else:
    DB_FILE_PATH = './free-sleep.db'


# Sample data
data = [
    {
        "side": "right",
        "entered_bed_at": "2025-02-02T05:01:35",
        "left_bed_at": "2025-02-02T14:06:39",
        "sleep_period_seconds": 32554,
        "times_exited_bed": 2,
        "present_intervals": [
            [
                "2025-02-02T05:01:35",
                "2025-02-02T05:05:11"
            ],
            [
                "2025-02-02T05:06:20",
                "2025-02-02T06:07:21"
            ],
            [
                "2025-02-02T06:08:42",
                "2025-02-02T14:06:39"
            ]
        ],
        "not_present_intervals": [
            [
                "2025-02-02T05:05:11",
                "2025-02-02T05:06:20"
            ],
            [
                "2025-02-02T06:07:21",
                "2025-02-02T06:08:42"
            ]
        ]
    }
]

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


def create_db_and_table():
    """
    Creates a local SQLite database (if it doesn't exist) at db_path
    and a sleep_records table (if it doesn't exist). Also creates an index
    on (side, entered_bed_at) for faster lookups.
    """

    conn = sqlite3.connect(DB_FILE_PATH)
    cur = conn.cursor()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS sleep_records (
        side TEXT NOT NULL,
        entered_bed_at TEXT NOT NULL,
        left_bed_at TEXT,
        sleep_period_seconds INTEGER,
        times_exited_bed INTEGER,
        present_intervals TEXT,
        not_present_intervals TEXT,
        PRIMARY KEY (side, entered_bed_at)
    );
    """
    cur.execute(create_table_query)

    # Create an index (side + entered_bed_at).
    create_index_query = """
    CREATE INDEX IF NOT EXISTS idx_sleep_records_side_entered_bed_at
    ON sleep_records (side, entered_bed_at);
    """
    cur.execute(create_index_query)

    conn.commit()
    cur.close()
    conn.close()
    logger.debug(f"Database '{DB_FILE_PATH}' and table 'sleep_records' ready.")


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
        entered_bed_at = sleep_record['entered_bed_at'].isoformat()
        left_bed_at = sleep_record.get('left_bed_at')
        left_bed_at = left_bed_at.isoformat() if left_bed_at else None
        sleep_period_seconds = sleep_record.get('sleep_period_seconds', 0)
        times_exited_bed = sleep_record.get('times_exited_bed', 0)

        # Encode intervals as JSON strings
        present_intervals_str = json.dumps([
            [start.isoformat(), end.isoformat()] for start, end in sleep_record.get('present_intervals', [])
        ])
        not_present_intervals_str = json.dumps([
            [start.isoformat(), end.isoformat()] for start, end in sleep_record.get('not_present_intervals', [])
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



# if __name__ == "__main__":
#     # 1. Create the database and table
#     create_db_and_table()
#
#     # 2. Insert sample data
#     insert_sleep_records(data)
