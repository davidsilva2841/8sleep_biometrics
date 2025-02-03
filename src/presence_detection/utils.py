import argparse
from datetime import datetime, timezone


def validate_datetime_utc(date_str):
    """
    Validate and parse datetime input as UTC.
    Expects format: YYYY-MM-DD HH:MM:SS
    """
    try:
        # Parse datetime string and attach UTC timezone
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime format: '{date_str}'. Use 'YYYY-MM-DD HH:MM:SS'."
        )
