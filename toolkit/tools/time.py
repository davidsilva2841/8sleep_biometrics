from datetime import datetime
from datetime import date as datetime_date
from datetime import timedelta
import re

from toolkit.logger import get_logger


log = get_logger()


def stringify_datetime(date, time=True, fmt=None, millisecond=False):
    """
    Stringifies a datetime object.

    Args:
        date (datetime):
        time (bool):
        fmt (str):
        millisecond (bool):

    Returns:
        str: Stringified datetime
    """
    if not fmt:
        fmt = '%Y-%m-%d'

        if time:
            if millisecond:
                fmt += ' %H:%M:%s'
            else:
                fmt += ' %H:%M:%S'

    return date.strftime(fmt)


def get_datestamp():
    return stringify_datetime(datetime.now(), time=False)


def get_timestamp(millisecond=True):
    return stringify_datetime(datetime.now(), time=True, millisecond=millisecond)


def get_datetime_from_string(date):
    """
    Get datetime object from string.

    Args:
        date (str):

    Returns:
        datetime: Datetime
    """

    # Any numeric character: \d
    # Match from m to n repetitions: {m,n}
    date_regex = r'\d{4}-\d{2}-\d{2}'  # Date format YYYY-MM-DD
    date_time_regex = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'  # Date and time format YYYY-MM-DD HH:MM:SS

    if re.match(date_time_regex, date):
        return datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    elif re.match(date_regex, date):
        return datetime.strptime(date, '%Y-%m-%d')
    else:
        raise Exception('Date input format not understood')


def datetime_from_epoch(epoch, milliseconds=False, apple_format=False):
    """
    Converts epoch timesstamp to datetime.

    Args:
        epoch (str or int):
        milliseconds (bool): If in milliseconds format or not
        apple_format (bool):

    Returns:
        datetime: Datetime
    """
    if milliseconds:
        return datetime.fromtimestamp(int(epoch) / 1000)
    elif apple_format:
        #  DATETIME(visit_time+978307200, 'unixepoch', 'localtime') as timestamp
        if type(epoch) == float:
            epoch = str(epoch).split('.')[0]

        return datetime.fromtimestamp(int(epoch)) + timedelta(seconds=978307200)

    else:
        try:
            date = datetime.fromtimestamp(int(epoch))
            return date
        except ValueError:
            log.warning('Returning date in milliseconds since epoch, value out of range.')

            return datetime_from_epoch(epoch, milliseconds=True)


def time_elapsed(oldest_date, newest_date=None):
    """
    Get time elapsed between two dates
    Args:
        oldest_date (datetime or datetime.date):
        newest_date (datetime):

    Returns:
        datetime: Datetime class
    """

    if not newest_date:
        if type(oldest_date) == datetime.date:
            newest_date = datetime_date.today()
        else:
            newest_date = datetime.now()

    return newest_date - oldest_date


def offset_date(date, years=0, days=0, hours=0, minutes=0, seconds=0):
    """
    Add or subtract time from datetime object.

    Args:
        date (datetime): Datetime object
        years (int):
        days (int):
        hours (int):
        minutes (int):
        seconds (int):

    Returns:
        datetime: Datetime
    """
    return (
        date +
        timedelta(days=years * 365) +
        timedelta(days=days) +
        timedelta(hours=hours) +
        timedelta(minutes=minutes) +
        timedelta(seconds=seconds)
    )
