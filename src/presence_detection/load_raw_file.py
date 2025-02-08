import numpy as np
import traceback
from datetime import datetime, timedelta, timezone
import cbor2
from pathlib import Path
import gc
import sys
import os


# Add the current directory to sys.path
sys.path.append(os.getcwd())
from presence_types import *

from logger import get_logger

logger = get_logger()


def get_current_files(folder_path: str):
    return [
        str(f.resolve())
        for f in Path(folder_path).glob('*.RAW')
        if f.is_file() and f.name != 'SEQNO.RAW'
    ]


def _decode_piezo_data(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, dtype=np.int32)


def _load_piezo_row(data: dict, side: Side):
    if side == 'left':
        if 'left1' in data:
            data['left1'] = _decode_piezo_data(data['left1'])
        if 'left2' in data:
            data['left2'] = _decode_piezo_data(data['left2'])
    else:
        if 'right1' in data:
            data['right1'] = _decode_piezo_data(data['right1'])
        if 'right2' in data:
            data['right2'] = _decode_piezo_data(data['right2'])


def _delete_other_side(decoded_data: dict, side: Side):
    """
    Delete other sides data for saving memory space
    """
    try:
        del_side = 'left'
        if side == 'left':
            del_side = 'right'

        if decoded_data['type'] == 'capSense':
            del decoded_data[del_side]
        else:
            # Delete sensor 2 of the current side
            if f'{side}2' in decoded_data:
                del decoded_data[f'{side}2']
            # Delete opposite side
            del decoded_data[f'{del_side}1']
            if f'{del_side}2' in decoded_data:
                del decoded_data[f'{del_side}2']
    except Exception as error:
        logger.error(error)
        traceback.print_exc()
        print(decoded_data)
        raise error


def _decode_cbor_file(file_path: str, data: dict, start_time, end_time, side: Side):
    # logger.debug(f'Loading cbor data from: {file_path}')
    checked_timespan = False
    with open(file_path, 'rb') as raw_data:
        while True:
            try:

                # Decode the next CBOR object
                row = cbor2.load(raw_data)
                decoded_data = cbor2.loads(row['data'])
                if not decoded_data['type'] in ['piezo-dual', 'capSense']:
                    continue
                _delete_other_side(decoded_data, side)
                if not checked_timespan:
                    timestamp_start = datetime.fromtimestamp(
                        decoded_data['ts'],
                        timezone.utc
                    )
                    timestamp_end = timestamp_start + timedelta(minutes=15)
                    if start_time <= timestamp_start <= end_time:
                        checked_timespan = True
                    else:
                        if start_time <= timestamp_end <= end_time:
                            checked_timespan = True
                        else:
                            raw_data.close()
                            return

                if decoded_data['type'] == 'piezo-dual':
                    _load_piezo_row(decoded_data, side)

                decoded_data['ts'] = datetime.fromtimestamp(
                    decoded_data['ts'],
                    timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")
                data[decoded_data['type']].append(decoded_data)

            except EOFError:
                break
            except Exception as error:
                logger.error(error)
        raw_data.close()
        gc.collect()
    return data



def _rename_keys(data: dict):
    key_mapping = {
        'piezo-dual': 'piezo_dual',
        'capSense': 'cap_senses',
    }
    for old_key, new_key in key_mapping.items():
        if old_key in data:
            data[new_key] = data.pop(old_key)



def load_raw_files(folder_path: str, start_time: datetime, end_time: datetime, side: Side):
    logger.debug(f'Loading RAW files from {folder_path} | {start_time.isoformat()} -> {end_time.isoformat()}')
    data = {
        'piezo-dual': [],
        'capSense': [],
    }
    file_paths = get_current_files(folder_path)

    if len(file_paths) == 0:
        logger.error('No file paths detected!')
        raise FileNotFoundError(f'No files found for: {folder_path}')

    for file_path in file_paths:
        if os.path.isfile(file_path):
            _decode_cbor_file(file_path, data, start_time, end_time, side)
        else:
            logger.warning(f'File path deleted before parsed! {file_path}')
    _rename_keys(data)
    gc.collect()
    return data




