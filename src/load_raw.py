"""
Decodes raw data from 8 sleep pods
"""
import numpy as np
import cbor2
from datetime import datetime, timezone

import tools
from data_types import *



def _decode_piezo_data(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, dtype=np.int32)


def _load_piezo_row(data: dict, file_path: str, seq: int):
    try:
        if 'left1' in data:
            data['left1'] = _decode_piezo_data(data['left1'])
        if 'left2' in data:
            data['left2'] = _decode_piezo_data(data['left2'])

        if 'right1' in data:
            data['right1'] = _decode_piezo_data(data['right1'])
        if 'right2' in data:
            data['right2'] = _decode_piezo_data(data['right2'])

    except Exception as error:
        print('-----------------------------------------------------------------------------------------------')
        print(f'Error decoding piezo data | file_path: {file_path} | seq: {seq}')
        print(f'raw_v2.py:54 data: {data}')
        print(error)
        raise error


def _decode_cbor_file(file_path: str, data: Data, piezo_only: bool):
    print(f'Loading cbor data from: {file_path}')

    with open(file_path, 'rb') as raw_data:
        while True:
            try:
                # Decode the next CBOR object
                row: RawRow = cbor2.load(raw_data)
                decoded_data = cbor2.loads(row['data'])
                if decoded_data['type'] == 'piezo-dual':
                    # piezo-dual rows have nested bytes we need to decode
                    _load_piezo_row(decoded_data, file_path, row['seq'])
                decoded_data['seq'] = row['seq']
                decoded_data['ts'] = datetime.fromtimestamp(
                    decoded_data['ts'],
                    timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")

                if piezo_only:
                    if decoded_data['type'] == 'piezo-dual':
                        # noinspection PyTypedDict
                        data[decoded_data['type']].append(decoded_data)
                else:
                    data[decoded_data['type']].append(decoded_data)

            except EOFError:
                break
            except Exception as error:
                data['errors'].append({
                    'error': error,
                    'raw_data': raw_data,
                    'file_path': file_path,
                })
    return data


def _rename_keys(data: dict) -> Data:
    key_mapping = {
        'log': 'logs',
        'piezo-dual': 'piezo_dual',
        'capSense': 'cap_senses',
        'frzTemp': 'freeze_temps',
        'bedTemp': 'bed_temps',
        'errors': 'errors',
    }

    renamed_data = {key_mapping[key]: value for key, value in data.items() if key in key_mapping}
    return renamed_data


def load_raw_data(folder_path=None, file_path=None, piezo_only: bool = False) -> Data:
    if folder_path:
        files = tools.list_dir_tree(folder_path, files_only=True)
    elif file_path:
        files = [file_path]
    else:
        raise TypeError("'folder_path' or 'file_path' must be present")

    data = {
        'log': [],
        'piezo-dual': [],
        'capSense': [],
        'frzTemp': [],
        'bedTemp': [],
        'rows': [],
        'errors': [],
    }
    for file in files:
        if not file.endswith('.DS_Store') and not file.endswith('SEQNO.RAW') and file.endswith('.RAW'):
            _decode_cbor_file(file, data, piezo_only)

    data = _rename_keys(data)
    return data



