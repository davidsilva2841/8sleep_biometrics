import sys
import os
from datetime import datetime, timezone
import cbor2
import argparse
import numpy as np
import pandas as pd



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
        print(error)
        raise error


def _decode_cbor_file(file_path: str, data, piezo_only: bool):
    print(f'Loading cbor data from: {file_path}')

    with open(file_path, 'rb') as raw_data:
        while True:
            try:
                # Decode the next CBOR object
                row = cbor2.load(raw_data)
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


def _rename_keys(data: dict):
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


def list_dir_tree(root_folder, files_only=False):
    if not os.path.isdir(root_folder):
        raise Exception("Folder doesn't exist")

    subfile_paths = []
    for root, folders, files in os.walk(root_folder):
        if not files_only:
            subfile_paths.append(root)
        for file in files:
            file_path = os.path.join(root, file)
            if files_only:
                if os.path.isfile(file_path):
                    subfile_paths.append(file_path)
            else:
                subfile_paths.append(file_path)

    return subfile_paths


def load_raw_data(folder_path=None, file_path=None, piezo_only: bool = False):
    if folder_path:
        files = list_dir_tree(folder_path, files_only=True)
    elif file_path:
        files = [file_path]
    else:
        raise TypeError("'folder_path' or 'file_path' must be present")

    if len(files) == 0:
        print(f'No raw files found in: {folder_path}, double check your from folder path')
    print(f'Decoding {len(files)} file(s)...')
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

    print(f'Decoded {len(files)} file(s)')
    data = _rename_keys(data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy all files from a source folder to a destination folder.")
    parser.add_argument("from_folder", type=str, help="Path to the source folder")
    parser.add_argument("to_folder", type=str, help="Path to the destination folder")
    # If no arguments provided, show a friendly message
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample usage:")
        print("  python script.py /path/data/raw/ /path/to/save_output")
        sys.exit(1)
    args = parser.parse_args()
    data = load_raw_data(folder_path=args.from_folder, piezo_only=True)
    df = pd.DataFrame(data['piezo_dual'])
    file_path = os.path.join(args.to_folder, 'output.pkl.zip')
    print(f'Saving data to file...')
    df.to_pickle(file_path, compression='zip')
    print(f'Saved file, please share {file_path}')

