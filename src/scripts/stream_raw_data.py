import os
import time
from datetime import datetime, timezone, timedelta
import cbor2
import struct
import re
import json
import math

PERSISTENT_DIR = '/persistent/'
DEVICE_STATUS_DICT_DIR = '/home/dac/free-sleep/data/'
DEVICE_STATUS_DICT_JSON = f'{DEVICE_STATUS_DICT_DIR}/device_status.json'
CHECK_INTERVAL_SECONDS = 1
FILE_UPDATE_THRESHOLD_SECONDS = 30 * 60  # 30 minutes


def pretty_print_dict(d, indent=0):
    """Pretty print a dictionary without using the pprint module."""
    for key, value in d.items():
        if isinstance(value, dict):
            print(' ' * indent + f'{key}:')
            pretty_print_dict(value, indent + 4)
        else:
            print(' ' * indent + f'{key}: {value}')


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _decode_piezo_and_pres_data_from_cbor(raw_bytes: bytes) -> list[int]:
    data_points = len(raw_bytes) // 4
    return list(struct.unpack('<' + 'i' * data_points, raw_bytes))


def decode_raw_row_data(raw_row):
    """Decodes the 'data' part of a raw row."""
    try:
        no_data = False
        if not isinstance(raw_row, dict):
            print(f"Error: raw_row is not a dictionary, but: {type(raw_row)}")
            print(f"raw_row: {raw_row}")
            time.sleep(CHECK_INTERVAL_SECONDS)
            return None

        if b'AsioTcpClinet.h' in raw_row.get('data', b''):
            # Ignoring spammy AsioTcpClinet
            decoded_data = cbor2.loads(raw_row['data'])
            ts = decoded_data.get('ts')
            decoded_data['ts'] = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            decoded_data['unix_ts'] = datetime.fromtimestamp(ts, timezone.utc).timestamp()
            decoded_data['seq'] = raw_row['seq']
            print("if b'AsioTcpClinet.h' in raw_row.get('data',b''): found")
            print(decoded_data)
            return None

        if 'type' in raw_row and raw_row['type'] == 'log':
            # raw_row: {'type': 'log', 'ts': 1738531506, 'level': 'debug', 'msg': '11384013 AsioTcpClinet.h:95 write|[asiotcp] write ec: 32 (Broken pipe)'}
            if 'AsioTcpClinet.h' in raw_row['msg']:
                # Ignoring spammy AsioTcpClinet
                print("if 'AsioTcpClinet.h' in raw_row['msg']: found")
                print(raw_row)
                return None
            no_data = True
        elif 'data' not in raw_row or 'seq' not in raw_row:  # Check for expected keys
            print(f"Error: raw_row missing 'data' or 'seq' keys. raw_row keys are: {raw_row.keys()}")
            print(f"raw_row: {raw_row}")
            return None

        if no_data:
            decoded_data = raw_row
        else:
            # print(f"Decoding raw_row: {raw_row['seq']}, {type(raw_row['data'])}, {raw_row['data']}")
            decoded_data = cbor2.loads(raw_row['data'])
            decoded_data['seq'] = raw_row['seq']

        # Flatten the dictionary
        flattened_data = flatten_dict(decoded_data)

        ts = flattened_data.get('ts')
        if ts is not None:
            flattened_data['ts'] = datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            flattened_data['unix_ts'] = datetime.fromtimestamp(ts, timezone.utc).timestamp()

        # Decode byte data
        for key, value in flattened_data.items():
            if isinstance(value, bytes):
                for code_name in ['left1', 'left2', 'right1', 'right2']:
                    if code_name in key:
                        try:
                            flattened_data[key] = _decode_piezo_and_pres_data_from_cbor(value)
                        except Exception as e:
                            print(f'Error decoding piezo bytes for key {key}: {e}')
                        break
        return flattened_data

    except Exception as e:
        print(f"General Error decoding data within raw_row: {e}")  # Catch other exceptions
        print(f"raw_row: {flattened_data}")
        time.sleep(CHECK_INTERVAL_SECONDS / 10)
        return None


def get_newest_raw_file(directory):
    """Returns the path to the newest .RAW file in the directory, or None if none exists."""
    raw_files = [f for f in os.listdir(directory) if f.endswith('.RAW')]
    if not raw_files:
        return None

    files_with_paths = [os.path.join(directory, f) for f in raw_files]
    newest_file = max(files_with_paths, key=os.path.getmtime)
    return newest_file


def delete_old_files(directory, threshold_seconds):
    """Deletes .RAW files in the directory that haven't been modified in threshold_seconds."""
    now = datetime.now(timezone.utc)
    for filename in os.listdir(directory):
        if filename.endswith('.RAW') and 'SEQNO.RAW' not in filename:  # added SEQNO.RAW exclusion
            filepath = os.path.join(directory, filename)
            modified_time = datetime.fromtimestamp(os.path.getmtime(filepath), timezone.utc)
            if (now - modified_time) > timedelta(seconds=threshold_seconds):
                try:
                    os.remove(filepath)
                    print(f"Deleted old file: {filepath}")
                except Exception as e:
                    print(f"Error deleting file {filepath}: {e}")


def f_to_c(f):
    c = float(f - 32) * 5 / 9
    return c


def c_to_f(c):
    f = (float(c) * 9 / 5 + 32)
    return f


def is_floatable(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_mean_and_stdev(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    stdev = math.sqrt(variance)
    return mean, stdev


def update_device_status_dict_with_decoded_row(decoded_line, device_status_dict):
    """Updates the device status dictionary with values extracted from the decoded log message."""
    if decoded_line.get('type') == 'log':
        msg = decoded_line.get('msg')
        if msg:
            # Use regex to extract values
            for key, regex in {
                'avg_samples_per_sec': r"avg samples/sec: ([\d.]+)",
                'avg_lps_samples_per_sec': r"avg lps samples/sec: ([\d.]+)",
                'cmd_temps': r"\[cmd] ([\d.]+) ([\d.]+) ([\d.]+) \| ([\d.]+) ([\d.]+) ([\d.]+) \| amb ([\d.]+)",
                'lisL_read': r"\[lisL] read \(([\d.]+)mg, ([-+]?[\d.]+)mg, ([-+]?[\d.]+)mg\)",
                'lisR_read': r"\[lisR] read \(([\d.]+)mg, ([-+]?[\d.]+)mg, ([-+]?[\d.]+)mg\)",
                'sampling_avg_rate_piezo': r"\[sampling] avg rate ([\d.]+)Hz",
                'sampling_adc_misses_per_sec': r"\[sampling] avg rate [\d.]+Hz, ([\d.]+) adc misses/sec",
                'capwater_raw': r"\[capwater] \(raw: ([\d.]+), perc: (\d+)\)",
                'condensation_temp': r"\[condensation] temp: ([\d.]+)",
                'condensation_humidity': r"\[condensation] temp: [\d.]+, humidity: ([\d.]+)",
                'condensation_dew_point': r"\[condensation] temp: [\d.]+, humidity: [\d.]+, dew point: ([\d.]+)",
                'solenoid_current': r"\[solenoid] solenoid_current @ ([\d.]+)A",
                'cap_sense': r"\[cap_sense] ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)",
                'cap_sense_ttd': r"\[cap_sense] ttd: ([\d.]+) ([\d.]+)",
                'temps': r"temps ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)",
                'therm_tref': r"\[therm] Tref=([\d.]+)",
                'ambient_temp': r"\[ambient] temp ([\d.]+)",
                'ambient_humidity': r"\[ambient] temp [\d.]+ humidity ([\d.]+) percent",
                'top_fan_speed': r"\[top-fan] ([\d.]+) @ (\d+) rpm",
                'bottom_fan_speed': r"\[bottom-fan] ([\d.]+) @ (\d+) rpm",
                'pid_heatsink': r"pid\[heatsink] ([-+]?[\d.]+) ([-+]?[\d.]+) ([-+]?[\d.]+) ([-+]?[\d.]+) ([-+]?[\d.]+)",
                'pump_left': r"pump\[left] ([a-z]+) @ (\d+)% \((\d+) rpm\) \[([a-z]+)\]",
                'pump_right': r"pump\[right] ([a-z]+) @ (\d+)% \((\d+) rpm\) \[([a-z]+)\]"
            }.items():
                match = re.search(regex, msg)
                if match:
                    if key in ('pump_left', 'pump_right'):
                        status, percent, rpm, pump_type = match.groups()
                        device_status_dict[key] = {
                            'status': status,
                            'percent': int(percent),
                            'rpm': int(rpm),
                            'type': pump_type
                        }
                    elif key == 'lisL_read' or key == 'lisR_read' or key == "cap_sense" or key == 'cap_sense_ttd' or key == "pid_heatsink":  # handle cases with multiple values
                        # store as float, or as string if not floatable
                        device_status_dict[key] = [float(x) for x in match.groups() if is_floatable(x)]
                    elif key == "capwater_raw":
                        # store as float, or as string if not floatable
                        device_status_dict['cap_water'] = {
                        }
                        if is_floatable(match.group(1)):
                            device_status_dict['cap_water']['raw'] = float(match.group(1))
                        else:
                            device_status_dict['cap_water']['raw'] = match.group(1)
                        if is_floatable(match.group(2)):
                            device_status_dict['cap_water']['percent'] = float(match.group(2))
                        else:
                            device_status_dict['cap_water']['percent'] = match.group(2)
                    elif key == 'cmd_temps':
                        device_status_dict[key] = [c_to_f(float(x)) for x in match.groups() if is_floatable(x)]
                    elif key == 'condensation_temp':
                        device_status_dict[key] = c_to_f(float(match.group(1)))
                    elif key == 'condensation_dew_point':
                        device_status_dict['condensation_dew_point_temp'] = c_to_f(float(match.group(1)))
                    elif key == 'condensation_humidity':
                        device_status_dict['condensation_humidity_percent'] = float(match.group(1))
                    elif key == 'ambient_temp':
                        device_status_dict['ambient_temp'] = c_to_f(float(match.group(1)))
                    elif key == 'ambient_humidity':
                        device_status_dict['ambient_humidity_percent'] = float(match.group(1))
                    elif key == 'therm_tref':
                        device_status_dict['therm_tref_reference_temp'] = c_to_f(float(match.group(1)))
                    elif key == 'temps':
                        device_status_dict[key] = [c_to_f(float(x)) for x in match.groups() if is_floatable(x)]
                    elif key == "top_fan_speed":
                        current, rpm = match.groups()
                        device_status_dict[key] = {
                            'current_maybe_not_sure': float(current),
                            'rpm': int(rpm),
                        }
                    elif key == "bottom_fan_speed":
                        current, rpm = match.groups()
                        device_status_dict[key] = {
                            'current_maybe_not_sure': float(current),
                            'rpm': int(rpm),
                        }
                    else:
                        try:
                            if is_floatable(match.group(1)):
                                device_status_dict[key] = float(match.group(1))
                            else:
                                device_status_dict[key] = match.group(1)
                        except ValueError:
                            device_status_dict[key] = match.group(1)
    elif decoded_line.get('type') == 'piezo-dual':
        decoded_type = decoded_line.pop('type')
        if 'last_updated' in device_status_dict.get(decoded_type, {}):
            last_updated = device_status_dict[decoded_type]['last_updated']
        else:
            last_updated = time.time()
            if decoded_type not in device_status_dict:
                device_status_dict[decoded_type] = {}
            device_status_dict[decoded_type]['last_updated'] = last_updated
        if time.time() - last_updated > 5:
            for key in ['left1', 'left2', 'right1', 'right2']:
                raw_data = decoded_line.get(key)
                if raw_data:
                    num_data_points = len(raw_data)
                    mean, stdev = get_mean_and_stdev(raw_data)
                    decoded_line[key] = {
                        'mean': mean,
                        'stdev': stdev,
                        'num_data_points': num_data_points
                    }
            device_status_dict[decoded_type] = decoded_line
            device_status_dict[decoded_type]['last_updated'] = time.time()
    elif decoded_line.get('type') == 'bedTemp':
        decoded_type = decoded_line.pop('type')
        decoded_line['ambient_temp'] = c_to_f(float(decoded_line.pop('amb', -1)) / 100)
        decoded_line['mcu_temp'] = c_to_f(float(decoded_line.pop('mcu', -1)) / 100)
        decoded_line['hu_temp'] = c_to_f(float(decoded_line.pop('hu', -1)) / 100)
        decoded_line['left_side_temp'] = c_to_f(float(decoded_line.pop('left_side', -1)) / 100)
        decoded_line['left_out_temp'] = c_to_f(float(decoded_line.pop('left_out', -1)) / 100)
        decoded_line['left_cen_temp'] = c_to_f(float(decoded_line.pop('left_cen', -1)) / 100)
        decoded_line['left_in_temp'] = c_to_f(float(decoded_line.pop('left_in', -1)) / 100)
        decoded_line['right_side_temp'] = c_to_f(float(decoded_line.pop('right_side', -1)) / 100)
        decoded_line['right_out_temp'] = c_to_f(float(decoded_line.pop('right_out', -1)) / 100)
        decoded_line['right_cen_temp'] = c_to_f(float(decoded_line.pop('right_cen', -1)) / 100)
        decoded_line['right_in_temp'] = c_to_f(float(decoded_line.pop('right_in', -1)) / 100)
        device_status_dict[decoded_type] = decoded_line
    elif decoded_line.get('type') == 'frzTemp':
        decoded_type = decoded_line.pop('type')
        decoded_line['left_temp'] = c_to_f(float(decoded_line.pop('left', -1)) / 100)
        decoded_line['right_temp'] = c_to_f(float(decoded_line.pop('right', -1)) / 100)
        decoded_line['hs_temp'] = c_to_f(float(decoded_line.pop('left', -1)) / 100)
        decoded_line['ambient_temp'] = c_to_f(float(decoded_line.pop('right', -1)) / 100)
    else:
        if 'type' in decoded_line:
            decoded_type = decoded_line.pop('type')
        else:
            decoded_type = 'unknown_type'
        device_status_dict[decoded_type] = decoded_line


def tail_and_decode_raw_file(file_path, device_status_dict):
    """Continuously tails a .RAW file, decodes CBOR objects directly, and prints to terminal."""
    current_file = None
    last_position = 0  # Track the last successful read position
    last_save_time = time.time()

    try:
        with open(file_path, 'rb') as f:
            current_file = f
            f.seek(last_position)  # Start from the last known position
            while True:
                try:
                    current_pos = f.tell()  # Remember the current position before reading
                    raw_row = cbor2.load(f)  # Load CBOR object directly from file stream
                    last_position = f.tell()  # Update last good position after successful read
                    decoded_line = decode_raw_row_data(raw_row)
                    if decoded_line:
                        update_device_status_dict_with_decoded_row(decoded_line, device_status_dict)
                        if time.time() - last_save_time > 3:
                            with open(DEVICE_STATUS_DICT_JSON, 'w+') as json_fp:
                                json.dump(device_status_dict, json_fp)
                            last_save_time = time.time()
                            print('\n\n\n\n\n')

                            pretty_print_dict(device_status_dict, indent=4)


                except cbor2.CBORDecodeError:
                    # Handle incomplete read by waiting for more data
                    f.seek(current_pos)  # Rewind to the position before the failed read
                    # Wait until the file size increases beyond current_pos
                    max_wait_time = 5  # Maximum time to wait for new data
                    exit_loop = False
                    while True:
                        current_size = os.path.getsize(file_path)
                        if current_size > current_pos:
                            # New data available, break to retry reading
                            break
                        time.sleep(CHECK_INTERVAL_SECONDS / 10)

                        if (time.time() - current_pos) > max_wait_time:
                            print(f"Timeout waiting for new data at position {current_pos}. Maybe the file is done being updated?")
                            exit_loop = True
                            break
                    if exit_loop:
                        break

                    time.sleep(CHECK_INTERVAL_SECONDS)
                except Exception as e:
                    print(f"General error during processing: {e}")
                    time.sleep(CHECK_INTERVAL_SECONDS)  # Prevent tight loop on repeated errors

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error tailing file: {e}")
    finally:
        if current_file:
            current_file.close()

    if time.time() - last_save_time > 3:
        with open(DEVICE_STATUS_DICT_JSON, 'w+') as json_fp:
            json.dump(device_status_dict, json_fp)
        last_save_time = time.time()


def main():
    last_checked_file = None
    if not os.path.exists(DEVICE_STATUS_DICT_DIR):
        os.makedirs(DEVICE_STATUS_DICT_DIR)
    device_status_dict = {}

    while True:
        # 1. Delete old files
        delete_old_files(PERSISTENT_DIR, FILE_UPDATE_THRESHOLD_SECONDS)

        # 2. Find the newest .RAW file
        newest_file_path = get_newest_raw_file(PERSISTENT_DIR)

        if newest_file_path:
            print(f"Monitoring new file: {newest_file_path}")
            last_checked_file = newest_file_path
            tail_and_decode_raw_file(newest_file_path, device_status_dict)
        else:
            if last_checked_file:
                print("No .RAW files found. Waiting...")
                last_checked_file = None
            time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
