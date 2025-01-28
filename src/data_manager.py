"""
Manages sleep data by handling raw sensor data and validation data processing.

SENSOR DATA:
- New raw data should be saved to `8sleep_biometrics/data/people/<PERSON>/raw/load/`.
- After filtering, data is stored in `8sleep_biometrics/data/people/<PERSON>/raw/<PERSON>_piezo_df.feather`.
- `DataManager` loads the processed data from the feather file to `DataManager.piezo_df`.
- Ensure `sleep_data.py` is updated with new sleep periods in UTC format; otherwise, data outside those periods will be excluded.

VALIDATION DATA:
- Handles 2 types of validation data, apple watch & polar
- Validation data is saved to 8sleep_biometrics/data/people/<PERSON>/validation/

EXAMPLE USAGE
    # Copy the raw data from the 8 Sleep Pod
    $ scp -r -P 8822 'pod2:/persistent/*.RAW' /Users/ds/main/8sleep_biometrics/data/people/david/raw/load

    # Load and process data in Python
    data = DataManager('david')

    data.piezo_df       # Sensor data
    data.sleep_periods

FUNCTIONALITY:
- Cleans and processes raw sensor data.
- Filters data based on defined sleep periods.
- Updates validation datasets for heart rate, HRV, respiratory rate, and sleep stages.
- Stores cleaned and processed data for further analysis.
"""
from typing import List
import os
from pathlib import Path
import re
import pandas as pd

import tools
from data_types import Data
from load_raw import load_raw_data
from sleep_data import SLEEP_DATA, TimePeriod, ValidationFormat, RawFormat, Name
from config import PROJECT_FOLDER_PATH



# ---------------------------------------------------------------------------------------------------
# region Helper functions
# Basic util helper functions for DataManager

def _clean_files(file_paths):
    for file_path in file_paths:
        print(f'Cleaning file: {file_path}')
        if file_path.endswith('.DS_Store'):
            tools.delete_file(file_path)
            continue
        data = tools.read_file(file_path)
        data = re.sub(r"^sep=.*\n", "", data, flags=re.MULTILINE)
        data = re.sub(r'\xa0', ' ', data)
        data = data.replace("'", '')
        data = data.replace(" +0000", '')
        tools.write_to_file(file_path, data)


def _filter_time(df: pd.DataFrame) -> pd.DataFrame:
    df['start_time'] = pd.to_datetime(df['start_time'])

    # Extract the time portion of the 'startDate' column
    df['time'] = df['start_time'].dt.time

    # Define the start and end time for the range you want to remove
    start_time = pd.to_datetime('05:00:00').time()
    end_time = pd.to_datetime('16:00:00').time()

    # Filter out rows where the time is between start_time and end_time
    df = df.loc[df['time'].between(start_time, end_time)].copy()  # Use .loc[] and create a copy

    # Format 'startDate' and drop 'time' column
    df['start_time'] = df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.drop(columns=['time'], inplace=True)

    return df


def _update_sleep(file_path, output_path):
    print(f'Updating sleep: {file_path}')
    dfs = []

    new_df = pd.read_csv(file_path)
    new_df.rename({
        'startDate': 'start_time',
        'endDate': 'end_time',
        'value': 'sleep_stage_actual'
    }, axis=1, inplace=True)

    if os.path.isfile(output_path):
        base_df = pd.read_csv(output_path)
        dfs.append(base_df)
        cols = base_df.columns
        new_df = new_df[cols]

    if 'sourceName' in new_df.columns:
        new_df = new_df[new_df['sourceName'].str.contains('Apple Watch', case=False, na=False)]
    dfs.append(new_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.sort_values(by=['start_time'], inplace=True)
    tools.delete_file(file_path)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved sleep to: {output_path}')


def _update_variability(file_path, output_path):
    print(f'Updating HRV: {file_path}')
    dfs = []

    new_df = pd.read_csv(file_path)
    new_df.rename({
        'startDate': 'start_time',
        'endDate': 'end_time',
        'value': 'hrv_actual'
    }, axis=1, inplace=True)

    if os.path.isfile(output_path):
        base_df = pd.read_csv(output_path)
        dfs.append(base_df)
        cols = base_df.columns
        new_df = new_df[cols]

    dfs.append(new_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.sort_values(by=['start_time'], inplace=True)
    tools.delete_file(file_path)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved HRV to: {output_path}')


def _update_heart_rate(file_path, output_path):
    print(f'Updating heart rate: {file_path}')
    dfs = []

    new_df = pd.read_csv(file_path)
    new_df.rename({
        'startDate': 'start_time',
        'endDate': 'end_time',
        'value': 'heart_rate_actual'
    }, axis=1, inplace=True)

    if os.path.isfile(output_path):
        base_df = pd.read_csv(output_path)
        dfs.append(base_df)
        cols = base_df.columns
        new_df = new_df[cols]

    dfs.append(new_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.sort_values(by=['start_time'], inplace=True)
    tools.delete_file(file_path)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved heart rate to: {output_path}')


def _update_breath_rate(file_path, output_path):
    print(f'Updating breath: {file_path}')
    dfs = []

    new_df = pd.read_csv(file_path)
    new_df.rename({
        'startDate': 'start_time',
        'endDate': 'end_time',
        'value': 'breathing_rate_actual'
    }, axis=1, inplace=True)

    if os.path.isfile(output_path):
        base_df = pd.read_csv(output_path)
        dfs.append(base_df)
        cols = base_df.columns
        new_df = new_df[cols]

    dfs.append(new_df)
    tools.delete_file(file_path)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.sort_values(by=['start_time'], inplace=True)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved breath rate to: {output_path}')

# endregion
# ---------------------------------------------------------------------------------------------------


class DataManager:
    piezo_df: pd.DataFrame

    def __init__(self, name: Name, load=True, new=False):
        self.folder_path: str = f'{PROJECT_FOLDER_PATH}data/people/{name}/'
        self.raw_folder: str = f'{self.folder_path}raw/'
        self.raw_folder_loaded: str = f'{self.raw_folder}loaded/'
        self.validation_folder: str = f'{self.folder_path}validation/'
        self.name = name

        self.piezo_df_file_path: str = f'{self.raw_folder}{name}_piezo_df.feather'
        self.heart_rate_file_path: str = f'{self.validation_folder}{name}_heart_rate.csv'
        self.breath_rate_file_path: str = f'{self.validation_folder}{name}_breath_rate.csv'
        self.hrv_file_path: str = f'{self.validation_folder}{name}_hrv.csv'
        self.sleep_data_file_path: str = f'{self.validation_folder}{name}_sleep.csv'

        sleep_info = SLEEP_DATA[name]
        self.sleep_periods: List[TimePeriod] = sleep_info['sleep_periods']
        self.sensor_count: int = sleep_info['sensor_count']
        self.validation_format: ValidationFormat = sleep_info['validation_format']
        self.raw_format: RawFormat = sleep_info['raw_format']

        # Load the validation data
        if not new:
            self.heart_rate_df: pd.DataFrame = pd.read_csv(self.heart_rate_file_path)
            self.heart_rate_df['start_time'] = pd.to_datetime(self.heart_rate_df['start_time'])
            self.hrv_df: pd.DataFrame = pd.read_csv(self.hrv_file_path)
            self.hrv_df['start_time'] = pd.to_datetime(self.hrv_df['start_time'])

        if self.validation_format == 'apple_watch' and not new:
            self.breath_rate_df: pd.DataFrame = pd.read_csv(self.breath_rate_file_path)
            self.sleep_df: pd.DataFrame = pd.read_csv(self.sleep_data_file_path)
            self.breath_rate_df['start_time'] = pd.to_datetime(self.breath_rate_df['start_time'])
            self.sleep_df['start_time'] = pd.to_datetime(self.sleep_df['start_time'])
        else:
            self.breath_rate_df = pd.DataFrame()
            self.sleep_df = pd.DataFrame()

        if load and not new:
            self.load_piezo_df()


    def load_new_raw_data(self):
        print('Checking for new raw files...')
        file_paths = tools.list_dir_files(f'{self.raw_folder}/load', full_path=True)
        file_paths = [file for file in file_paths if file.endswith('.RAW')]
        for file_path in file_paths:
            data = load_raw_data(file_path=file_path, piezo_only=True)
            if len(data['piezo_dual']) == 0:
                tools.delete_file(file_path)
            else:
                start_time = data['piezo_dual'][0]['ts']
                end_timestamp = data['piezo_dual'][-1]['ts']
                new_file_name = f'{start_time.replace(':', '.')}___{end_timestamp.replace(':', '.')}.RAW'
                new_folder_path = tools.create_folder(f'{self.raw_folder_loaded}{end_timestamp[:-9]}/')
                new_file_path = tools.rename_file(file_path, new_file_name)
                to_file_path = Path.joinpath(Path(new_folder_path), os.path.basename(new_file_path))
                if not os.path.isfile(to_file_path):
                    tools.move_file(new_file_path, to_folder_path=new_folder_path)
                else:
                    tools.delete_file(new_file_path)

        self._update_piezo_df()


    def _load_new_apple_watch_validation_data(self):
        print(f'Loading apple watch validation data for: {self.name}')
        file_paths = tools.list_dir_files(
            f'{self.validation_folder}load/',
            full_path=True
        )

        _clean_files(file_paths)
        for file_path in file_paths:
            if file_path.endswith('.DS_Store'):
                continue
            if 'heartratevariability' in file_path.lower():
                _update_variability(file_path, self.hrv_file_path)
            elif 'heartrate' in file_path.lower():
                _update_heart_rate(file_path, self.heart_rate_file_path)
            elif 'respiratory' in file_path.lower():
                _update_breath_rate(file_path, self.breath_rate_file_path)
            elif 'sleep' in file_path.lower():
                _update_sleep(file_path, self.sleep_data_file_path)

        # Load the dataframes
        self.heart_rate_df: pd.DataFrame = pd.read_csv(self.heart_rate_file_path)
        self.breath_rate_df: pd.DataFrame = pd.read_csv(self.breath_rate_file_path)
        self.hrv_df: pd.DataFrame = pd.read_csv(self.hrv_file_path)
        self.sleep_df: pd.DataFrame = pd.read_csv(self.sleep_data_file_path)


    def _load_new_polar_validation_data(self):
        df = pd.read_csv(f'{self.validation_folder}/load/hr+hrv-polar-h10.csv', sep=';')
        df.rename({
            'Phone timestamp': 'start_time',
            'HR [bpm]': 'heart_rate',
            'HRV [ms]': 'hrv',
        }, axis=1, inplace=True)
        heart_rate_df = df[['start_time', 'heart_rate']]
        hrv_df = df[['start_time', 'hrv']]
        heart_rate_df.to_csv(self.heart_rate_file_path, index=False)
        hrv_df.to_csv(self.hrv_file_path, index=False)

        return df


    def load_new_validation_data(self):
        if self.validation_format == 'apple_watch':
            self._load_new_apple_watch_validation_data()
        elif self.validation_format == 'polar':
            self._load_new_polar_validation_data()


    def load_raw_data(self) -> Data:
        """Loads the raw data from 8 sleep"""
        return load_raw_data(folder_path=self.raw_folder_loaded)


    def load_piezo_df(self):
        piezo_df = pd.read_feather(self.piezo_df_file_path)
        self.piezo_df = piezo_df
        return piezo_df


    def _load_pkl_piezo(self) -> pd.DataFrame:
        file_paths = tools.list_dir_files(f'{self.raw_folder}/loaded', full_path=True)
        dfs = []
        for file_path in file_paths:
            dfs.append(pd.read_pickle(file_path))
        df = pd.concat(dfs)
        return df


    def _load_raw_piezo(self) -> pd.DataFrame:
        raw_data = load_raw_data(folder_path=self.raw_folder_loaded, piezo_only=True)
        return pd.DataFrame(raw_data['piezo_dual'])


    def _update_piezo_df(self):
        if self.raw_format == 'raw':
            self.piezo_df = self._load_raw_piezo()
        else:
            self.piezo_df = self._load_pkl_piezo()

        # Ensure ts column is in datetime format before setting index
        self.piezo_df['ts'] = pd.to_datetime(self.piezo_df['ts'])

        # Sort and set index
        self.piezo_df.sort_values(by='ts', ascending=True, inplace=True)
        self.piezo_df.set_index('ts', inplace=True)

        # Trim data to sleep periods
        self.piezo_df = self._trim_piezo_df()

        # Save after trimming to ensure only relevant data is saved
        print(f'Saving df to feather: {self.piezo_df_file_path}')
        self.piezo_df.to_feather(self.piezo_df_file_path)

        return self.piezo_df


    def _trim_piezo_df(self):
        print('Trimming piezo df to only sleep period data...')

        t0_row_count = self.piezo_df.shape[0]
        filtered_dfs = []

        for period in self.sleep_periods:
            start = pd.to_datetime(period['start_time'])
            end = pd.to_datetime(period['end_time'])

            # Use the index directly for filtering
            mask = (self.piezo_df.index >= start) & (self.piezo_df.index <= end)
            filtered_dfs.append(self.piezo_df.loc[mask])

        # Handle empty result case
        if filtered_dfs:
            filtered_df = pd.concat(filtered_dfs, ignore_index=False)
        else:
            filtered_df = pd.DataFrame(columns=self.piezo_df.columns)

        t1_row_count = filtered_df.shape[0]
        print(f'Deleted {t0_row_count - t1_row_count:,} extra rows | Remaining rows: {t1_row_count:,}')

        return filtered_df

