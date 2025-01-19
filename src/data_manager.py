from typing import List
import os
from pathlib import Path
import re
import pandas as pd

from toolkit import tools
from src.load_raw import load_raw_data
from src.sleep_data import SLEEP_DATA, TimePeriod



def _clean_files(file_paths):
    for file_path in file_paths:
        data = tools.read_file(file_path)
        data = re.sub(r"^sep=.*\n", "", data, flags=re.MULTILINE)
        data = re.sub(r'\xa0', ' ', data)
        data = data.replace("'", '')
        data = data.replace(" +0000", '')
        tools.write_to_file(file_path, data)



def _filter_time(df: pd.DataFrame) -> pd.DataFrame:
    df['startDate'] = pd.to_datetime(df['startDate'])

    # Extract the time portion of the 'startDate' column
    df['time'] = df['startDate'].dt.time

    # Define the start and end time for the range you want to remove
    start_time = pd.to_datetime('05:00:00').time()
    end_time = pd.to_datetime('16:00:00').time()

    # Filter out rows where the time is between start_time and end_time
    df = df.loc[df['time'].between(start_time, end_time)].copy()  # Use .loc[] and create a copy

    # Format 'startDate' and drop 'time' column
    df['startDate'] = df['startDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.drop(columns=['time'], inplace=True)

    return df


def _update_sleep(file_path, output_path):
    print(f'Updating sleep: {file_path}')
    base_df = pd.read_csv(output_path)
    new_df = pd.read_csv(file_path)
    cols = base_df.columns
    new_df = new_df[cols]
    new_df = _filter_time(new_df)
    new_df = new_df[new_df['sourceName'].str.contains('Apple Watch', case=False, na=False)]
    combined_df = pd.concat([base_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.sort_values(by=['startDate'], inplace=True)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved sleep to: {output_path}')


def _update_variability(file_path, output_path):
    print(f'Updating HRV: {file_path}')
    base_df = pd.read_csv(output_path)
    new_df = pd.read_csv(file_path)
    cols = base_df.columns
    new_df = new_df[cols]
    new_df = _filter_time(new_df)
    combined_df = pd.concat([base_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.sort_values(by=['startDate'], inplace=True)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved HRV to: {output_path}')


def _update_heart_rate(file_path, output_path):
    print(f'Updating heart rate: {file_path}')
    base_df = pd.read_csv(output_path)
    new_df = pd.read_csv(file_path)
    cols = base_df.columns
    new_df = new_df[cols]
    new_df = _filter_time(new_df)
    combined_df = pd.concat([base_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved heart rate to: {output_path}')



def _update_breath_rate(file_path, output_path):
    print(f'Updating breath: {file_path}')
    base_df = pd.read_csv(output_path)
    new_df = pd.read_csv(file_path)
    cols = base_df.columns
    new_df = new_df[cols]
    new_df = _filter_time(new_df)
    combined_df = pd.concat([base_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(inplace=True)
    combined_df.to_csv(output_path, index=False)
    print(f'Saved breath rate to: {output_path}')






class DataManager:
    piezo_df: pd.DataFrame

    def __init__(self, name: str, load=True):
        self.folder_path: str = f'/Users/ds/main/8sleep_biometrics/data/people/{name}/'
        self.raw_folder: str = f'{self.folder_path}raw/'
        self.raw_folder_loaded: str = f'{self.raw_folder}loaded/'
        self.validation_folder: str = f'{self.folder_path}validation/'
        self.name = name

        self.piezo_df_file_path: str = f'{self.raw_folder}{name}_piezo_df.parquet'
        self.heart_rate_file_path: str = f'{self.validation_folder}{name}_heart_rate.csv'
        self.breath_rate_file_path: str = f'{self.validation_folder}{name}_breath_rate.csv'
        self.hrv_file_path: str = f'{self.validation_folder}{name}_hrv.csv'
        self.sleep_data_file_path: str = f'{self.validation_folder}{name}_sleep.csv'

        # Load the dataframes
        self.heart_rate_df: pd.DataFrame = pd.read_csv(self.heart_rate_file_path)
        self.breath_rate_df: pd.DataFrame = pd.read_csv(self.breath_rate_file_path)
        self.hrv_df: pd.DataFrame = pd.read_csv(self.hrv_file_path)
        self.sleep_df: pd.DataFrame = pd.read_csv(self.sleep_data_file_path)

        self.heart_rate_df.sort_values(by='startDate', inplace=True)
        self.breath_rate_df.sort_values(by='startDate', inplace=True)
        self.hrv_df.sort_values(by='startDate', inplace=True)
        self.sleep_df.sort_values(by='startDate', inplace=True)
        self.sleep_periods: List[TimePeriod] = SLEEP_DATA[name]['sleep_periods']

        if load:
            self.load_piezo_df()



    def load_new_raw_data(self):
        file_paths = tools.list_dir_files(f'{self.raw_folder}/load', full_path=True)
        file_paths = [file for file in file_paths if file.endswith('.RAW')]
        for file_path in file_paths:
            data = load_raw_data(file_path=file_path, piezo_only=True)
            if len(data['piezo_dual']) == 0:
                tools.delete_file(file_path)
            else:
                start_time = data['piezo_dual'][0]['ts']
                end_timestamp = data['piezo_dual'][-1]['ts']
                end_time = end_timestamp[-8:]
                if end_time < '14:30:00' and end_time > '06:00:00':
                    new_file_name = f'{start_time.replace(':', '.')}___{end_timestamp.replace(':', '.')}.RAW'
                    new_folder_path = tools.create_folder(f'{self.raw_folder_loaded}{end_timestamp[:-9]}/')
                    new_file_path = tools.rename_file(file_path, new_file_name)
                    to_file_path = Path.joinpath(Path(new_folder_path), os.path.basename(new_file_path))
                    if not os.path.isfile(to_file_path):
                        tools.move_file(new_file_path, to_folder_path=new_folder_path)
                    else:
                        tools.delete_file(new_file_path)
                else:
                    tools.delete_file(file_path)
                    print('-----------------------------------------------------------------------------------------------------')
                    print(f'Skipping time range: {start_time} -> {end_timestamp}')
                    print(f'data_manager.py:141 end_time: {end_time}')


    def load_new_validation_data(self):
        file_paths = tools.list_dir_files(
            f'{self.validation_folder}load/',
            full_path=True
        )
        _clean_files(file_paths)
        for file_path in file_paths:
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


    def load_raw_data(self):
        return load_raw_data(folder_path=self.raw_folder_loaded)


    def load_piezo_df(self):
        piezo_df = pd.read_feather(self.piezo_df_file_path)
        piezo_df['ts'] = pd.to_datetime(piezo_df['ts'])
        piezo_df.set_index('ts', inplace=True)
        self.piezo_df = piezo_df
        return piezo_df


    def update_piezo_df(self):
        raw_data = load_raw_data(folder_path=self.raw_folder_loaded, piezo_only=True)
        piezo_df = pd.DataFrame(raw_data['piezo_dual'])

        piezo_df.sort_values(by='ts', ascending=True, inplace=True)
        print(f'Saving df to feather: {self.piezo_df_file_path}')
        piezo_df.to_feather(self.piezo_df_file_path)
        piezo_df['ts'] = pd.to_datetime(piezo_df['ts'])
        piezo_df.set_index('ts', inplace=True)

        self.piezo_df = piezo_df

        return piezo_df


