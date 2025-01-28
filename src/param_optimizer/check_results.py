import json
import os
import pandas as pd
import ast
import tools

param_columns = [
    'params_hash',
    'side',
    'window',
    'slide_by',
    'moving_avg_size',
    'hr_std_range',
    'hr_percentile',
    'r_window_avg',
    'r_min_periods',
    'signal_percentile',
    'key'
]


def build_summary(filtered_df: pd.DataFrame):
    # Grouping statistics for filtered data
    summary_df = filtered_df.groupby('key').agg(
        mean_corr=('corr', 'mean'),
        min_corr=('corr', 'min'),
        max_corr=('corr', 'max'),
        std_corr=('corr', 'std'),
        mean_mae=('mae', 'mean'),
        min_mae=('mae', 'min'),
        max_mae=('mae', 'max'),
        std_mae=('mae', 'std'),
        mean_rmse=('rmse', 'mean'),
        min_rmse=('rmse', 'min'),
        max_rmse=('rmse', 'max'),
        std_rmse=('rmse', 'std')
    ).round(3).reset_index()


    # Convert unhashable list columns to strings
    filtered_df['hr_std_range'] = filtered_df['hr_std_range'].apply(lambda x: str(x) if isinstance(x, list) else x)
    filtered_df['hr_percentile'] = filtered_df['hr_percentile'].apply(lambda x: str(x) if isinstance(x, list) else x)
    filtered_df['signal_percentile'] = filtered_df['signal_percentile'].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Drop duplicates to get unique parameter combinations
    df_params = filtered_df[param_columns].drop_duplicates()

    # Convert back list columns if needed
    df_params['hr_percentile'] = df_params['hr_percentile'].apply(ast.literal_eval)
    df_params['hr_std_range'] = df_params['hr_std_range'].apply(ast.literal_eval)

    # Merge summaries with unique parameter values
    summary_df = summary_df.merge(df_params, on=['key'], how='left')

    return summary_df

files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/tmp/param_optimizer/results', full_path=True)

rows = []

for f in files:
    with open(f, 'r') as file:
        json_hash = json.load(file)
        file_hash = os.path.basename(f)
        json_hash['file'] = file_hash
        rows.append(json_hash)


df = pd.DataFrame(rows)
df.dropna(subset=['hr_percentile'], inplace=True)
df.dropna(subset=['side'],inplace=True)

df['key'] = df['params_hash'] + '__' + df['side']
df['corr'] = df['corr'].str.rstrip('%').astype(float) / 100
unique_columns = [
    'key',
    'name',
    'start_time',
    'end_time',
]
df.drop_duplicates(subset=unique_columns, inplace=True)

counts = df['key'].value_counts()

# Get params_hash values that appear at least 20 times
filtered_hashes = counts[counts >= 15].index.tolist()

# Ensure filtered_df contains only valid hashes
filtered_df = df[df['key'].isin(filtered_hashes)].copy()

summary_df = build_summary(filtered_df)
top_100_df = summary_df.sort_values(by='mean_rmse', ascending=True).head(100)


top_100_df['window'].value_counts()
top_100_df['hr_std_range'].value_counts()
top_100_df['hr_percentile'].value_counts()
top_100_df['signal_percentile'].value_counts()
top_100_df['moving_avg_size'].value_counts()
top_100_df['r_window_avg'].value_counts()
top_100_df['r_min_periods'].value_counts()
best = df[df['key'] == '7a18d1861109abded99dc0b8878ba0d1c863e936__combined']
sum_selected = summary_df[summary_df['params_hash'] == '0831c421c3c21764421c3cabb4511b56a7527ba0']

