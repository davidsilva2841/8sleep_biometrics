import json
import os
import pandas as pd
import ast
import tools


def build_summary(df: pd.DataFrame):


    df.drop_duplicates(inplace=True, subset=['name', 'start_time', 'end_time', 'params_hash'])

    # Grouping statistics for filtered data
    summary_df = df.groupby('params_hash').agg(
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
    ).round(3)

    # Convert unhashable list columns to strings
    df.loc[:, 'hr_std_range'] = df['hr_std_range'].apply(lambda x: str(x) if isinstance(x, list) else x)
    df.loc[:, 'percentile'] = df['percentile'].apply(lambda x: str(x) if isinstance(x, list) else x)
    # df.loc[:, 'signal_percentile'] = df['signal_percentile'].apply(lambda x: str(x) if isinstance(x, list) else x)

    param_columns = [
        'params_hash', 'window', 'slide_by', 'moving_avg_size',
        'hr_std_range', 'percentile', 'r_window_avg', 'r_min_periods', 'signal_percentile'
    ]


    # Drop duplicates to get unique parameter combinations
    df_params = df.loc[:, param_columns].drop_duplicates()

    # Convert back list columns if needed
    df_params['percentile'] = df_params['percentile'].apply(ast.literal_eval)
    df_params['hr_std_range'] = df_params['hr_std_range'].apply(ast.literal_eval)

    # Merge summaries with unique parameter values
    summary_df = summary_df.merge(df_params, on='params_hash', how='left')

    return summary_df

files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/src/param_optimizer/results_old/', full_path=True)

rows = []

for f in files:
    with open(f, 'r') as file:
        json_hash = json.load(file)
        file_hash = os.path.basename(f)
        json_hash['file'] = file_hash
        rows.append(json_hash)



df = pd.DataFrame(rows)

df.dropna(subset=['percentile'], inplace=True)

df['corr'] = df['corr'].str.rstrip('%').astype(float) / 100

counts = df['params_hash'].value_counts()

# Get params_hash values that appear at least 20 times
filtered_hashes = counts[counts >= 20].index.tolist()

# Ensure filtered_df contains only valid hashes
filtered_df = df[df['params_hash'].isin(filtered_hashes)].copy()

counts = df['params_hash'].value_counts()

# Get params_hash values that appear at least 20 times
filtered_hashes = counts[counts >= 20].index.tolist()  # Convert index to list

summary_df = build_summary(filtered_df)



best = df[df['params_hash'] == '35cde4e0dfe960f9e49fc90ce801aa4f784014ef']
sum_selected = summary_df[summary_df['params_hash'] == '0831c421c3c21764421c3cabb4511b56a7527ba0']

top_100_df = summary_df.sort_values(by='mean_rmse', ascending=True).head(100)
top_100_df['window'].value_counts()
top_100_df['hr_std_range'].value_counts()
top_100_df['percentile'].value_counts()
top_100_df['signal_percentile'].value_counts()
top_100_df['moving_avg_size'].value_counts()
top_100_df['r_window_avg'].value_counts()
top_100_df['r_min_periods'].value_counts()
top_100_df['r_min_periods'].value_counts()


