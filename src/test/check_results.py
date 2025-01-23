import json
import os
import os.path

import tools
import pandas as pd
import ast
# files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/src/test/results', full_path=True)
files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/src/test/results', full_path=True)





def build_summary(filtered_df: pd.DataFrame):
    # Grouping statistics for filtered data
    summary_df = filtered_df.groupby('params_hash').agg(
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

    param_columns = [
        'params_hash', 'window', 'slide_by', 'moving_avg_size',
        'hr_std_range', 'percentile', 'r_window_avg', 'r_min_periods'
    ]

    # Convert unhashable list columns to strings
    filtered_df.loc[:, 'hr_std_range'] = filtered_df['hr_std_range'].apply(lambda x: str(x) if isinstance(x, list) else x)
    filtered_df.loc[:, 'percentile'] = filtered_df['percentile'].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Drop duplicates to get unique parameter combinations
    df_params = filtered_df.loc[:, param_columns].drop_duplicates()

    # Convert back list columns if needed
    df_params['hr_std_range'] = df_params['hr_std_range'].apply(ast.literal_eval)
    df_params['percentile'] = df_params['percentile'].apply(ast.literal_eval)

    # Merge summaries with unique parameter values
    summary_df = summary_df.merge(df_params, on='params_hash', how='left')

    return summary_df

rows = []

for f in files:
    with open(f, 'r') as file:
        json_hash = json.load(file)
        file_hash = os.path.basename(f)
        json_hash['file'] = file_hash
        rows.append(json_hash)


df = pd.DataFrame(rows)
# df.drop_duplicates(subset=['column1', 'column2'], keep='first', inplace=True)
df['corr'] = df['corr'].str.rstrip('%').astype(float) / 100

counts = df['params_hash'].value_counts()

# Get params_hash values that appear at least 20 times
filtered_hashes = counts[counts >= 20].index.tolist()

# Ensure filtered_df contains only valid hashes
filtered_df = df[df['params_hash'].isin(filtered_hashes)].copy()

counts = df['params_hash'].value_counts()

# Get params_hash values that appear at least 9 times
filtered_hashes = counts[counts >= 20].index.tolist()  # Convert index to list

summary_df = build_summary(filtered_df)



best = df[df['params_hash'] == 'df61844719ffdd6a3156722e548ada2e53f0d1d7']
sum_selected = summary_df[summary_df['params_hash'] == '0831c421c3c21764421c3cabb4511b56a7527ba0']

top_100_df = summary_df.sort_values(by='mean_rmse', ascending=True).head(100)
top_100_df['hr_std_range'].value_counts()
top_100_df['window'].value_counts()
top_100_df['percentile'].value_counts()
top_100_df['moving_avg_size'].value_counts()
top_100_df['r_window_avg'].value_counts()
top_100_df['r_min_periods'].value_counts()




# top_100_df.to_csv('/Users/ds/main/8sleep_biometrics/src/test/aggregated_results.csv', index=False)
# df.to_csv('/Users/ds/main/8sleep_biometrics/src/test/all_results.csv', index=False)
