import json
import tools
import pandas as pd
import ast
# files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/src/test/results', full_path=True)
files = tools.list_dir_files('/Users/ds/main/8sleep_biometrics/src/test/results', full_path=True)
rows = []




def build_summary(df: pd.DataFrame):

    # Perform the calculations after filtering
    # Calculate statistics for 'corr'
    mean_corr = filtered_df.groupby('params_hash')['corr'].mean().rename('mean_corr')
    min_corr = filtered_df.groupby('params_hash')['corr'].min().rename('min_corr')
    max_corr = filtered_df.groupby('params_hash')['corr'].max().rename('max_corr')
    std_corr = filtered_df.groupby('params_hash')['corr'].std().rename('std_corr')

    # Calculate statistics for 'mae'
    mean_mae = filtered_df.groupby('params_hash')['mae'].mean().rename('mean_mae')
    min_mae = filtered_df.groupby('params_hash')['mae'].min().rename('min_mae')
    max_mae = filtered_df.groupby('params_hash')['mae'].max().rename('max_mae')
    std_mae = filtered_df.groupby('params_hash')['mae'].std().rename('std_mae')

    # Calculate statistics for 'rmse'
    mean_rmse = filtered_df.groupby('params_hash')['rmse'].mean().rename('mean_rmse')
    min_rmse = filtered_df.groupby('params_hash')['rmse'].min().rename('min_rmse')
    max_rmse = filtered_df.groupby('params_hash')['rmse'].max().rename('max_rmse')
    std_rmse = filtered_df.groupby('params_hash')['rmse'].std().rename('std_rmse')

    # Combine all calculated statistics into a single DataFrame
    summary_df = pd.concat(
        [
            mean_corr, min_corr, max_corr, std_corr,
            mean_mae, min_mae, max_mae, std_mae,
            mean_rmse, min_rmse, max_rmse, std_rmse
        ],
        axis=1
    )

    # Round all values to 3 decimal places
    summary_df = summary_df.round(3)

    param_columns = ['params_hash', 'window', 'slide_by', 'moving_avg_size',
                     'hr_std_range', 'percentile', 'r_window_avg', 'r_min_periods']

    # Convert unhashable list columns to strings to enable deduplication
    df['hr_std_range'] = df['hr_std_range'].apply(lambda x: str(x) if isinstance(x, list) else x)
    df['percentile'] = df['percentile'].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Drop duplicate rows based on 'params_hash'
    df_params = df[param_columns].drop_duplicates()

    # Convert back to original format if needed
    df_params['hr_std_range'] = df_params['hr_std_range'].apply(eval)
    df_params['percentile'] = df_params['percentile'].apply(eval)

    # Merge the cleaned parameters with the summary DataFrame
    summary_df = summary_df.merge(df_params, on='params_hash', how='left')
    return summary_df

for f in files:
    with open(f, 'r') as file:
        rows.append(json.load(file))


df = pd.DataFrame(rows)
df['corr'] = df['corr'].str.rstrip('%').astype(float) / 100

counts = df['params_hash'].value_counts()

# Get params_hash values that appear at least 9 times
filtered_hashes = counts[counts >= 14].index.tolist()  # Convert index to list

# Debug: Check unique counts and selected params_hash values
print("Total unique 'params_hash' values:", len(counts))
print("Filtered hash count (≥9 occurrences):", len(filtered_hashes))

# Ensure data types are consistent and filter the DataFrame
df['params_hash'] = df['params_hash'].astype(str).str.strip()  # Ensure no leading/trailing spaces
filtered_df = df[df['params_hash'].isin(filtered_hashes)]

summary_df = build_summary(df)

# Check if filtering worked
if filtered_df.empty:
    print("No rows match the filtering criteria. Check for inconsistent data formats.")
else:
    print("Filtered DataFrame size:", filtered_df.shape)
best = df[df['params_hash'] == 'e4fc1070cf14d1b45469cfa4d3d4ea2a5816bd64']
sum_selected = summary_df[summary_df['params_hash'] == '0831c421c3c21764421c3cabb4511b56a7527ba0']

top_100_df = summary_df.sort_values(by='mean_rmse', ascending=True).head(100)
top_100_df['hr_std_range'].value_counts()
top_100_df['window'].value_counts()
top_100_df['percentile'].value_counts()
top_100_df['moving_avg_size'].value_counts()
top_100_df['r_window_avg'].value_counts()
top_100_df['r_min_periods'].value_counts()




top_100_df.to_csv('/Users/ds/main/8sleep_biometrics/src/test/aggregated_results.csv', index=False)
df.to_csv('/Users/ds/main/8sleep_biometrics/src/test/all_results.csv', index=False)
