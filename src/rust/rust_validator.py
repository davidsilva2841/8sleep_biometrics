from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor
import subprocess
from pathlib import Path
import json
import hashlib
import time
from typing import Dict, List, Any

from data_manager import DataManager

class ParameterSpace:
    """Defines the parameter space to search through"""
    def __init__(self, test_mode=False):
        if test_mode:
            # Smaller parameter space for testing
            self.params = {
                'hr_window_seconds': [10.0],
                'hr_window_overlap': [0.1],
                'br_window_seconds': [120.0],
                'br_window_overlap': [0.0],
                'harmonic_penalty_close': [0.8],
                'harmonic_penalty_far': [0.5],
                'hr_smoothing_window': [60],
                'hr_smoothing_strength': [0.25]
            }
        else:
            # Full parameter space
            self.params = {
                'hr_window_seconds': [10.0, 15.0, 20.0, 30.0],
                'hr_window_overlap': [0.1, 0.33, 0.67, 0.9],
                'br_window_seconds': [120.0],
                'br_window_overlap': [0.0],
                'harmonic_penalty_close': [0.6, 0.7, 0.8, 0.9],
                'harmonic_penalty_far': [0.3, 0.4, 0.5, 0.6],
                'hr_smoothing_window': [10, 30, 60],
                'hr_smoothing_strength': [0.25],
                'hr_history_window': [30, 60, 120, 180, 240],
                'hr_outlier_percentile': [0.01, 0.033, 0.067, 0.10]
            }

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations"""
        keys = self.params.keys()
        values = self.params.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    @staticmethod
    def generate_params_hash(params: Dict[str, Any]) -> str:
        """Generate a hash of the parameters for unique identification"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha1(param_str.encode()).hexdigest()

class RustValidator:
    """Manages validation runs of the Rust binary"""
    def __init__(self, data_manager: DataManager, rust_binary_path: str):
        self.data_manager = data_manager
        self.rust_binary_path = rust_binary_path
        self.results = []

    def build_command(self, params: Dict[str, Any], input_path: str, output_path: str) -> str:
        """Construct the Rust command with parameters"""
        # Format timestamps to YYYY-MM-DD HH:MM
        start_time = params.get('start_time', '2025-01-01 00:00:00')[:16]  # Strip seconds
        end_time = params.get('end_time', '2025-01-31 23:59:00')[:16]     # Strip seconds

        cmd = [
            self.rust_binary_path,
            input_path,
            "--feather-input",
            f"--csv-output={output_path}",
            f"--start-time=\"{start_time}\"",
            f"--end-time=\"{end_time}\"",
            f"--hr-window-seconds={params['hr_window_seconds']}",
            f"--hr-window-overlap={params['hr_window_overlap']}",
            f"--br-window-seconds={params['br_window_seconds']}",
            f"--br-wcecindow-overlap={params['br_window_overlap']}",
            f"--harmonic-penalty-close={params['harmonic_penalty_close']}",
            f"--harmonic-penalty-far={params['harmonic_penalty_far']}",
            f"--hr-smoothing-window={params['hr_smoothing_window']}",
            f"--hr-smoothing-strength={params['hr_smoothing_strength']}",
            f"--hr-history-window={params['hr_history_window']}",
            f"--hr-outlier-percentile={params['hr_outlier_percentile']}"
        ]

        return " ".join(str(x) for x in cmd)

    def calculate_metrics(self, df_pred: pd.DataFrame, period: Dict[str, str]) -> Dict[str, Any]:
        """Calculate metrics between predictions and ground truth data"""
        try:
            # Convert timestamps to datetime
            df_pred['start_time'] = pd.to_datetime(df_pred['start_time'])

            # Get ground truth data for this period
            ground_truth = self.data_manager.heart_rate_df.copy()
            ground_truth['start_time'] = pd.to_datetime(ground_truth['start_time'])
            ground_truth['heart_rate_actual'] = ground_truth['heart_rate_actual']

            # Filter ground truth data to period
            period_start = pd.to_datetime(period['start_time'])
            period_end = pd.to_datetime(period['end_time'])
            ground_truth = ground_truth[
                (ground_truth['start_time'] >= period_start) &
                (ground_truth['start_time'] <= period_end)
                ]

            # Merge predictions with ground truth
            merged = pd.merge_asof(
                ground_truth,
                df_pred[['start_time', 'heart_rate']],
                on='start_time',
                direction='nearest',
                tolerance=pd.Timedelta('30s')
            )

            # Drop rows with missing values
            merged = merged.dropna(subset=['heart_rate', 'heart_rate_actual'])

            if merged.empty:
                raise ValueError("No overlapping data points found between predictions and ground truth")

            # Calculate metrics exactly as in analyze.py

            pre_merge_count = len(df_pred)
            post_merge_count = len(merged)

            return {
                'mean': round(merged['heart_rate'].mean(), 2),
                'std': round(merged['heart_rate'].std(), 2),
                'min': round(merged['heart_rate'].min(), 2),
                'max': round(merged['heart_rate'].max(), 2),
                'corr': f'{round(merged["heart_rate_actual"].corr(merged["heart_rate"]), 2) * 100:.2f}%',
                'mae': round(mean_absolute_error(merged['heart_rate_actual'], merged['heart_rate']), 2),
                'mse': round(mean_squared_error(merged['heart_rate_actual'], merged['heart_rate']), 2),
                'mape': f"{round(np.mean(np.abs((merged['heart_rate_actual'] - merged['heart_rate']) / merged['heart_rate_actual'])) * 100, 2)}%",
                'rmse': round(np.sqrt(mean_squared_error(merged['heart_rate_actual'], merged['heart_rate'])), 2),
                'pre_merge_count': pre_merge_count,
                'post_merge_count': post_merge_count,
            }

        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")

    def run_validation(self, params: Dict[str, Any], period: Dict[str, str]) -> Dict[str, Any]:
        """Run a single validation with given parameters"""
        try:
            start_time = time.time()

            # Create temporary output directory
            param_hash = ParameterSpace.generate_params_hash(params)
            temp_dir = Path(f"temp_results/{param_hash}")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Get the date from the period's start time and construct raw data path
            period_date = period['start_time'][:10]  # Extract YYYY-MM-DD
            raw_data_path = Path(self.data_manager.raw_folder) / period_date

            # if not raw_data_path.exists():
            #     return {
            #         'params_hash': param_hash,
            #         'error': f"Skipping - No RAW data for {period_date}",
            #         'rust': params
            #     }

            # Add time range to parameters
            params_with_time = params.copy()
            params_with_time['start_time'] = period['start_time']
            params_with_time['end_time'] = period['end_time']

            # Build and run command
            cmd = self.build_command(
                params_with_time,
                self.data_manager.piezo_df_file_path,
                str(temp_dir / "output")
            )
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=False,
                    # stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(result.stdout)
                print(result.stderr)
                # Check if the process completed successfully
                if result.returncode != 0:
                    return {
                        'params_hash': param_hash,
                        'error': f"Skipping - Process failed with code {result.returncode}",
                        'rust': params
                    }

                # Get the side the user slept on from the period data
                side = period.get('side', 'right').lower()
                output_file = temp_dir / f"output_{side}_combined_period_0.csv"

                if not output_file.exists():
                    other_side = 'left' if side == 'right' else 'right'
                    other_file = temp_dir / f"output_{other_side}_combined_period_0.csv"
                    if other_file.exists():
                        output_file = other_file
                    else:
                        return {
                            'params_hash': param_hash,
                            'error': f"Skipping - No output file found",
                            'rust': params
                        }

            except Exception as e:
                return {
                    'params_hash': param_hash,
                    'error': f"Skipping - {str(e)}",
                    'rust': params
                }

            try:
                # Load predictions
                df_pred = pd.read_csv(output_file)

                # Rename columns to match expected format
                df_pred = df_pred.rename(columns={
                    'fft_hr_smoothed': 'heart_rate',
                    'timestamp': 'start_time'
                })

                # Calculate metrics directly
                metrics = self.calculate_metrics(df_pred, period)

                # Add additional information to results
                result = {
                    'start_time': period['start_time'],
                    'end_time': period['end_time'],
                    'name': self.data_manager.name,
                    **metrics,
                    'params_hash': param_hash,
                    'rust': params
                }

                # Save individual result
                output_dir = Path('validation_results/individual')
                output_dir.mkdir(parents=True, exist_ok=True)
                result_file = output_dir / f"{result['name']}_{param_hash}_{period['start_time'].replace(' ', '_')}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=4)

                return result

            except Exception as e:
                return {
                    'params_hash': param_hash,
                    'error': f"Skipping - Error processing results: {str(e)}",
                    'rust': params
                }

            finally:
                # Clean up temporary files silently
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass

        except Exception as e:
            return {
                'params_hash': param_hash,
                'error': f"Skipping - {str(e)}",
                'rust': params
            }

    def parallel_validate(self, parameter_space: ParameterSpace, max_workers: int = 4) -> List[Dict[str, Any]]:
        """Run validations in parallel and track best results"""
        combinations = parameter_space.generate_combinations()
        results = []
        parameter_results = {}  # Track results by parameter hash
        best_avg_mae = float('inf')
        total_combinations = len(combinations)
        completed_params = 0

        print(f"\nStarting validation with {total_combinations} parameter combinations...")
        print(f"Testing each successful combination against {len(self.data_manager.sleep_periods)} periods")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process parameter sets in batches
            for batch_start in range(0, total_combinations, max_workers):
                batch_end = min(batch_start + max_workers, total_combinations)
                batch = combinations[batch_start:batch_end]
                completed_params += len(batch)

                # Test first period for all parameter sets in this batch
                first_period = self.data_manager.sleep_periods[0]
                first_period_futures = [
                    executor.submit(self.run_validation, params, first_period)
                    for params in batch
                ]

                # Process results and launch remaining periods for successful parameter sets
                remaining_futures = {}  # Map param_hash to list of futures for remaining periods
                for future in first_period_futures:
                    first_result = future.result()
                    param_hash = first_result['params_hash']

                    # Skip parameter sets that fail the first period
                    if 'error' in first_result:
                        continue

                    # Initialize results tracking for this parameter set
                    parameter_results[param_hash] = {
                        'results': [first_result],
                        'params': first_result['rust'],
                        'has_errors': False
                    }

                    # Launch remaining periods for this parameter set
                    remaining_futures[param_hash] = [
                        executor.submit(self.run_validation, first_result['rust'], period)
                        for period in self.data_manager.sleep_periods[1:]
                    ]

                # Process remaining period results as they complete
                for param_hash, futures in remaining_futures.items():
                    for future in futures:
                        result = future.result()
                        if 'error' in result:
                            parameter_results[param_hash]['has_errors'] = True
                            break
                        parameter_results[param_hash]['results'].append(result)

                    # If all periods succeeded, calculate metrics
                    if not parameter_results[param_hash]['has_errors']:
                        results_for_params = parameter_results[param_hash]['results']
                        avg_metrics = {
                            'avg_mae': sum(r['mae'] for r in results_for_params) / len(results_for_params),
                            'avg_mse': sum(r['mse'] for r in results_for_params) / len(results_for_params),
                            'avg_rmse': sum(r['rmse'] for r in results_for_params) / len(results_for_params),
                            'avg_corr': sum(float(r['corr'].strip('%')) for r in results_for_params) / len(results_for_params),
                            'worst_mae': max(r['mae'] for r in results_for_params),
                            'best_mae': min(r['mae'] for r in results_for_params),
                            'periods_tested': len(results_for_params)
                        }

                        # Save aggregate results
                        aggregate_result = {
                            'params_hash': param_hash,
                            'rust': parameter_results[param_hash]['params'],
                            **avg_metrics
                        }

                        results.append(aggregate_result)

                        # Check if this is a new best result
                        if avg_metrics['avg_mae'] < best_avg_mae:
                            best_avg_mae = avg_metrics['avg_mae']
                            print(f"\nNew best average result ({completed_params}/{total_combinations}):")
                            print(f"Average MAE: {best_avg_mae:.2f}")
                            print(f"Worst MAE: {avg_metrics['worst_mae']:.2f}")
                            print(f"Best MAE: {avg_metrics['best_mae']:.2f}")
                            print(f"Average Correlation: {avg_metrics['avg_corr']:.2f}%")
                            print(f"Parameters: {json.dumps(parameter_results[param_hash]['params'], indent=2)}")

                # Print progress after each batch
                print(f"\nProgress: {completed_params}/{total_combinations} parameter sets tested")
                print(f"Valid parameter sets: {len(results)}/{completed_params}")
                if results:
                    print(f"Current best average MAE: {best_avg_mae:.2f}")

        # Save final summary file
        if results:
            summary_file = Path('validation_results') / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w') as f:
                summary = {
                    'test_subject': self.data_manager.name,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_parameter_sets': total_combinations,
                    'tested_parameter_sets': completed_params,
                    'valid_parameter_sets': len(results),
                    'parameter_combinations': sorted(results, key=lambda x: x['avg_mae'])
                }
                json.dump(summary, f, indent=4)
            print(f"\nFinal summary saved to: {summary_file}")

        self.results = results
        return results

    def analyze_results(self) -> List[Dict[str, Any]]:
        """Analyze and rank parameter combinations"""
        if not self.results:
            return []

        # Sort by average MAE
        return sorted(self.results, key=lambda x: x['avg_mae'])

def main():
    # Initialize
    data_manager = DataManager('david', load=False)
    rust_binary_path = "/home/ds/main/sleep-decoder/target/release/sleep-decoder"  # Update this

    # Create parameter space (use test_mode=True for initial testing)
    param_space = ParameterSpace(test_mode=False)
    print(f"Testing with {len(param_space.generate_combinations())} parameter combinations")

    # Create validator
    validator = RustValidator(data_manager, rust_binary_path)

    # Run parallel validation
    results = validator.parallel_validate(param_space, max_workers=2)

    # Analyze and print best results
    ranked_results = validator.analyze_results()
    if ranked_results:
        print("\nBest parameter combination:")
        print(json.dumps(ranked_results[0], indent=4))

if __name__ == "__main__":
    main()


