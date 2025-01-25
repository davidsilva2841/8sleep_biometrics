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
                'harmonic_close_threshold': [5.0],
                'harmonic_far_threshold': [10.0],
                'hr_smoothing_window': [60],
                'hr_smoothing_strength': [0.25],
                'hr_history_window': [180],
                'hr_outlier_percentile': [0.01]
            }
        else:
            # Full parameter space
            self.params = {
                'hr_window_seconds': [10],
                'hr_window_overlap': [0.67],
                'br_window_seconds': [120.0],
                'br_window_overlap': [0.0],
                'harmonic_penalty_close': [0.7],
                'harmonic_penalty_far': [0.3],
                'harmonic_close_threshold': [1.0, 3.0, 5.0, 10.0],
                'harmonic_far_threshold': [3.0, 5.0, 10.0, 15.0],
                'hr_smoothing_window': [60, 75],
                'hr_smoothing_strength': [0.25],
                'hr_history_window': [160, 180, 240],
                'hr_outlier_percentile': [0.01],
                'sensor_id': [0, 1]
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
    def __init__(self, data_managers: List[DataManager], rust_binary_path: str):
        self.data_managers = data_managers
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
            f"--br-window-overlap={params['br_window_overlap']}",
            f"--harmonic-penalty-close={params['harmonic_penalty_close']}",
            f"--harmonic-penalty-far={params['harmonic_penalty_far']}",
            f"--harmonic-close-threshold={params['harmonic_close_threshold']}",
            f"--harmonic-far-threshold={params['harmonic_far_threshold']}",
            f"--hr-smoothing-window={params['hr_smoothing_window']}",
            f"--hr-smoothing-strength={params['hr_smoothing_strength']}",
            f"--hr-history-window={params['hr_history_window']}",
            f"--hr-outlier-percentile={params['hr_outlier_percentile']}",
            f"--sensor={params['sensor_id']}"
        ]
        return " ".join(str(x) for x in cmd)

    def calculate_metrics(self, df_pred: pd.DataFrame, period: Dict[str, str], data_manager: DataManager) -> Dict[str, Any]:
        """Calculate metrics between predictions and ground truth data"""
        try:
            # Convert timestamps to datetime
            df_pred['start_time'] = pd.to_datetime(df_pred['start_time'], format='%Y-%m-%d %H:%M')

            # Get ground truth data for this period
            ground_truth = data_manager.heart_rate_df.copy()
            ground_truth['start_time'] = pd.to_datetime(ground_truth['start_time'], format='%Y-%m-%d %H:%M:%S')

            # Filter ground truth data to period
            period_start = pd.to_datetime(period['start_time'], format='%Y-%m-%d %H:%M:%S')
            period_end = pd.to_datetime(period['end_time'], format='%Y-%m-%d %H:%M:%S')
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
                tolerance=pd.Timedelta('60s')
            )

            # Drop rows with missing values
            merged = merged.dropna(subset=['heart_rate', 'heart_rate_actual'])

            # print(f"Merged: {merged.head()}")
            if merged.empty:
                print(merged.head())
                # print(f"No overlapping data points for {period['start_time']} to {period['end_time']}")
                # print(f"Predictions: {df_pred.head()}")
                # print(f"Ground truth: {ground_truth.head()}")
                raise ValueError("No overlapping data points found between predictions and ground truth")

            # Calculate metrics exactly as in analyze.py
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np

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
            print(f"Error!!!!!!!!!!!!!!: {e}")
            raise ValueError(f"Error calculating metrics: {str(e)}")

    def run_validation(self, params: Dict[str, Any], period: Dict[str, str], data_manager: DataManager) -> Dict[str, Any]:
        """Run a single validation with given parameters"""
        try:
            start_time = time.time()

            # Create temporary output directory with unique path for each user and period
            param_hash = ParameterSpace.generate_params_hash(params)
            period_date = period['start_time'][:10]  # Extract YYYY-MM-DD
            temp_dir = Path(f"temp_results/{param_hash}/{data_manager.name}/{period_date}")
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Get raw data path
            raw_data_path = Path(data_manager.raw_folder) / f"{data_manager.name}_piezo_df.feather"

            if not raw_data_path.exists():
                print(f"No RAW data for user {data_manager.name} on {period_date}")
                return {
                    'params_hash': param_hash,
                    'error': f"Skipping - No RAW data for {period_date}",
                    'rust': params
                }

            # Add time range to parameters
            params_with_time = params.copy()
            params_with_time['start_time'] = period['start_time']
            params_with_time['end_time'] = period['end_time']

            # Build and run command with unique output path
            output_base = temp_dir / f"{data_manager.name}_{period_date}"
            cmd = self.build_command(
                params_with_time,
                str(raw_data_path),
                str(output_base)
            )
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    check=False,
                    # stdout=subprocess.PIPE,
                    # stderr=subprocess.PIPE,
                    text=True
                )
                print(result.stderr)
                print(result.stdout)
                # Check if the process completed successfully
                if result.returncode != 0:
                    print(f"Process failed with code {result.returncode} for user {data_manager.name} on {period['start_time']} to {period['end_time']}")
                    return {
                        'params_hash': param_hash,
                        'error': f"Skipping - Process failed with code {result.returncode}",
                        'rust': params
                    }

                # Get the side the user slept on from the period data
                side = period.get('side', 'right').lower()
                output_file = output_base.with_name(f"{output_base.name}_{side}_combined_period_0.csv")

                if not output_file.exists():
                    other_side = 'left' if side == 'right' else 'right'
                    other_file = output_base.with_name(f"{output_base.name}_{other_side}_combined_period_0.csv")
                    if other_file.exists():
                        output_file = other_file
                    else:
                        print(f"Error!!!!!!!!!!!!!!: No output file found for {data_manager.name} on {period_date}")
                        return {
                            'params_hash': param_hash,
                            'error': f"Skipping - No output file found",
                            'rust': params
                        }

            except Exception as e:
                print(f"Error!!!!!!!!!!!!!!: {e}")
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
                metrics = self.calculate_metrics(df_pred, period, data_manager)

                # Add additional information to results
                result = {
                    'start_time': period['start_time'],
                    'end_time': period['end_time'],
                    'name': data_manager.name,
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
                print(f"Error!!!!!!!!!!!!!!: {e}")
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
            print(f"Error!!!!!!!!!!!!!!: {e}")
            return {
                'params_hash': param_hash,
                'error': f"Skipping - {str(e)}",
                'rust': params
            }

    def parallel_validate(self, parameter_space: ParameterSpace, max_workers: int = 4) -> List[Dict[str, Any]]:
        """Run validations in parallel and track best results across all data managers"""
        combinations = parameter_space.generate_combinations()
        results = []
        parameter_results = {}  # Track results by param_hash
        best_avg_mae = float('inf')
        total_combinations = len(combinations)
        completed_params = 0

        print(f"\nStarting validation with {total_combinations} parameter combinations...")
        print(f"Testing against {len(self.data_managers)} subjects")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process parameter sets in batches
            for batch_start in range(0, total_combinations, max_workers):
                batch_end = min(batch_start + max_workers, total_combinations)
                batch = combinations[batch_start:batch_end]
                completed_params += len(batch)

                # For each parameter set in the batch, test first period for each data manager
                for params in batch:
                    param_hash = ParameterSpace.generate_params_hash(params)
                    parameter_results[param_hash] = {
                        'params': params,
                        'manager_results': {},  # Results per data manager
                        'has_errors': False
                    }

                    # Test first period for each data manager
                    first_period_futures = []
                    for manager in self.data_managers:
                        if not manager.sleep_periods:
                            print(f"Warning: No sleep periods for {manager.name}, skipping")
                            continue
                        first_period = manager.sleep_periods[0]
                        first_period_futures.append((
                            manager,
                            executor.submit(self.run_validation, params, first_period, manager)
                        ))

                    # Process first period results and launch remaining periods
                    remaining_futures = {}  # Map (param_hash, manager_name) to futures
                    for manager, future in first_period_futures:
                        first_result = future.result()

                        # Skip if first period failed
                        if 'error' in first_result:
                            parameter_results[param_hash]['manager_results'][manager.name] = {
                                'has_errors': True,
                                'results': []
                            }
                            continue

                        # Initialize results for this manager
                        parameter_results[param_hash]['manager_results'][manager.name] = {
                            'has_errors': False,
                            'results': [first_result]
                        }

                        # Launch remaining periods for this manager
                        remaining_futures[(param_hash, manager.name)] = [
                            executor.submit(self.run_validation, params, period, manager)
                            for period in manager.sleep_periods[1:]
                        ]

                    # Process remaining period results
                    for (param_hash, manager_name), futures in remaining_futures.items():
                        for future in futures:
                            result = future.result()
                            if 'error' in result:
                                parameter_results[param_hash]['manager_results'][manager_name]['has_errors'] = True
                                break
                            parameter_results[param_hash]['manager_results'][manager_name]['results'].append(result)

                    # Calculate aggregate metrics if we have valid results for all managers
                    all_valid = True
                    manager_metrics = []

                    for manager_name, manager_data in parameter_results[param_hash]['manager_results'].items():
                        if manager_data['has_errors'] or not manager_data['results']:
                            all_valid = False
                            break

                        # Calculate metrics for this manager
                        results_for_manager = manager_data['results']
                        manager_avg_metrics = {
                            'name': manager_name,
                            'avg_mae': sum(r['mae'] for r in results_for_manager) / len(results_for_manager),
                            'avg_mse': sum(r['mse'] for r in results_for_manager) / len(results_for_manager),
                            'avg_rmse': sum(r['rmse'] for r in results_for_manager) / len(results_for_manager),
                            'avg_corr': sum(float(r['corr'].strip('%')) for r in results_for_manager) / len(results_for_manager),
                            'worst_mae': max(r['mae'] for r in results_for_manager),
                            'best_mae': min(r['mae'] for r in results_for_manager),
                            'periods_tested': len(results_for_manager)
                        }
                        manager_metrics.append(manager_avg_metrics)

                    if all_valid:
                        # Calculate overall metrics across all managers
                        overall_metrics = {
                            'avg_mae': sum(m['avg_mae'] for m in manager_metrics) / len(manager_metrics),
                            'avg_mse': sum(m['avg_mse'] for m in manager_metrics) / len(manager_metrics),
                            'avg_rmse': sum(m['avg_rmse'] for m in manager_metrics) / len(manager_metrics),
                            'avg_corr': sum(m['avg_corr'] for m in manager_metrics) / len(manager_metrics),
                            'worst_mae': max(m['worst_mae'] for m in manager_metrics),
                            'best_mae': min(m['best_mae'] for m in manager_metrics),
                            'manager_results': manager_metrics
                        }

                        # Save aggregate results
                        aggregate_result = {
                            'params_hash': param_hash,
                            'rust': params,
                            **overall_metrics
                        }

                        results.append(aggregate_result)

                        # Check if this is a new best result
                        if overall_metrics['avg_mae'] < best_avg_mae:
                            best_avg_mae = overall_metrics['avg_mae']
                            print(f"\nNew best average result ({completed_params}/{total_combinations}):")
                            print(f"Overall Average MAE: {best_avg_mae:.2f}")
                            print("Per-subject results:")
                            for metrics in manager_metrics:
                                print(f"  {metrics['name']}: MAE={metrics['avg_mae']:.2f}, Correlation={metrics['avg_corr']:.2f}%")
                            print(f"Parameters: {json.dumps(params, indent=2)}")

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
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'subjects': [m.name for m in self.data_managers],
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
    data_managers = [
        DataManager('david', load=True),
        DataManager('tally', load=True),
        DataManager('trinity', load=True),
    ]
    rust_binary_path = "/home/ds/main/sleep-decoder/target/release/sleep-decoder"  # Update this

    # Create parameter space (use test_mode=True for initial testing)
    param_space = ParameterSpace(test_mode=False)
    print(f"Testing with {len(param_space.generate_combinations())} parameter combinations")

    # Create validator with all data managers
    validator = RustValidator(data_managers, rust_binary_path)

    # Run parallel validation
    results = validator.parallel_validate(param_space, max_workers=1)

    # Analyze and print best results
    ranked_results = validator.analyze_results()
    if ranked_results:
        print("\nBest parameter combination:")
        print(json.dumps(ranked_results[0], indent=4))

if __name__ == "__main__":
    main()


