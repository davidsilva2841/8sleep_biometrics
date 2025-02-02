import pandas as pd
from run_data import ChartInfo

from src.data_manager import DataManager
from src.analyze import analyze_predictions

"""
HR_WINDOW_SECONDS=10
HR_WINDOW_OVERLAP_PERCENT=0.1
BR_WINDOW_SECONDS=120
BR_WINDOW_OVERLAP_PERCENT=0

cargo run --release /Users/ds/main/8sleep_biometrics/data/people/tally/raw/loaded/2025-01-21 \
  --csv-output=/Users/ds/main/8sleep_biometrics/predictions/tally_01_21/21 \
  
            

cargo run --bin sleep-decoder --release \
/Users/ds/main/8sleep_biometrics/data/people/tally/raw/tally_piezo_df.feather \
--start-time "2025-01-21 05:30" \
--end-time "2025-01-21 14:08" \
--csv-output /Users/ds/main/8sleep_biometrics/predictions/tally/01_21 \
--hr-window-seconds 10.0 \
--hr-window-overlap 0.67 \
--br-window-seconds=120.0 \
--br-window-overlap=0.0 \
--harmonic-penalty-close=0.7 \
--harmonic-penalty-far=0.3 \
--hr-smoothing-window=60 \
--hr-smoothing-strength=0.25 \
--hr-history-window=180 \
--hr-outlier-percentile=0.01 \
--feather-input

"""

"""
cargo run --release /Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-28 --start-time="2025-01-01 22:00" --end-time="2025-01-02 09:00"
cargo run --release --bin sleep-decoder /Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-27  --csv-output=/Users/ds/main/8sleep_biometrics/tmp/27/27 --hr-window-seconds=10.0 --hr-window-overlap=0.1 --hr-smoothing-window=60 --hr-smoothing-strength=0.25 --hr-outlier-percentile=0.05 --hr-history-window=60 --br-window-seconds=120.0 --br-window-overlap=0.0 --harmonic-penalty-close=0.8 --harmonic-penalty-far=0.5 --harmonic-close-threshold=5.0 --harmonic-far-threshold=10.0
                  
cargo run --release --bin sleep-decoder /Users/ds/main/8sleep_biometrics/data/people/david/raw/loaded/2025-01-28 --start-time="2025-01-01 22:00" \
  --csv-output=/Users/ds/main/8sleep_biometrics/tmp/28/ \      
  --hr-window-seconds=10.0 \                       
  --hr-window-overlap=0.1 \                        
  --hr-smoothing-window=60 \                       
  --hr-smoothing-strength=0.25 \                   
  --hr-outlier-percentile=0.05 \                   
  --hr-history-window=60 \                         
  --br-window-seconds=120.0 \                      
  --br-window-overlap=0.0 \                        
  --harmonic-penalty-close=0.8 \                   
  --harmonic-penalty-far=0.5 \                     
  --harmonic-close-threshold=5.0 \                 
  --harmonic-far-threshold=10.0    
  
"""


def main():

    david = DataManager('david', load=False)
    data = david
    period = data.sleep_periods[-1]
    start_time = period['start_time']
    end_time = period['end_time']

    file_path = '/Users/ds/main/8sleep_biometrics/tmp/27/27_right_combined_period_0.csv'
    df_pred = pd.read_csv(file_path)

    df_pred.rename({
        'fft_hr_smoothed': 'heart_rate',
        'timestamp': 'start_time',
    }, axis=1, inplace=True)


    chart_info: ChartInfo = {
        'labels': {
            'start_time': start_time,
            'end_time': end_time,
            'source': 'overlord'
        }
    }

    analyze_predictions(data, df_pred, start_time, end_time, chart_info, plot=True)


