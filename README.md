# 8 Sleep Analysis

```
conda env create -f environment.yml
conda develop src/
conda develop src/biometrics/
conda develop toolkit/
```

## Overview

This project is focused on analyzing heart rate data collected from 8 Sleep devices. The code in the `src/heart/`
directory is derived from the open-source library **HeartPy**, originally developed by Paul van Gent. The existing code
is currently being refactored and optimized to better suit the specific needs of 8 Sleep data analysis.

## Attribution

All original code in `src/heart/` was sourced from the HeartPy library and related resources. We acknowledge the
contributions of the original author and related research publications:

### Original Author:

- **Paul van Gent**
    - [HeartPy on PyPI](https://pypi.org/project/heartpy/)
    - [GitHub Repository](https://github.com/paulvangentcom/heartrate_analysis_python)
    - [Heart Rate Analysis for Human Factors: Development and Validation of an Open-Source Toolkit for Noisy Naturalistic Heart Rate Data](https://www.researchgate.net/publication/325967542_Heart_Rate_Analysis_for_Human_Factors_Development_and_Validation_of_an_Open_Source_Toolkit_for_Noisy_Naturalistic_Heart_Rate_Data)
    - [Analysing Noisy Driver Physiology in Real-Time Using Off-the-Shelf Sensors: Heart Rate Analysis Software from the Taking the Fast Lane Project](https://www.researchgate.net/publication/328654252_Analysing_Noisy_Driver_Physiology_Real-Time_Using_Off-the-Shelf_Sensors_Heart_Rate_Analysis_Software_from_the_Taking_the_Fast_Lane_Project?channel=doi&linkId=5bdab2c84585150b2b959d13&showFulltext=true)


## Project Structure

```
8sleep_biometrics/
│-- src/
│   │-- predictions.py        # (START HERE) Main script for running heart rate predictions
│   │-- data_manager.py       # Handles loading and cleaning data
│   │-- data_types.py         # Type definitions for raw data from 8 sleep
│   │-- calculations.py       # Core calculations and heart rate estimation
│   │-- analyze.py            # Prediction accuracy analysis
│   │-- run_data.py           # Data structures for managing run parameters
│   │-- rust/                 # Rust model validation scripts (external repo)
│   └-- param_optimizer/      # Scripts for hyperparameter tuning

data/                         # Raw and processed data storage
│-- people/
│   │-- david/
│   │   │-- raw/
│   │   │   │-- load/         # Folder for new raw data uploads (original data is cbor encoded)
│   │   │   |-- loaded/       # Raw files renamed and grouped into folders by date
│   │   │   |   └-- 2025-01-01/       
│   │   │   |       └-- 2025-01-10 05.00.22___2025-01-10 05.15.21.RAW       
│   │   │   └-- david_piezo_df.feather  # Processed piezoelectric sensor data
│   │   │
│   │   └-- validation/
│   │       │-- load/                   # Folder for uploading new validation data
│   │       ├-- david_breath_rate.csv   # Validation breathing rate data
│   │       ├-- david_heart_rate.csv    # Validation heart rate data
│   │       ├-- david_hrv.csv           # Validation heart rate variability (HRV) data
│   │       └-- david_sleep.csv         # Validation sleep stage data
```


### Adding New Data

1. Place raw data in the appropriate directory:
   ```
   8sleep_biometrics/data/people/<PERSON>/raw/load/
   ```

2. Update sleep periods in `src/sleep_data.py` to include new timestamps.




