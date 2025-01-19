# Heart Rate Calculation Example

## Data Description
This data was for 1 person sleeping in the bed alone (on the right side of the bed)
- Male
- 168lbs
- 5'11"
- 29 years old
- Relatively athletic/fit

The timestamps are all in UTC format. Sleeping period was from 2025-01-15 06:30:00 UTC -> 2025-01-15 14:00:00 UTC

## Directory Structure
```
├── README.md              # This documentation
├── calculations.py        # Python script for data processing and analysis
├── heart_rate.csv         # Validation data: heart rate (cleaned from Apple Watch)
├── hrv.csv                # Validation data: heart rate variability (HRV)
├── plot.png               # Visualization of results
├── respiratory_rate.csv   # Validation data: respiratory rate
├── sensor_data.pkl        # Raw sensor data from the 8 Sleep piezo sensors
├── sleep_stage.csv        # Validation data: sleep stages
```

## Script Overview

The `calculations.py` script performs the following steps:

1. **Load Data**: Loads validation data from CSV files and raw sensor data from the pickle file.
2. **Preprocess Data**: Cleans and sorts the raw sensor data.
3. **Heart Rate Estimation**: Uses the `HeartPy` library to process raw piezoelectric sensor data, estimating heart rate, HRV, and respiratory rate in time intervals.
4. **Validation**: Compares estimated heart rate data with validation data from the Apple Watch.
5. **Visualization**: Generates plots comparing estimated biometrics to validation data.

### Key Parameters in the Script

- **Window size**: 10 seconds (time period for analyzing raw sensor data).
- **Sliding interval**: 5 seconds (overlap between consecutive windows).
- **Filter settings**: Bandpass filter with 0.05 Hz to 15 Hz range.

### Validation Plot

![Plot](./plot_results.png)


### Validation Output
```
{
    "mean": 67.3,
    "std": 4.45,
    "min": 55.17,
    "max": 78.7,
    "corr": "41.00%",
    "mae": 4.45,
    "mse": 36.83,
    "mape": "7.14%",
    "rmse": 6.07
}
```

There's a more complicated variation I'm building of `calculations.py`, which has better results. 
It's more complicated than this and uses both sensors. It's also shown much better results on other nights, ~80% correlation & 2-3 MAE. 
I chose this example because it has the most measurements from my watch and my model doesn't seem to have great results.
![Plot](./custom_model_outside_of_this_example.png)

