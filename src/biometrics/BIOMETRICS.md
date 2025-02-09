# Biometrics


## Data Sources
- There's 2 main sensors used to measure biometrics, they're both available in /persistent/*.RAW files
- This data is only available if the Pod cannot access the internet. You can block internet access to the pod by setting up firewall rules
- The raw files are encoded in cbor & can be loaded with `load_raw_files.py`


### 1. Capacitance sensor data
- This measures pressure in 1 second intervals. There's 3 sensors for each side 

- Sample:
```json
{
  "type": "capSense",
  "ts": "2025-01-10 11:00:22",
  "left": {
    "out": 387,
    "cen": 381,
    "in": 505,
    "status": "good"
  },
  "right": {
    "out": 1076,
    "cen": 1075,
    "in": 1074,
    "status": "good"
  },
  "seq": 1610679
}
```

### 2. Piezo sensor data
- This measures pressure 500x a second
- Pod 3 has 2 piezo sensors, Pod 4 has 1 piezo sensor
```json
{
  "adc": 1,
  "freq": 500,
  "gain": 400,
  "left1": [
    -163532,
    -161494
    //  ...500 more
  ],
  "left2": [
    -59995,
    -63199
    //  ...500 more
  ],
  "right1": [
    81464,
    80593
    //  ...500 more
  ],
  "right2": [
    722955,
    723792,
    //  ...500 more
  ],
  "seq": 1610681,
  "ts": "2025-01-10 11:00:22",
  "type": "piezo-dual"
}
```


## Calculating metrics

### Presence detection
#### Step 1: Establish a baseline for capacitance sensors
- Run this for a time interval no one was in the bed
- `python3 sleep_detection/calibrate_sensor_thresholds.py --side=left --start_time="2025-02-08 18:00:00" --end_time="2025-02-08 18:50:00"`
- Calculates the thresholds for the capacitance sensors and saves the thresholds to `/persistent/free-sleep-data/SIDE_cap_baseline.json`

#### Step 2: Detect presence
- Run this for any time frame to detect when someone was in a bed
- `python3 sleep_detection/sleep_detector.py --side=left --start_time="2025-02-08 06:00:00" --end_time="2025-02-08 14:50:00"`
- Saves results to `sleep_records` @ SQLite `/persistent/free-sleep-data/free-sleep.db`


### Heart rate, HRV, breathing rate
- Run this during a period someone was in the bed
- `python3 vitals/calculate_vitals.py`




