# Jobs

1. Capacitance sensor calibration, happens around priming time
    1. `python3 calibrate_sensor_thresholds.py --side=left --start_time="2025-02-02 20:00:00" --end_time="2025-02-02 21:14:00"`
1. Sleep detection, happens after power off
    1. `python3 analyze_sleep.py --side=right --start_time="2025-02-02 06:00:00" --end_time="2025-02-02 15:01:00"` 

