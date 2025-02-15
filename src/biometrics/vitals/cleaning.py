import numpy as np


def interpolate_outliers_in_wave(data, lower_percentile: float = 2, upper_percentile: float = 98):
    """
    Replaces outliers in the data using linear interpolation.

    Parameters
    ----------
    data : 1D np.array
        Input data array.
    lower_percentile : float
        Lower bound percentile (e.g., 2 for the 2nd percentile).
    upper_percentile : float
        Upper bound percentile (e.g., 98 for the 98th percentile).

    Returns
    -------
    np.array
        Array with outliers replaced via interpolation.
    """
    data = data.astype(float).copy()

    if data.size == 0:
        return data  # Return empty array if input is empty

    # Compute bounds using percentiles
    lower_bound, upper_bound = np.percentile(data, [lower_percentile, upper_percentile])

    # Mask outliers
    mask = (data < lower_bound) | (data > upper_bound)
    valid_idx = np.where(~mask)[0]

    if valid_idx.size == 0:
        # If no valid values, return clipped data to bounds
        return np.clip(data, lower_bound, upper_bound)

    # Interpolate missing (outlier) values
    data[mask] = np.interp(np.where(mask)[0], valid_idx, data[valid_idx])

    return data

