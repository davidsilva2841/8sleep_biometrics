import numpy as np

def interpolate_outliers_in_wave(data, percentile=2):
    """
    Parameters
    ----------
    data : 1D np.array
    percentile : float
        We treat the [percentile, 100 - percentile] range as valid.

    Returns
    -------
    out : 1D np.array of same length
    """

    # Convert to float, copy so we don’t overwrite original
    out = data.astype(float).copy()
    n = len(out)
    if n < 2:
        return out  # trivial edge case

    # 1) Compute bounds based on percentile
    lower_idx = int(n * (percentile / 100.0))
    upper_idx = int(n * (1.0 - percentile / 100.0))
    sorted_vals = np.sort(out)
    lower_bound = sorted_vals[lower_idx]
    upper_bound = sorted_vals[upper_idx]

    # 2) Replace outliers with linear interpolation
    for i in range(n):
        if out[i] < lower_bound or out[i] > upper_bound:
            # find previous valid
            j_prev = i - 1
            while j_prev >= 0 and (out[j_prev] < lower_bound or out[j_prev] > upper_bound):
                j_prev -= 1

            # find next valid
            j_next = i + 1
            while j_next < n and (out[j_next] < lower_bound or out[j_next] > upper_bound):
                j_next += 1

            if j_prev < 0 and j_next >= n:
                # everything's out of range, just clip to bounds
                out[i] = max(min(out[i], upper_bound), lower_bound)
            elif j_prev < 0:
                # no previous valid, so take the next valid or bounds
                out[i] = out[j_next]
            elif j_next >= n:
                # no next valid, so take the previous valid
                out[i] = out[j_prev]
            else:
                # Linear interpolation between out[j_prev] and out[j_next]
                out[i] = 0.5 * (out[j_prev] + out[j_next])
    return out

