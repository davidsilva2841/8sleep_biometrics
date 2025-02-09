# heart_peaks.pyx
from libc.math cimport fmax
import numpy as np
cimport numpy as np
# from heart.analysis import calc_rr
# ---------------------------------------------------------------------------------------------------
# import numpy as np  # Regular Python NumPy import
# cimport numpy as np  # Cython NumPy import

# Define Cython types for NumPy arrays
ctypedef np.float64_t FLOAT_t
ctypedef np.int32_t INT_t

def calc_rr(np.ndarray[INT_t, ndim=1] peaklist, double sample_rate, dict working_data):
    """
    Optimized Cython version of calc_rr using cimport numpy as np.
    """
    cdef int i, n
    cdef np.ndarray[FLOAT_t, ndim=1] rr_list, rr_diff, rr_sqdiff
    cdef list rr_indices

    # Ensure peaklist is valid
    n = peaklist.shape[0]
    if n == 0:
        return working_data

    # Delete first peak if within first 150ms
    if peaklist[0] <= ((sample_rate / 1000.0) * 150):
        peaklist = peaklist[1:]  # Faster slicing than np.delete()
        working_data['peaklist'] = peaklist
        working_data['ybeat'] = working_data['ybeat'][1:]

    # Allocate memory for RR calculations
    rr_list = np.empty(n - 1, dtype=np.float64)
    rr_diff = np.empty(n - 2, dtype=np.float64)
    rr_sqdiff = np.empty(n - 2, dtype=np.float64)
    rr_indices = []

    # Compute RR intervals
    for i in range(n - 1):
        rr_list[i] = (peaklist[i + 1] - peaklist[i]) / sample_rate * 1000.0
        if i < n - 2:
            rr_diff[i] = abs(rr_list[i + 1] - rr_list[i])
            rr_sqdiff[i] = rr_diff[i] ** 2
        rr_indices.append((peaklist[i], peaklist[i + 1]))

    # Store results
    working_data['RR_list'] = rr_list
    working_data['RR_indices'] = rr_indices
    working_data['RR_diff'] = rr_diff
    working_data['RR_sqdiff'] = rr_sqdiff

    return working_data


# ---------------------------------------------------------------------------------------------------




def detect_peaks(np.ndarray[np.float64_t, ndim=1] hrdata,
                 np.ndarray[np.float64_t, ndim=1] rol_mean,
                 int ma_perc,
                 double sample_rate,
                 bint update_dict=True,
                 dict working_data={}):

    cdef int i, start_idx, end_idx
    cdef np.ndarray[np.int32_t, ndim=1] peaksx
    cdef np.ndarray[np.float64_t, ndim=1] peaksy
    cdef np.ndarray[np.int32_t, ndim=1] peakedges
    cdef list peaklist = []

    # Optimized rolling mean calculation
    cdef double mn = np.mean(rol_mean) / 100 * ma_perc
    rol_mean += mn

    # Get indices where hrdata > rolling mean
    peaksx = np.where(hrdata > rol_mean)[0]
    peaksy = hrdata[peaksx]

    # Identify peak edges (points where peaks are separated)
    peakedges = np.concatenate((np.array([0]), np.where(np.diff(peaksx) > 1)[0], np.array([len(peaksx)])))

    # Find max peak in each segment
    for i in range(len(peakedges) - 1):
        start_idx = peakedges[i]
        end_idx = peakedges[i + 1]
        if start_idx < end_idx:  # Prevent errors
            max_index = start_idx + np.argmax(peaksy[start_idx:end_idx])
            peaklist.append(peaksx[max_index])

    if update_dict:
        working_data['peaklist'] = peaklist
        working_data['ybeat'] = [hrdata[x] for x in peaklist]
        working_data['rolling_mean'] = rol_mean
        working_data = calc_rr(working_data['peaklist'], sample_rate, working_data=working_data)
        working_data['rrsd'] = np.std(working_data['RR_list']) if len(working_data['RR_list']) > 0 else float('inf')
        return working_data
    else:
        return peaklist, working_data


def fit_peaks(np.ndarray[np.float64_t, ndim=1] hrdata,
              np.ndarray[np.float64_t, ndim=1] rol_mean,
              double sample_rate,
              int bpmmin=40,
              int bpmmax=180,
              dict working_data={}):

    cdef int ma_perc
    cdef double bpm
    cdef list rrsd = []
    cdef list valid_ma = []

    cdef int[18] ma_perc_list
    ma_perc_list[:] = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 300]

    for ma_perc in ma_perc_list:
        working_data = detect_peaks(hrdata, rol_mean, ma_perc, sample_rate, update_dict=True, working_data=working_data)
        bpm = (len(working_data['peaklist']) / (len(hrdata) / sample_rate)) * 60
        rrsd.append([working_data['rrsd'], bpm, ma_perc])

    for _rrsd, _bpm, _ma_perc in rrsd:
        if (_rrsd > 0.1) and (bpmmin <= _bpm <= bpmmax):
            valid_ma.append([_rrsd, _ma_perc])

    if len(valid_ma) > 0:
        best_ma = min(valid_ma, key=lambda t: t[0])[1]
        working_data['best'] = best_ma
        working_data = detect_peaks(hrdata, rol_mean, best_ma, sample_rate, update_dict=True, working_data=working_data)
        return working_data
    else:
        raise Exception(
            '\n----------------\nCould not determine best fit for given signal. '
            'Please check the source signal.\n Probable causes:\n'
            '- detected heart rate falls outside of bpmmin<->bpmmax constraints\n'
            '- no detectable heart rate present in signal\n'
            '- very noisy signal (consider filtering and scaling)\n'
            'If you\'re sure the signal contains heart rate data, consider filtering and/or scaling first.\n'
            '----------------\n'
        )
