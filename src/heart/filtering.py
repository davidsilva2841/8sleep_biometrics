from scipy.signal import butter, filtfilt




def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_signal(data, cutoff, sample_rate, order=2):
    b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)

    return filtfilt(b, a, data)


