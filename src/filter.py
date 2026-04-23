import numpy as np
import scipy.io
from scipy.signal import butter, lfilter，filtfilt
import math

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq  = 0.5 * fs  # Nyquist Frequency
    low  = lowcut / nyq
    high = highcut /nyq
    b, a = butter(order, [low, high], btype='band')
    y    = lfilter(b, a, data)
    return y

def filter_set(data_set, lowcut, highcut, fs, order=5):
    filtered_set = np.zeros_like(data_set)
    for i in range(data_set.shape[0]):
        for j in range(data_set.shape[1]):
            filtered_set[i, j, :] = bandpass_filter(data_set[i, j, :], lowcut, highcut, fs, order=5)
    return filtered_set