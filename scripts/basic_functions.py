## Defines some basic functions 

from scipy.signal import butter, filtfilt


# Defines a bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    '''
    Defines a bandpass filter with nyq being the nyquist, lowcut the lowpass and highcut the highpass frequency. The order markes the order of the bandpass filter.
    '''
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)