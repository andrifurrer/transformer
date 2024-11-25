## Defines some basic functions 

from scipy.signal import butter, filtfilt
import pandas as pd


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

## Data Loader
def data_loader(subject, action):
    '''
    Automate input reading: select subject, action
    Read in csv file
    '''
    df_data = pd.read_csv(
        '../data/Finger/csv/s'+ str(subject) + '_' + str(action) + '.csv',
        sep=',',           # specify delimiter (default is ',')
        header=0,          # row number to use as column names (0 means the first row)
        na_values=['NA', ''],  # specify which values should be considered NaN
    )
    df_filtered = pd.DataFrame()

    # Sample data and sampling frequency
    fs = 500  
    # Define bandpass range for PPG 
    lowcut = 0.4
    highcut = 10

    df_filtered['ecg'] = bandpass_filter(df_data['ecg'], lowcut, highcut, fs, order=4)
    df_filtered['pleth_4'] = bandpass_filter(df_data['pleth_4'], lowcut, highcut, fs, order=4)
    df_filtered['pleth_5'] = bandpass_filter(df_data['pleth_5'], lowcut, highcut, fs, order=4)
    df_filtered['pleth_6'] = bandpass_filter(df_data['pleth_6'], lowcut, highcut, fs, order=4)
    return df_filtered