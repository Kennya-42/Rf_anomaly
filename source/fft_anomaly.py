#!/usr/bin/python3
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
###################################
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000#5mil/sec
SAMPLE_SIZE = 100000
data_raw = scipy.fromfile('out_longest.dat', dtype=complex)#load data
data = np.reshape(data_raw, (-1, SAMPLE_SIZE))          #resize into SAMPLE_SIZE chunks
print(data.shape)
for i in range(11,12):
    freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
    freq += CENTER_FREQ
    fftd = np.fft.fft(data[i:i+1],axis=1)
    fftd_abs = np.abs(fftd.T)
    fftd_db = 20*np.log(fftd_abs)
    fftd_db += 877
    mask = fftd_abs < 0
    fftd_abs[mask] = 0
    mask = fftd_abs.flatten().argsort()[::-1][:20]
    print(mask)
    peaks = freq[mask]*10e-7
    print(peaks)
    plt.plot(freq,fftd_abs)
    for index in mask:
        plt.axvspan(freq[index], freq[index], color='red', alpha=1)
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xticks([8.95e7,9.03e7,9.13e7,9.19e7,9.23e7])
    plt.show()