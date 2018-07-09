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
SAMPLE_SIZE = SAMPLE_RATE//25
data_raw = scipy.fromfile('out_long.dat', dtype=complex)    #load data
data = np.reshape(data_raw, (-1, SAMPLE_SIZE))              #resize into SAMPLE_SIZE chunks
data = data[:5]
freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
freq += CENTER_FREQ
fftd = np.fft.fft(data,axis=1)
fftd_abs = np.abs(fftd.T)
fftd_abs = np.average(fftd_abs,axis=1)
fftd_db = 20*np.log(fftd_abs)
fftd_db += 875
# mask = fftd_db < 0
# fftd_db[mask] = 0
plt.plot(freq,fftd_abs)
plt.grid(True)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.xticks([8.95e7,9.03e7,9.13e7,9.19e7,9.23e7])

mask = fftd_db.argsort(axis=0).flatten()[:10]
print(mask.shape)
peaks = freq[mask]*10e-7
print(peaks)
plt.show()