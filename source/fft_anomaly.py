#!/usr/bin/python3
# fft_anomaly.py
# Author: Ken Alexopoulos
# Script to look at fft features
import numpy as np 
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import warnings
from rf_func import *
warnings.filterwarnings("ignore")
###################################
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000#5mil/sec
SAMPLE_SIZE = 2**15
data = scipy.fromfile('out_longest.dat', dtype=complex)[:SAMPLE_SIZE * 10]
seconds = data.shape[0]/(2 * SAMPLE_RATE)
print('Seconds of Data: ',seconds,flush=True)
# timescale = np.linspace(0, seconds, int(data.shape[0]))
# plt.plot(timescale,data)
# plt.xlabel('Seconds')
# plt.ylabel('Amplitude')
# plt.show()
data = np.reshape(data,(-1, SAMPLE_SIZE))
freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
freq += CENTER_FREQ
print(freq[0])
fft = np.fft.fft(data,axis=1)
phase = np.angle(fft)/np.pi
phase = phase.mean(axis=0)
fft = fft[0]
s = freq.shape[0]//2
# band = (SAMPLE_SIZE/SAMPLE_RATE)*200000#200khz
band = (SAMPLE_SIZE/SAMPLE_RATE)*500000
freq = np.append(freq[s:],freq[:s])
l,u = int(s-band),int(s+band)
fft = np.append(fft[s:],fft[:s])
plt.plot(freq[l:u],fft[l:u])
plt.show()

# samples = fft.shape[0]
# Npoints = (samples)*SAMPLE_SIZE  
# seconds = (Npoints)/(2 * SAMPLE_RATE)           
# timescale = np.linspace(0, seconds, samples)
# plt.plot(timescale,phase)
# plt.show()
# fftd_abs = np.abs(fftd)
# fftd_db = 20*np.log(fftd_abs)
# fftd_db += 877
# mask = fftd_abs < 0
# fftd_abs[mask] = 0
# mask = fftd_abs.flatten().argsort()[::-1][:20]
# peaks = freq[mask]*10e-7
# samples = spectrogram.shape[0]
# Npoints = (samples)*SAMPLE_SIZE  
# seconds = (Npoints)/(2 * SAMPLE_RATE)           
# timescale = np.linspace(0, 0.5, samples)
# plt.plot(freq,fftd_abs)
# for index in mask:
#     plt.axvspan(freq[index], freq[index], color='red', alpha=1)
# plt.grid(True)
# plt.xlabel('Frequency [MHz]')
# plt.ylabel('Amplitude [dB]')
# plt.xticks([89.5e6,90.3e6,91.3e6,91.9e6,92.3e6])
# plt.show()