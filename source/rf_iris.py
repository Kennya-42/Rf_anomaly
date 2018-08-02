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
data = scipy.fromfile('out_longest.dat', dtype=complex)[:SAMPLE_SIZE]
seconds = data.shape[0]/(2 * SAMPLE_RATE)
print('Seconds of Data: ',seconds,flush=True)
freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
freq += CENTER_FREQ
fft = np.fft.fft(data)
s = freq.shape[0]//2 
freqo = np.fft.fftshift(freq)
ffto = np.fft.fftshift(fft)
band = (SAMPLE_SIZE/SAMPLE_RATE)*500000#500mhz
fft,freq = fft_downsample(ffto,sampsize=SAMPLE_SIZE,mode='mean',b=200000,rate=2)
fft = np.asarray(fft)
freq = np.asarray(freq)
reduct = ((ffto.shape[0]-fft.shape[0])/ffto.shape[0])*100
print("%2d%% reduction in size!"% (reduct))
ffto = np.fft.ifftshift(ffto)
fft = np.fft.ifftshift(fft)
s = np.fft.ifft(ffto)
l = ffto.shape[0]
sn = np.fft.ifft(fft,n=l)
plt.plot(s)
plt.plot(sn,alpha=0.5)
plt.plot((s-sn),alpha=0.5)
plt.show()
# plt.plot(freqo,ffto,label='og')
# plt.plot(freq,fft,label='ds')
# plt.legend()
# plt.show()


