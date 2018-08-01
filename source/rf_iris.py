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
def fft_downsample(ffto,b=500000,rate=2,sampsize=500000,samprate=5000000,cfreq=91.3e6,mode='mean'):
    freqo = np.fft.fftfreq(sampsize,1/samprate)
    s = freqo.shape[0]//2
    freqo = np.append(freqo[s:],freqo[:s])
    freqo += CENTER_FREQ
    band = int((SAMPLE_SIZE/SAMPLE_RATE)*b)#500mhz
    ft,freq = [],[]
    fftn,freqn = [],[]
    #Downsample using the mean to combine samples.
    for i in range(0,int(s-band),rate):
        if mode == 'mean':
            ft = np.append(ft,(ffto[i]+ffto[i+1])/2)#downsample before primaryband
            fftn.append((ffto[s+band+i]+ffto[s+band+i+1])/2)#downsample after primaryband
            freq = np.append(freq,(freqo[i]+freqo[i+1])/2)#downsample freq
            freqn.append((freqo[s+band+i]+freqo[s+band+i+1])/2)
        else:
            ft = np.append(ft,ffto[i])#downsample before primaryband
            fftn.append(ffto[s+band+i])#downsample after primaryband
            freq = np.append(freq,(freqo[i]))#downsample freq
            freqn.append(freqo[s+band+i])
    #######################################################        
    fftn = np.asarray(fftn)
    freqn = np.asarray(freqn)
    ft = np.append(ft,ffto[s-band:s+band])
    ft = np.append(ft,fftn)
    freq = np.append(freq,freqo[s-band:s+band])
    freq= np.append(freq,freqn)
    return ft,freq

data = scipy.fromfile('out_longest.dat', dtype=complex)[:SAMPLE_SIZE]
seconds = data.shape[0]/(2 * SAMPLE_RATE)
print('Seconds of Data: ',seconds,flush=True)
freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
freq += CENTER_FREQ
fft = np.fft.fft(data)
s = freq.shape[0]//2 
# freqo = np.append(freq[s:],freq[:s])
freqo = np.fft.fftshift(freq)
ffto = np.fft.fftshift(fft)
# ffto = np.append(fft[s:],fft[:s])
# print(test[0],ffto[0])
band = (SAMPLE_SIZE/SAMPLE_RATE)*500000#500mhz
fft,freq = fft_downsample(ffto,sampsize=SAMPLE_SIZE,mode='mean',b=200000,rate=2)
# plt.plot(freqo,ffto,label='og')
fft = np.asarray(fft)
freq = np.asarray(freq)
reduct = ((ffto.shape[0]-fft.shape[0])/ffto.shape[0])*100
print("%2d%% reduction in size!"% (reduct))
ffto = np.fft.ifftshift(ffto)
fft = np.fft.ifftshift(fft)
s = np.fft.ifft(ffto)
sn = np.fft.ifft(fft)
plt.plot(s)
plt.plot(sn)
plt.show()
# plt.plot(freq,fft,label='ds')
# plt.legend()
# plt.show()


