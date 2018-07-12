#!/usr/bin/python3
# rf_anomaly.py
# Author: Ken Alexopoulos
# Script to look at rf data
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
###################################
#Finds the ARMA prediction for given data.
def ARMA_Prediction(data, order=(2,0)):
    model = sm.tsa.ARMA(data, order)
    result = model.fit(trend='c',disp=0)
    return np.asarray(result.predict())

#############VAR#############
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000#5mil/sec
N = 2**21
SAMPLE_SIZE = 10000#10e3
#############MAIN#############                                             
data_raw = scipy.fromfile('out.dat', dtype=complex)    #load data
data_uf = np.reshape(data_raw, (-1, SAMPLE_SIZE))         #resize into SAMPLE_SIZE chunks
data = scipy.signal.wiener(data_uf)                       #apply wiener filter to reduce noise
n = data.shape[0] 
std = np.std(data,axis=1)
mean = np.mean(data,axis=1)
skew = scipy.stats.skew(data,axis=1)
kurt = scipy.stats.kurtosis(data,axis=1)
predictions_mean = ARMA_Prediction(mean,order=(2,0))
predictions_std = ARMA_Prediction(std,order=(2,0))
predictions_skew = ARMA_Prediction(skew,order=(2,0))
predictions_kurt = ARMA_Prediction(kurt,order=(2,0))
freqlist = list()
print(data.shape)
for index in range(n):
    freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
    freq += CENTER_FREQ
    fftd = np.fft.fft(data[index])
    fftd_abs = np.abs(fftd.T)
    mask = fftd_abs.flatten().argsort()[::-1][:10]
    peaks = freq[mask]*10e-7
    freqlist.append(peaks[0])

print(freqlist[:10])
fig, axs = plt.subplots(5, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)
axs[0].plot(mean,label='mean',linewidth=1)
axs[0].set_title('mean')
axs[0].plot(predictions_mean,label='mean_pred',linewidth=1,linestyle='dashed')
axs[1].plot(std,label='std')
axs[1].set_title('std')
axs[1].plot(predictions_std,label='std_pred',linewidth=1,linestyle='dashed')
axs[2].plot(skew,label='skew')
axs[2].set_title('skew')
axs[2].plot(predictions_skew,label='skew_pred',linewidth=1,linestyle='dashed')
axs[3].plot(kurt,label='kurt')
axs[3].set_title('kurt')
axs[3].plot(predictions_kurt,label='kurt_pred',linewidth=1,linestyle='dashed')
axs[4].plot(freqlist,label='freq')
axs[4].set_yticks([88.1,91.3,95.3])
axs[4].set_title('freq')
fig.legend()
plt.show()