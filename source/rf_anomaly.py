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
data = np.reshape(data_raw, (-1, SAMPLE_SIZE))              #resize into SAMPLE_SIZE chunks
data = scipy.signal.wiener(data)                            #apply wiener filter to reduce noise
print(data.shape) 
std = np.std(data,axis=1)
mean = np.mean(data,axis=1)
skew = scipy.stats.skew(data,axis=1)
kurt = scipy.stats.kurtosis(data,axis=1)
predictions_mean = ARMA_Prediction(mean,order=(2,0))
predictions_std = ARMA_Prediction(std,order=(2,0))
predictions_skew = ARMA_Prediction(skew,order=(2,0))
predictions_kurt = ARMA_Prediction(kurt,order=(2,0))
mse = mean_squared_error(mean, predictions_mean)
print(abs(mse))
freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
freq += CENTER_FREQ
fftd = np.fft.fft(data[:1],axis=1)
fftd_abs = np.abs(fftd.T)
mask = fftd_abs.flatten().argsort()[::-1][:10]
peaks = freq[mask]*10e-7
print(peaks)
