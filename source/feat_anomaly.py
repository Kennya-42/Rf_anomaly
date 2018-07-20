#!/usr/bin/python3
# rf_anomaly.py
# Author: Ken Alexopoulos
# Script to look at rf data
import numpy as np 
import scipy
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as plticker
warnings.filterwarnings("ignore")
###################################
#Finds the ARMA prediction for given data.
def ARMA_P(data, order=(1,0)):
    model = sm.tsa.ARMA(data, order)
    result = model.fit(trend='c',disp=0)
    return np.asarray(result.predict())

#############VAR#############
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000
N = 2**21
SAMPLE_SIZE = 500000#100000
FILTER_SIZE = 61
ARMA_SIZE = 25
#############MAIN#############                                             
data = scipy.fromfile('out_longest.dat', dtype=complex)\
    .astype(np.float32)[:SAMPLE_RATE * 10]                #load data
# data *= 10e24                                           #rescale data
# data = np.clip(data,-10e4,10e4)                         #CLIP data since some values are extreme(maybe cause problems later)
np.log10(np.absolute(data),out=data)
res = np.isfinite(data)
np.bitwise_not(res,out=res)
data[res] = 0
data += abs(data.min())
plt.plot(data[:SAMPLE_SIZE],markevery=100)
plt.ion()
seconds = data.shape[0]/(2 * SAMPLE_RATE)                 #seconds of data
print('seconds: ',seconds)
data = gaussian_filter(data,sigma=1)                      #guassian filter raw data
print('Data has been smoothed!')
data = np.reshape(data,(-1, SAMPLE_SIZE))                 #resize into SAMPLE_SIZE chunks
NWIN = data.shape[0]
timescale = np.linspace(0, seconds, NWIN)
data = scipy.signal.wiener(data,mysize=FILTER_SIZE)       #apply wiener filter to reduce noise
mean = np.mean(data,axis=1)
std =  np.std(data,axis=1)
skew = scipy.stats.skew(data,axis=1)                      #get mean 
mean = scipy.signal.wiener(mean,mysize=FILTER_SIZE)       #wiener filter mean data
std = scipy.signal.wiener(std,mysize=FILTER_SIZE)
skew = scipy.signal.wiener(skew,mysize=FILTER_SIZE)
mean = np.reshape(mean,(-1, ARMA_SIZE))                   #reshape mean data
std = np.reshape(std,(-1, ARMA_SIZE))
skew = np.reshape(skew,(-1, ARMA_SIZE))
meanp,stdp,skewp = [],[],[]                                                                 
for index in range(mean.shape[0]):
    meanp.append( ARMA_P(mean[index],order=(1,0)) )

# for index in range(std.shape[0]):
#     stdp.append( ARMA_P(std[index],order=(1,0)) )

# for index in range(skew.shape[0]):
#     skewp.append( ARMA_P(skew[index],order=(1,0)) )

meanp = np.asarray(meanp).flatten()
# stdp = np.asarray(stdp).flatten()
# skewp = np.asarray(skewp).flatten()
np.absolute(mean,out=mean)
np.absolute(meanp,out=meanp)
meanp += abs(meanp.min())
mean += abs(mean.min())
# diff1 = np.abs(meanp - mean.flatten())
# diff1 = (diff1 - diff1.min())/diff1.max()
# diff2 = np.abs(stdp - std.flatten()) 
# diff2 = (diff2 - diff2.min())/diff2.max()
# diff3 = np.abs(skewp - skew.flatten())
# diff3 = (diff3 - diff3.min())/diff3.max() 
############################################################################
# fig, axs = plt.subplots(2, 1, sharex=True)
# fig.subplots_adjust(hspace=0.8,left=0.12,right=0.98)

# plt.plot(timescale,mean.flatten(),label='mean',markevery=5)
# plt.plot(timescale,meanp.flatten(),label='meanp',markevery=5)
# plt.legend()
# axs[1].set_title("mean diffrence")
# axs[1].set_xlabel('seconds')
# print(diff1.shape)
# axs[1].plot(diff1,label='diff1')
# loc = plticker.MultipleLocator(base=1)                #this locator puts ticks at regular intervals
# axs[0].xaxis.set_major_locator(loc)
# fig.legend(loc=3)
plt.show()


