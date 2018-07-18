#!/usr/bin/python3
# rf_anomaly.py
# Author: Ken Alexopoulos
# Script to look at rf data
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as plticker
warnings.filterwarnings("ignore")
###################################
#Finds the ARMA prediction for given data.
def ARMA_P(data, order=(2,0)):
    model = sm.tsa.ARMA(data, order)
    result = model.fit(trend='c',disp=0)
    return np.asarray(result.predict())

def getfftInfo(tsig):
    pfreq = list()
    for i in range(tsig.shape[0]):
        freq = np.fft.fftfreq(SAMPLE_SIZE,1/SAMPLE_RATE)
        freq += CENTER_FREQ
        fftd = np.fft.fft(tsig[i:i+1],axis=1)
        fftd_abs = np.abs(fftd.T)
        fftd_db = 20*np.log(fftd_abs)
        fftd_db += 877
        mask = fftd_db < 0
        fftd_db[mask] = 0
        mask = fftd_db.flatten().argsort()[::-1][:1]
        peaks = freq[mask]*10e-7
        pfreq.append(peaks[0])
    return pfreq
#############VAR#############
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000
N = 2**21
SAMPLE_SIZE = 500000#100000
FILTER_SIZE = 61
ARMA_SIZE = 25
#############MAIN#############                                             
data = scipy.fromfile('out_longest.dat', dtype=complex).astype(np.float32)[:SAMPLE_RATE * 60]#load data
data *= 10e24                                                                                #rescale data
data = np.clip(data,-10e4,10e4)                                                              #CLIP data since some values are extreme(maybe cause problems later)
seconds = data.shape[0]/(2 * SAMPLE_RATE)                                                    #seconds of data
data = scipy.ndimage.filters.gaussian_filter(data,sigma=1)                                   #guassian filter raw data
data = np.reshape(data,(-1, SAMPLE_SIZE))                                                    #resize into SAMPLE_SIZE chunks
fftpeaks = getfftInfo(data)
data = scipy.signal.wiener(data,mysize=FILTER_SIZE)                                          #apply wiener filter to reduce noise
mean = np.mean(data,axis=1) 
std =  np.std(data,axis=1)
skew = scipy.stats.skew(data,axis=1)                                                                   #get mean 
mean = scipy.signal.wiener(mean,mysize=FILTER_SIZE)                                          #wiener filter mean data
std = scipy.signal.wiener(std,mysize=FILTER_SIZE)
skew = scipy.signal.wiener(skew,mysize=FILTER_SIZE)
mean = np.reshape(mean,(-1, ARMA_SIZE))                                                      #reshape mean data
std = np.reshape(std,(-1, ARMA_SIZE))
skew = np.reshape(skew,(-1, ARMA_SIZE))
meanp = list()
stdp = list()
skewp = list()
for index in range(mean.shape[0]):
    meanp.append( ARMA_P(mean[index],order=(1,0)) )

for index in range(std.shape[0]):
    stdp.append( ARMA_P(std[index],order=(1,0)) )

for index in range(skew.shape[0]):
    skewp.append( ARMA_P(skew[index],order=(1,0)) )

meanp = np.asarray(meanp).flatten()
stdp = np.asarray(stdp).flatten()
skewp = np.asarray(skewp).flatten()
diff1 = np.abs(meanp - mean.flatten())
diff1 = (diff1 - diff1.min())/diff1.max()
diff2 = np.abs(stdp - std.flatten()) 
diff2 = (diff2 - diff2.min())/diff2.max()
diff3 = np.abs(skewp - skew.flatten())
diff3 = (diff3 - diff3.min())/diff3.max() 
############################################################################
fig, axs = plt.subplots(4, 1, sharex=True)
fig.subplots_adjust(hspace=0.8,left=0.12,right=0.98)
x2 = np.linspace(0, seconds, (mean.shape[0]*ARMA_SIZE))
print(x2.shape)
axs[0].set_title("mean diffrence")
axs[0].set_xlabel('seconds')
axs[0].plot(x2,diff1)
axs[1].plot(x2,diff2)
axs[1].set_title("std diffrence")
axs[1].set_xlabel('seconds')
# axs[2].set_title('skew raw/filter')
# axs[2].plot(x2,skew_r.flatten(),label='raw',linewidth=1)
# axs[2].plot(x2,skew_f.flatten(),label='wiener',linewidth=1)
# axs[3].set_title('skew filter/arma')
# axs[3].plot(x2,skew.flatten(),label='wiener',linewidth=1)
# axs[3].plot(x2,skewp,label='arma',linewidth=1)
axs[2].plot(x2,diff3)
axs[2].set_title("skew diffrence")
axs[2].set_xlabel('seconds')
axs[3].plot(x2,fftpeaks)
axs[3].set_title("peak frequencies")
loc = plticker.MultipleLocator(base=1)                                                              #this locator puts ticks at regular intervals
axs[1].xaxis.set_major_locator(loc)
fig.legend(loc=3)
plt.show()


