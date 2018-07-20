#!/usr/bin/python3
# rf_anomaly.py
# Author: Ken Alexopoulos
# Script to look at rf data
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
import matplotlib.ticker as plticker
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
np.set_printoptions(precision=4)
#############VAR#############
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000
SAMPLE_SIZE = 500000#100000
FILTER_SIZE = 61
ARMA_SIZE = 25
########################################################
#Finds the ARMA prediction for given data.
# @profile
def ARMA_P(sample, order=(1,0)):
    model = sm.tsa.ARMA(sample, order)
    result = model.fit(trend='c',disp=0)
    return np.asarray(result.predict())
#get the peak frequencies from fft
def getfftInfo(tsig):
    pfreq = []
    tsig = np.reshape(tsig,(-1, SAMPLE_SIZE))
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
#convert to finite array in log scale
def linear2db(d):
    np.log10(np.absolute(d),out=d)                         #convert data from linear to log
    res = np.isfinite(d)                                   #zero all none finite values
    np.bitwise_not(res,out=res)
    d[res] = 0
#############MAIN#############
                                             
data = scipy.fromfile('out_longest.dat', dtype=complex)\
    .astype(np.float32)[:SAMPLE_RATE * 120]                 #load data
seconds = data.shape[0]/(2 * SAMPLE_RATE)                  #seconds of data  
print('Seconds of Data: ',seconds,flush=True)  
fftpeaks = getfftInfo(data)                                #get fft info
linear2db(data)                                            #pass array by ref to reduce stack usage
data += abs(data.min())                                    #shift data
data = np.reshape(data,(-1, SAMPLE_SIZE))                  #resize into SAMPLE_SIZE chunks
NWIN = data.shape[0]                                       #number of windows
timescale = np.linspace(0, seconds, NWIN)
data = scipy.signal.wiener(data,mysize=FILTER_SIZE)        #apply wiener filter to reduce noise
print('raw data has been denoised.')
mean = np.mean(data,axis=1) 
std =  np.std(data,axis=1)
skew = scipy.stats.skew(data,axis=1)                       #get mean 
del data
mean = scipy.signal.wiener(mean,mysize=FILTER_SIZE)        #wiener filter mean data
std = scipy.signal.wiener(std,mysize=FILTER_SIZE)
skew = scipy.signal.wiener(skew,mysize=FILTER_SIZE)
#PCA#
features = np.vstack((mean,std,skew))
pca = PCA(n_components=3).fit(features)
ratios = pca.explained_variance_ratio_
####
mean = np.reshape(mean,(-1, ARMA_SIZE))                    #reshape mean data
std = np.reshape(std,(-1, ARMA_SIZE))
skew = np.reshape(skew,(-1, ARMA_SIZE))
meanp,stdp,skewp = [],[],[]
for index in range(mean.shape[0]):
    meanp.append(ARMA_P(mean[index]))
    stdp.append( ARMA_P( std[index]))
    skewp.append(ARMA_P(skew[index])) 
print('Aram predicitons have been made.')
meanp = np.asarray(meanp).flatten()
stdp = np.asarray(stdp).flatten()
skewp = np.asarray(skewp).flatten()
diff1 = np.abs(meanp - mean.flatten())          #get dif between prediction
diff2 = np.abs(stdp - std.flatten())       
diff3 = np.abs(skewp - skew.flatten())
diff1 = (diff1 - diff1.min())/diff1.max()       #normalize the data
diff2 = (diff2 - diff2.min())/diff2.max()
diff3 = (diff3 - diff3.min())/diff3.max()
diff4 = (diff1 * ratios[0])+(diff2 * ratios[1])+(diff3 * ratios[2])
del meanp
del stdp
del skewp
del mean
del std
del skew
anomalys = np.argwhere(diff4 > 0.5)
for i in anomalys:
    print('anomaly at: ',timescale[i],'sec')
############################################################################
fig, axs = plt.subplots(5, 1, sharex=True)
fig.subplots_adjust(hspace=0.8,left=0.12,right=0.98)
axs[0].set_title("mean diffrence")
axs[0].set_xlabel('seconds')
axs[0].plot(timescale,diff1,markevery=10)
axs[1].plot(timescale,diff2)
axs[1].set_title("std diffrence")
axs[1].set_xlabel('seconds')
axs[2].plot(timescale,diff3)
axs[2].set_title("skew diffrence")
axs[2].set_xlabel('seconds')
axs[3].plot(timescale,diff4)
axs[3].set_title("combined diffrence")
axs[3].set_xlabel('seconds')
axs[4].plot(timescale,fftpeaks)
axs[4].set_title("peak frequencies")
axs[4].set_yticks([89.3,91.3,93.1])
axs[4].set_xlabel('seconds')
loc = plticker.MultipleLocator(base=1)
axs[0].xaxis.set_major_locator(loc)
for index in anomalys:
        axs[3].axvspan(timescale[index-1], timescale[index+1], color='red', alpha=1)
# plt.show()
