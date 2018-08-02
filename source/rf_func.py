#!/usr/bin/python3
# rf_func.py
# Author: Ken Alexopoulos
# helper functions to look at rf data
import numpy as np
import scipy
import gc
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#Finds the ARMA prediction for given data.
def ARMA_P(sample, order=(1,0)):
    model = sm.tsa.ARMA(sample, order)
    result = model.fit(trend='c',disp=0)
    return np.asarray(result.predict())
# get the peak frequencies from fft
# d: raw flat time series data
# pfreq: peak frequencies of each window.
def getfftInfo(d,sampsize=500000,samprate=5000000,cfreq=91.3e6,fftsize=500000):
    pfreq = []
    d = np.reshape(d,(-1, sampsize))
    for i in range(d.shape[0]):
        # freq = np.fft.fftfreq(sampsize,1/samprate)
        freq = np.fft.fftfreq(fftsize,1/samprate)
        freq += cfreq
        fftd = np.fft.fft(d[i:i+1],axis=1,n=fftsize).flatten()
        fftd,freq = fft_downsample(fftd,b=500000,rate=2,sampsize=500000,samprate=5000000,cfreq=91.3e6,mode='mean',fftsize=fftsize)
        fftd_abs = np.abs(fftd.T)
        fftd_db = 20*np.log(fftd_abs)
        fftd_db += 877
        mask = fftd_db < 0
        fftd_db[mask] = 0
        mask = fftd_abs.flatten().argsort()[::-1][:1]
        peaks = freq[mask]*10e-7
        pfreq.append(peaks[0])
    return pfreq
# convert to finite array in log scale
# d: raw linear timeseries data
# returns the transformation inplace to save space
def linear2db(d):
    np.log10(d,out=d)
    res = np.isfinite(d)
    np.bitwise_not(res,out=res)
    d[res] = 0
# main pipeline preformed in chunksize pieces
# data: raw timeseries data
# n: current iteration
def pipeline(data,n=1,sampsize=500000,samprate=5000000,f_size=61,a_size=25):
    seconds = data.shape[0]/(2 * samprate)        #seconds of data 
    print('Seconds of Data processed: ',(seconds * n),flush=True)
    linear2db(data) 
    data += abs(data.min())                       #shift data
    data = np.reshape(data,(-1, sampsize))     #resize to SAMPLE_SIZE chunks
    # data = scipy.signal.wiener(data,mysize=f_size)#wiener filter noise
    mean = np.mean(data,axis=1) 
    std =  np.std(data,axis=1)
    skew = scipy.stats.skew(data,axis=1)            #get mean 
    del data
    mean = scipy.signal.wiener(mean,mysize=f_size)  #wiener filter mean data
    std = scipy.signal.wiener(std,mysize=f_size)
    skew = scipy.signal.wiener(skew,mysize=f_size)
    features = np.vstack((mean,std,skew))
    pca = PCA(n_components=3).fit(features)
    ratios = pca.explained_variance_ratio_
    mean = np.reshape(mean,(-1, a_size))            #reshape mean data
    std = np.reshape(std,(-1, a_size))
    skew = np.reshape(skew,(-1, a_size))
    meanp,stdp,skewp = [],[],[]
    for index in range(mean.shape[0]):
        meanp.append(ARMA_P(mean[index]))
        stdp.append( ARMA_P( std[index]))
        skewp.append(ARMA_P(skew[index]))
    meanp = np.asarray(meanp).flatten()
    stdp = np.asarray(stdp).flatten()
    skewp = np.asarray(skewp).flatten()
    diff1 = np.abs(meanp - mean.flatten())        #get dif between prediction
    diff2 = np.abs(stdp - std.flatten())       
    diff3 = np.abs(skewp - skew.flatten())
    # diff1 = (diff1 - diff1.min())/diff1.max()
    # diff2 = (diff2 - diff2.min())/diff2.max()
    # diff3 = (diff3 - diff3.min())/diff3.max()
    # diffc = (diff1 * ratios[0])+(diff2 * ratios[1])+(diff3 * ratios[2])
    return diff1,diff2,diff3

def fft_downsample(ffto,b=500000,rate=2,sampsize=500000,samprate=5000000,cfreq=91.3e6,mode='mean',fftsize=500000):
    # freqo = np.fft.fftfreq(sampsize,1/samprate)
    ffto = np.fft.fftshift(ffto)
    freqo = np.fft.fftfreq(fftsize,1/samprate)
    s = freqo.shape[0]//2
    freqo = np.append(freqo[s:],freqo[:s])
    freqo += cfreq
    band = int((fftsize/samprate)*b)#500mhz
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