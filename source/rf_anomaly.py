#!/usr/bin/python3
# rf_anomaly.py
# Author: Ken Alexopoulos
# Script to look at rf data
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy
import gc
from rf_func import *
warnings.filterwarnings("ignore")
def main():
#############VAR#################
    CHUNK_SIZE = 1600000000
    CENTER_FREQ = 91300000
    SAMPLE_RATE = 5000000
    SAMPLE_SIZE = 500000
    FILTER_SIZE = 61
    ARMA_SIZE = 25
    FFT_SIZE = 2**15
    f = open("out_longest.dat", "rb")
    meand,skewd,stdd,fftp = [],[],[],[]
    n = 1
    with f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            data = np.frombuffer(chunk,dtype=complex).astype(np.complex64)
            fftp = np.append(fftp,getfftInfo(data,fftsize=FFT_SIZE))
            np.abs(data,out=data) #convert to magnitude
            data = data.real.astype(np.float32)
            t1,t2,t3 = pipeline(data,n)
            meand = np.append(meand,t1)
            stdd  = np.append(stdd,t2)
            skewd = np.append(skewd,t3)
            if n == 2:
                break
            n += 1
            gc.collect()
    print('Finish!')
    meand = (meand - meand.min())/meand.max()                  
    stdd = (stdd - stdd.min())/stdd.max()
    skewd = (skewd - skewd.min())/skewd.max()
    features = np.vstack((meand,stdd,skewd))
    pca = PCA(n_components=3).fit(features)
    ratios = pca.explained_variance_ratio_
    diff = (meand * ratios[0])+(stdd * ratios[1])+(skewd * ratios[2])
    samples = diff.shape[0]
    Npoints = (samples)*SAMPLE_SIZE             
    seconds = (Npoints)/(2 * SAMPLE_RATE)           
    timescale = np.linspace(0, seconds, samples)
    anomalys = np.argwhere(diff > 0.5)
    for i in anomalys:
        print('anomaly at: ',timescale[i],'sec')
        print('peak freak is: ',fftp[i])
    fig, axs = plt.subplots(5, 1, sharex=True)
    fig.subplots_adjust(hspace=0.8,left=0.12,right=0.98)
    axs[0].set_title("mean diffrence")
    axs[0].set_xlabel('seconds')
    axs[0].plot(timescale,meand)
    axs[1].set_title("std diffrence")
    axs[1].set_xlabel('seconds')
    axs[1].plot(timescale,stdd)
    axs[2].set_title("skew diffrence")
    axs[2].set_xlabel('seconds')
    axs[2].plot(timescale,skewd)
    axs[3].set_title("comb diffrence")
    axs[3].set_xlabel('seconds')
    axs[3].plot(timescale,diff)
    axs[4].set_title("peak freqs")
    axs[4].set_yticks([89.3,91.3,93.1])
    axs[4].set_xlabel('seconds')
    axs[4].plot(timescale,fftp)
    for index in anomalys:
        axs[3].axvspan(timescale[index], timescale[index+1], color='red', alpha=1)
    plt.show()
if __name__ == "__main__":
    main()

        



        
