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
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
###################################
#Finds the ARMA prediction for given data.
def ARMA_P(data, order=(2,0)):
    model = sm.tsa.ARMA(data, order)
    result = model.fit(trend='c',disp=0)
    return np.asarray(result.predict())
#############VAR#############
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000
N = 2**21
SAMPLE_SIZE = 100000
FILTER_SIZE = 61
ARMA_SIZE = 25
#############MAIN#############                                             
data = scipy.fromfile('out_longest.dat', dtype=complex).astype(np.float32)[:60*SAMPLE_RATE]  #load data
data *= 10e24                                                                                #rescale data
data = np.clip(data,-10e4,10e4)                                                              #CLIP data since some values are extreme(maybe cause problems later)
seconds = data.shape[0]/(2 * SAMPLE_RATE)                                                    #seconds of data
data = scipy.ndimage.filters.gaussian_filter(data,sigma=1)                                   #guassian filter raw data
data = np.reshape(data,(-1, SAMPLE_SIZE)).astype(float)                                      #resize into SAMPLE_SIZE chunks
data = scipy.signal.wiener(data,mysize=FILTER_SIZE)                                          #apply wiener filter to reduce noise
mean = np.mean(data,axis=1) 
std =  np.std(data,axis=1)                                                                #get mean 
mean = scipy.signal.wiener(mean,mysize=FILTER_SIZE)                                          #wiener filter mean data
std = scipy.signal.wiener(std,mysize=FILTER_SIZE)
mean = np.reshape(mean,(-1, ARMA_SIZE))                                                      #reshape mean data
std = np.reshape(std,(-1, ARMA_SIZE))
print(mean.shape)
meanp = list()
stdp = list()
for index in range(mean.shape[0]):
    meanp.append( ARMA_P(mean[index],order=(1,0)) )

for index in range(std.shape[0]):
    stdp.append( ARMA_P(std[index],order=(1,0)) )

meanp = np.asarray(meanp).flatten()
stdp = np.asarray(stdp).flatten()
diff1 = np.abs(meanp - mean.flatten())
diff2 = np.abs(stdp - std.flatten()) 
############################################################################
fig, axs = plt.subplots(4, 1, sharex=True,sharey='col')
fig.subplots_adjust(hspace=0.4,left=0.12,right=0.98)
x2 = np.linspace(0, seconds, (mean.shape[0]*ARMA_SIZE))
axs[0].set_title('wiener/aram')
axs[0].semilogx(x2,mean.flatten(),label='wiener',linewidth=1)
axs[0].semilogx(x2,meanp,label='arma',linewidth=1)
axs[1].set_title("diffrence arma/mean_filtered")
axs[1].set_xlabel('seconds')
axs[1].semilogx(x2,diff1)
# axs[2].set_title('wiener/aram')
# axs[2].plot(x2,std.flatten(),label='wiener',linewidth=1)
# axs[2].plot(x2,stdp,label='arma',linewidth=1)
# axs[3].plot(x2,diff2)
# axs[3].set_title("diffrence arma/std_filtered")
# axs[3].set_xlabel('seconds')

loc = plticker.MultipleLocator(base=1)                                                              #this locator puts ticks at regular intervals
axs[1].xaxis.set_major_locator(loc)
fig.legend(loc=3)
# mse = mean_squared_error(mean.flatten(),meanp)
plt.show()


