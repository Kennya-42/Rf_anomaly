#!/usr/bin/python3
# feat_anomaly.py
# Author: Ken Alexopoulos
# Script to look at features in rf data
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
import scipy.ndimage.filters
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as plticker
warnings.filterwarnings("ignore")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
###################################
#Finds the ARMA prediction for given data.
def ARMA_Prediction(data,i,order=(2,0)):
    model = sm.tsa.ARMA(data, order)
    # try:
    result = model.fit(trend='c',disp=0)
    # except:
    #     print(i)
    #     return None
    return np.asarray(result.predict())

CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000#5mil/sec
# SAMPLE_SIZE = (SAMPLE_RATE//100)#10e3
SAMPLE_SIZE = 100000
size = 61#61 #best filter size via hyperparam sweep, order= (2,0)
#############MAIN#######################################################             
data_raw = scipy.fromfile('out_longest.dat', dtype=complex).astype(np.float32)[:120 * SAMPLE_RATE]#load data
data_raw *= 10e24                                                     #rescale data
data_raw = np.clip(data_raw,-10e4,10e4)#CLIP data since some values are extreme(maybe cause problems later)
seconds = data_raw.shape[0]/(2 * SAMPLE_RATE)
# print(np.absolute(data_raw[np.nonzero(data_raw)]).max())
# print(np.absolute(data_raw[np.nonzero(data_raw)]).min())
data_sm = scipy.ndimage.filters.gaussian_filter(data_raw,sigma=1)     #guassian filter raw data
data_rs = np.reshape(data_sm,(-1, SAMPLE_SIZE)).astype(float)         #resize into SAMPLE_SIZE chunks
data = scipy.signal.wiener(data_rs,mysize=size)                       #apply wiener filter to reduce noise
mean = np.mean(data,axis=1)                                           #get mean 
mean_rs = scipy.signal.wiener(mean,mysize=size)                       #wiener filter mean data
mean_rs = np.reshape(mean_rs,(-1, 25))
print(mean_rs.shape)
predictions_mean = list()
for index in range(mean_rs.shape[0]):
    predictions_mean.append( ARMA_Prediction(mean_rs[index],i=index,order=(1,0)) )
predictions_mean = np.asarray(predictions_mean).flatten()
diff1 = np.abs(predictions_mean - mean_rs.flatten()) 
############################################################################
fig, axs = plt.subplots(3, 1, sharex=True,sharey='col')
fig.subplots_adjust(hspace=0.4,left=0.12,right=0.98)
x2 = np.linspace(0, seconds, (mean_rs.shape[0]*25))
axs[0].set_title('mean/wiener-mean')
axs[0].plot(x2,mean,label='mean',linewidth=1)
axs[0].plot(x2,mean_rs.flatten(),label='wiener',linewidth=1)
axs[1].set_title('wiener/aram')
axs[1].plot(x2,mean_rs.flatten(),label='wiener',linewidth=1)
axs[1].plot(x2,predictions_mean,label='arma',linewidth=1)
axs[2].plot(x2,diff1)
axs[2].set_title("diffrence arma/mean_filtered")
axs[2].set_xlabel('seconds')
loc = plticker.MultipleLocator(base=1) # this locator puts ticks at regular intervals
axs[2].xaxis.set_major_locator(loc)
fig.legend(loc=3)
# plt.plot(data_raw,label='raw')
# plt.plot(data_sm,label='smooth')
# plt.legend(loc=3)
mse = mean_squared_error(mean_rs.flatten(),predictions_mean)
print(mse)
print(diff1.max())
plt.show()