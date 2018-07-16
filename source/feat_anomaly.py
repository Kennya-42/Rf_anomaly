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
warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
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
SAMPLE_SIZE = (SAMPLE_RATE//100)#10e3
SAMPLE_SIZE = 100000
print(SAMPLE_SIZE)
size = 61#61 #best filter size via hyperparam sweep, order= (2,0)
#############MAIN#######################################################             
data_raw = scipy.fromfile('out_long.dat', dtype=complex).astype(float)#load data
data_raw *= 10e24                                                     #rescale data
data_sm = scipy.ndimage.filters.gaussian_filter(data_raw,sigma=1)     #guassian filter raw data
data_rs = np.reshape(data_sm,(-1, SAMPLE_SIZE)).astype(float)         #resize into SAMPLE_SIZE chunks
data = scipy.signal.wiener(data_rs,mysize=size)                       #apply wiener filter to reduce noise
mean = np.mean(data,axis=1)                                           #get mean 
print(mean.shape)
mean_rs = scipy.signal.wiener(mean,mysize=size)                       #wiener filter mean data

print(np.isfinite(mean_rs).any()==False)
mean_rs = np.reshape(mean_rs,(-1, 50))
print(mean_rs.shape)
predictions_mean = list()
for index in range(mean_rs.shape[0]):
    predictions_mean.append( ARMA_Prediction(mean_rs[index],i=index,order=(1,0)) )
predictions_mean = np.asarray(predictions_mean).flatten()
diff1 = np.abs(predictions_mean - mean_rs.flatten()) 
############################################################################
fig, axs = plt.subplots(3, 1, sharex=True,sharey='col')
fig.subplots_adjust(hspace=0.4,left=0.12,right=0.98)
axs[0].set_title('mean/wiener-mean')
axs[0].plot(mean,label='mean',linewidth=1)
axs[0].plot(mean_rs.flatten(),label='wiener',linewidth=1)
axs[1].set_title('wiener/aram')
axs[1].plot(mean_rs.flatten(),label='wiener',linewidth=1)
axs[1].plot(predictions_mean,label='arma',linewidth=1)
axs[2].plot(diff1)
axs[2].set_title("diffrence arma/mean_filtered")
fig.legend(loc=3)
mse = mean_squared_error(mean_rs.flatten(),predictions_mean)
print(mse)
print(diff1.max())
plt.show()