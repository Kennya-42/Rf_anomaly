#!/usr/bin/python3
# feat_anomaly.py
# Author: Ken Alexopoulos
# Script to look at features in rf data
import numpy as np 
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error
from pylab import *
import scipy.signal
from spectrum import *
warnings.filterwarnings("ignore")
##################################################################
##################################################################
##################################################################
#Finds the ARMA prediction for given data.
def ARMA_Prediction(data):
    a,b, rho = arma_estimate(data, 15, 15, 30)
    print(a)
    return None
##################################################################
CENTER_FREQ = 91.3e6 
SAMPLE_RATE = 5000000
SAMPLE_SIZE = (SAMPLE_RATE//100)
N = 500                                    
data_raw = scipy.fromfile('out_long.dat', dtype=complex)
data_raw *= 10e24
data_uf = np.reshape(data_raw,(-1, SAMPLE_SIZE)).astype(float)
data = scipy.signal.wiener(data_uf,mysize=29)
mean = np.mean(data,axis=1)
mean_rs = scipy.signal.wiener(mean,mysize=29)
mean_rs = np.reshape(mean_rs,(-1, N))
predictions_mean = list()
for index in range(mean_rs.shape[0]):
    temp = ARMA_Prediction(mean_rs[index])
    if np.isnan(temp).any():
        break
    print(temp.shape)
    predictions_mean.append(temp)
predictions_mean = np.asarray(predictions_mean).flatten()
diff1 = np.abs(predictions_mean - mean_rs.flatten())
##################################################################
##################################################################
fig, axs = plt.subplots(3, 1, sharex=True)
fig.subplots_adjust(hspace=0.4,left=0.1,right=0.98)
axs[0].set_title('mean/mean filtered')
axs[0].plot(mean,label='mean',linewidth=1)
axs[0].plot(mean_rs.flatten(),label='mean filtered',linewidth=1,linestyle='dashed')
axs[1].set_title('mean filtered/mean pred')
axs[1].plot(mean,label='mean filtered',linewidth=1,linestyle='dashed')
axs[1].plot(predictions_mean,label='mean_pred',linewidth=1,linestyle='dashed')
axs[2].set_title('diffrence')
axs[2].plot(diff1,label='mean_pred_diff',linewidth=1,linestyle='dashed')
fig.legend(loc=3)
mse = mean_squared_error(mean_rs.flatten(),predictions_mean)
print(mse)
print(diff1.max())
plt.show()