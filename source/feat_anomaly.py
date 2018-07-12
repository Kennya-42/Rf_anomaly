#!/usr/bin/python3
# feat_anomaly.py
# Author: Ken Alexopoulos
# Script to look at features in rf data
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
###################################
#Finds the ARMA prediction for given data.
def ARMA_Prediction(data, order=(2,0)):
    model = sm.tsa.ARMA(data, order)
    result = model.fit(trend='c',disp=0,tol=1e-24)
    return np.asarray(result.predict())

CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000#5mil/sec
SAMPLE_SIZE = (SAMPLE_RATE//100)#10e3
#############MAIN#################                                          
data_raw = scipy.fromfile('out_long.dat', dtype=complex)        #load data
data_raw *= 10e24
data_uf = np.reshape(data_raw,(-1, SAMPLE_SIZE)).astype(float)  #resize into SAMPLE_SIZE chunks
for size in range(35,65,2):
    for x in range(12):
        for y in range(12):
            try:
                data = scipy.signal.wiener(data_uf,mysize=size)                    #apply wiener filter to reduce noise
                mean = np.mean(data,axis=1)
                mean_rs = scipy.signal.wiener(mean,mysize=size)
                mean_rs = np.reshape(mean_rs,(-1, 25))
                predictions_mean = list()
                for index in range(mean_rs.shape[0]):
                    predictions_mean.append( ARMA_Prediction(mean_rs[index],order=(x,y)) )
                predictions_mean = np.asarray(predictions_mean).flatten()
                diff1 = np.abs(predictions_mean - mean_rs.flatten())
                print(diff1.max())
                print(size,x,y)
            except:
                print('fail:',size,x,y)
# fig, axs = plt.subplots(3, 1, sharex=True)
# fig.subplots_adjust(hspace=0.4,left=0.1,right=0.98)
# axs[0].set_title('mean/mean filtered')
# axs[0].plot(mean,label='mean',linewidth=1)
# axs[0].plot(mean_rs.flatten(),label='mean filtered',linewidth=1,linestyle='dashed')
# axs[1].set_title('mean filtered/mean pred')
# axs[1].plot(mean,label='mean filtered',linewidth=1,linestyle='dashed')
# axs[1].plot(predictions_mean,label='mean_pred',linewidth=1,linestyle='dashed')
# axs[2].set_title('diffrence')
# axs[2].plot(diff1,label='mean_pred_diff',linewidth=1,linestyle='dashed')
# fig.legend(loc=3)
# mse = mean_squared_error(mean_rs.flatten(),predictions_mean)
# print(mse)
# print(diff1.max())
# plt.show()