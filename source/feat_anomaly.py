import numpy as np 
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from timeit import default_timer as timer
from scipy.fftpack import fft, fftfreq, fftshift
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
###################################
#Finds the ARMA prediction for given data.
def ARMA_Prediction(data, order=(2,0)):
    model = sm.tsa.ARMA(data, order)
    result = model.fit(trend='c',disp=0)
    return np.asarray(result.predict())

#############VAR#############
#160MB raw is 1sec of capture data.
#outlong = 5sec
#outlonger = 25sec
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000#5mil/sec
N = 2**21
# SAMPLE_SIZE = (SAMPLE_RATE*2)#10e6
SAMPLE_SIZE = (SAMPLE_RATE//500)#10e3
#############MAIN#############
start = timer()                                             
data_raw = scipy.fromfile('out_long.dat', dtype=complex)    #load data
data = np.reshape(data_raw, (-1, SAMPLE_SIZE))              #resize into SAMPLE_SIZE chunks
data = data[:500]                                            #limit data size
data = scipy.signal.wiener(data)                            #apply wiener filter to reduce noise
print(data.shape) 
std = np.std(data,axis=1)
mean = np.mean(data,axis=1)
skew = scipy.stats.skew(data,axis=1)
kurt = scipy.stats.kurtosis(data,axis=1)
predictions_mean = ARMA_Prediction(mean,order=(5,0))
predictions_mean2 = ARMA_Prediction(mean,order=(6,0))
predictions_mean3 = ARMA_Prediction(mean,order=(7,0))
predictions_mean4 = ARMA_Prediction(mean,order=(4,0))
predictions_std = ARMA_Prediction(std,order=(2,0))
predictions_skew = ARMA_Prediction(skew,order=(2,0))
predictions_kurt = ARMA_Prediction(kurt,order=(2,0))
mse = mean_squared_error(mean, predictions_mean)
print(abs(mse))
fig, axs = plt.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0.5)

axs[0].plot(mean,label='mean',linewidth=1)
axs[0].set_title('mean')
axs[0].plot(predictions_mean,label='mean_pred',linewidth=1,linestyle='dashed')
axs[0].plot(predictions_mean2,label='mean_pred2',linewidth=1,linestyle='dashed')
axs[0].plot(predictions_mean3,label='mean_pred3',linewidth=1,linestyle='dashed')
axs[0].plot(predictions_mean4,label='mean_pred4',linewidth=1,linestyle='dashed')
# axs[1].plot(mean,label='mean',linewidth=1)
axs[1].set_title('diffrence')
axs[1].plot(predictions_mean - mean,label='mean_pred',linewidth=1,linestyle='dashed')
axs[1].plot(predictions_mean2 - mean ,label='mean_pred2',linewidth=1,linestyle='dashed')
axs[1].plot(predictions_mean3 - mean,label='mean_pred3',linewidth=1,linestyle='dashed')
axs[1].plot(predictions_mean4 - mean,label='mean_pred4',linewidth=1,linestyle='dashed')
fig.legend(loc=3)
# fig, axs = plt.subplots(4, 1, sharex=True)
# fig.subplots_adjust(hspace=0.5)
# axs[0].plot(mean,label='mean',linewidth=1)
# axs[0].set_title('mean')
# axs[0].plot(predictions_mean,label='mean_pred',linewidth=1,linestyle='dashed')
# axs[1].plot(std,label='std')
# axs[1].set_title('std')
# axs[1].plot(predictions_std,label='std_pred',linewidth=1,linestyle='dashed')
# axs[2].plot(skew,label='skew')
# axs[2].set_title('skew')
# axs[2].plot(predictions_skew,label='skew_pred',linewidth=1,linestyle='dashed')
# axs[3].plot(kurt,label='kurt')
# axs[3].set_title('kurt')
# axs[3].plot(predictions_kurt,label='kurt_pred',linewidth=1,linestyle='dashed')
# fig.legend()
end = timer()
print((end - start)/60.0,'mins')
plt.show()