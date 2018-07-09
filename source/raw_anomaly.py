#!/usr/bin/python3
# raw_anomaly.py
# Author: Ken Alexopoulos
# Script to look at raw rf data
import numpy as np 
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
import skrf as rf
warnings.filterwarnings("ignore")
################PROCESS###################
SAMPLE_RATE = 5000000#5mil/sec
SAMPLE_SIZE = (SAMPLE_RATE//1000)#5000
data = scipy.fromfile('out_longer.dat', dtype=complex)
# data = data.astype(float)
samples_n = int(data.shape[0]//25)
data = data[:samples_n]
# filtered = scipy.signal.wiener(data)
################PLOT######################
# plt.plot(data,label='raw',linewidth=1)
# plt.plot(filtered,label='filtered',linewidth=1,linestyle='dashed')
# plt.title('complex ARMA+wiener filter')
# plt.legend()
# plt.show()
