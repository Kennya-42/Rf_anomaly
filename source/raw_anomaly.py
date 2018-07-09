import numpy as np 
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from timeit import default_timer as timer
import skrf as rf
warnings.filterwarnings("ignore")
################PROCESS###################
SAMPLE_RATE = 5000000#5mil/sec
SAMPLE_SIZE = (SAMPLE_RATE//1000)#5000
start = timer()
data = scipy.fromfile('out_longer.dat', dtype=complex)
data = data.astype(float)
samples_n = data.shape[0]
data = data[:samples_n//2]
filtered = scipy.signal.wiener(data)
################PLOT######################
plt.plot(data,label='raw',linewidth=1)
plt.plot(filtered,label='filtered',linewidth=1,linestyle='dashed')
plt.legend()
plt.title('complex ARMA+wiener filter')
end = timer()
print((end - start)/60.0,'mins')
plt.show()
