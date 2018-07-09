import numpy as np 
import scipy
import matplotlib.pyplot as plt
import warnings
from timeit import default_timer as timer
warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
###################################
def FFT_get(data):
    spectrum = np.fft.fft(data,n=N)
    freq = np.linspace(CENTER_FREQ-(SAMPLE_RATE/2),CENTER_FREQ+(SAMPLE_RATE/2),N)
    window = np.hanning(N)
    spectrum = 10*np.log10(spectrum)
    spectrum = np.abs(spectrum)
    spectrum = spectrum - min(spectrum.flatten())
    threshold = 0.97 * max(spectrum.flatten())
    mask = spectrum > threshold
    mask = mask.any(axis=0)
    peaks = freq[mask]*10e-7
    print(peaks)
    plt.plot(freq.T, spectrum[0] )
    plt.plot(freq.T, spectrum[1] )
    plt.plot(freq.T, spectrum[2] )
    plt.grid(True)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    # plt.xticks([8.95e7,9.03e7,9.13e7,9.19e7,9.23e7])
    
CENTER_FREQ = 91.3e6
SAMPLE_RATE = 5000000#5mil/sec
N = 2**21
SAMPLE_SIZE = (SAMPLE_RATE*2)#10e6
# SAMPLE_SIZE = (SAMPLE_RATE//500)#10e3
N = SAMPLE_SIZE
start = timer()                                             
data_raw = scipy.fromfile('out_long.dat', dtype=complex)    #load data
data = np.reshape(data_raw, (-1, SAMPLE_SIZE))              #resize into SAMPLE_SIZE chunks
print("Sample size: ",SAMPLE_SIZE)
FFT_get(data)                                             #get fft info
end = timer()
print((end - start)/60.0,'mins')
plt.show()