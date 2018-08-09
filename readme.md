RF Anomaly detection for attention switching using retina downsampling

rf_func: main functions for time series and frequency analysis 
rf_anomaly: main driver file for both time series and frequency analysis
feat_anomaly: driver file for time series feature extraction testing
fft_anomaly: driver file to test fft features
rf_iris: driver file to test downsampling

*.dat raw rf timeseries data

testradio/src/* files for gnuradio flow charts
pybombs run gnuradio-companion #to run gnuradio companion to edit and run flow charts
pybombs run python *.py #to run just the python script for a flow chart

dependencies: #latest should work from pip
numpy
scipy
sklearn
statsmodels
matplotlib

for gnuradio:
requires uhd for gnuradio and gnuradio
only got it to work via pybombs 
http://files.ettus.com/manual/page_python.html