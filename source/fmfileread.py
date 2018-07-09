#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
##################################################
# GNU Radio Python Flow Graph
# Title: Fmfileread
# Generated: Fri Jul  6 16:50:18 2018
# GNU Radio version: 3.7.12.0
##################################################

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print "Warning: failed to XInitThreads()"

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import wxgui
from gnuradio.eng_option import eng_option
from gnuradio.fft import window
from gnuradio.filter import firdes
from gnuradio.wxgui import fftsink2
from grc_gnuradio import wxgui as grc_wxgui
from optparse import OptionParser
import pmt
import wx


class fmfileread(grc_wxgui.top_block_gui):

    def __init__(self):
        grc_wxgui.top_block_gui.__init__(self, title="Fmfileread")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 5e6
        self.lpf_decim = lpf_decim = 20
        self.audio_samp_rate = audio_samp_rate = 96e3
        self.Center_Freq = Center_Freq = 91.3e6

        ##################################################
        # Blocks
        ##################################################
        self.notebook_0 = self.notebook_0 = wx.Notebook(self.GetWin(), style=wx.NB_TOP)
        self.notebook_0.AddPage(grc_wxgui.Panel(self.notebook_0), "fft")
        self.Add(self.notebook_0)
        self.wxgui_fftsink2_0 = fftsink2.fft_sink_c(
        	self.notebook_0.GetPage(0).GetWin(),
        	baseband_freq=Center_Freq,
        	y_per_div=10,
        	y_divs=10,
        	ref_level=0,
        	ref_scale=2.0,
        	sample_rate=samp_rate,
        	fft_size=2048,
        	fft_rate=15,
        	average=False,
        	avg_alpha=None,
        	title='FFT Plot',
        	peak_hold=True,
        )
        self.notebook_0.GetPage(0).Add(self.wxgui_fftsink2_0.win)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, '/home/ken/Rf_anomaly/source/out_long.dat', True)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.wxgui_fftsink2_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.wxgui_fftsink2_0.set_sample_rate(self.samp_rate)

    def get_lpf_decim(self):
        return self.lpf_decim

    def set_lpf_decim(self, lpf_decim):
        self.lpf_decim = lpf_decim

    def get_audio_samp_rate(self):
        return self.audio_samp_rate

    def set_audio_samp_rate(self, audio_samp_rate):
        self.audio_samp_rate = audio_samp_rate

    def get_Center_Freq(self):
        return self.Center_Freq

    def set_Center_Freq(self, Center_Freq):
        self.Center_Freq = Center_Freq
        self.wxgui_fftsink2_0.set_baseband_freq(self.Center_Freq)


def main(top_block_cls=fmfileread, options=None):

    tb = top_block_cls()
    tb.Start(True)
    tb.Wait()


if __name__ == '__main__':
    main()
