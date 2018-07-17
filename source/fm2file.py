#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
##################################################
# GNU Radio Python Flow Graph
# Title: Fm2File
# Generated: Wed Jun 27 16:24:13 2018
# GNU Radio version: 3.7.12.0
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import time


class fm2file(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Fm2File")
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
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(('', "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(Center_Freq, 0)
        self.uhd_usrp_source_0.set_gain(45, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, 2400000000)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/ken/Rf_anomaly/source/out_longest.dat', False)
        self.blocks_file_sink_0.set_unbuffered(False)
        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_head_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

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
        self.uhd_usrp_source_0.set_center_freq(self.Center_Freq, 0)

def main(top_block_cls=fm2file, options=None):

    tb = top_block_cls()
    tb.start()
    tb.wait()

if __name__ == '__main__':
    main()
