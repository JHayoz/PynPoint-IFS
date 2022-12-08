"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings

from typing import Union, Tuple

import numpy as np
from PyAstronomy.pyasl import dopplerShift

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

class SelectWavelengthRangeModule(ProcessingModule):
    """
    Module to select spectral channels based on the wavelength.
    """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(self,
                 range_f: Tuple[float, float],
                 range_i: Tuple[float, float] = (1.92854,2.47171),
                 name_in: str = "Select_range",
                 image_in_tag: str = "initial_spectrum",
                 image_out_tag: str = "spectrum_selected",
                 wv_out_tag: str = "wavelengths"):
        """
            Constructor of SelectWavelengthRangeModule.
            
            :param range_f: final wavelegth range for the selected frames
            :type range_f: tuple(float,float)
            :param range_i: initial wavelegth range for the cube
            :type range_i: tuple(float,float)
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            :param wv_out_tag: Tag of the database entry for the wavelength that is written as output. Should be different from *image_in_tag*.
            :type wv_out_tag: str
            
            :return: None
            """
        
        super(SelectWavelengthRangeModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_wv_out_port = self.add_output_port(wv_out_tag)
        
        self.m_range_f = range_f
        self.m_range_i = range_i
    
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        
        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        
        spectrum_arr = np.linspace(self.m_range_i[0], self.m_range_i[-1], nframes[0])
        n_before = 0
        while spectrum_arr[n_before]<self.m_range_f[0]:
            n_before+=1
        
        n_after = nframes[0]-1
        while spectrum_arr[n_after]>self.m_range_f[1]:
            n_after-=1
        
        start_time = time.time()
        for i, nframes_i in enumerate(nframes):
            progress(i, len(nframes), 'SelectWavelengthRangeModule...', start_time)
            frames_i = self.m_image_in_port[i*nframes_i+n_before:i*nframes_i+n_after,:,:]
            self.m_image_out_port.append(frames_i)

        
        nframes_final = np.ones(len(nframes), dtype=int)*len(spectrum_arr[n_before: n_after])
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_attribute("NFRAMES",nframes_final, False)
        self.m_image_out_port.add_history("Select Spectrum", "lambda range = "+str(self.m_range_f))
        
        self.m_wv_out_port.set_all(spectrum_arr[n_before: n_after])
        self.m_wv_out_port.add_history("Select Spectrum", "lambda range = "+str(self.m_range_f))
        self.m_wv_out_port.close_port()




class CorrectWavelengthModule(ProcessingModule):
    """
        Module to select spectral channels based on the wavelength.
        """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(self,
                 name_in: str = "Correct_wl",
                 wv_in_tag: str = "initial_spectrum",
                 wv_out_tag: str = "spectrum_selected",
                 shift_km_s = 0.):
        """
            Constructor of CorrectWavelengthModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param wv_in_tag: Tag of the database entry that is read as input.
            :type wv_in_tag: str
            :param wv_out_tag: Tag of the database entry for the wavelength that is written as output. Should be different from *image_in_tag*.
            :type wv_out_tag: str
            
            :return: None
            """
        
        super(CorrectWavelengthModule, self).__init__(name_in)
        
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_wv_out_port = self.add_output_port(wv_out_tag)
    
        self.m_shift_km_s = shift_km_s
    
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        wv = self.m_wv_in_port.get_all()
        
        _, wl_shift = dopplerShift(wv, np.ones(len(wv)), self.m_shift_km_s)
        
        
        self.m_wv_out_port.set_all(wl_shift)
        self.m_wv_out_port.copy_attributes(self.m_wv_in_port)
        self.m_wv_out_port.add_history("Select Shifted", "RV = "+str(self.m_shift_km_s))
        self.m_wv_out_port.close_port()


