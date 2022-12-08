"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import copy

from typing import Union, Tuple

import numpy as np
from PyAstronomy.pyasl import crosscorrRV

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import shift_image, rotate




class CrossCorrelationModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(self,
                 name_in: str = "CrossCorr",
                 RV: float = 2500.,
                 dRV: float = 10.,
                 image_in_tag: str = "data_mask",
                 data_wv_in_tag: str = "wavelength_range",
                 model_wv: np.ndarray = None,
                 model_abs: np.ndarray = None,
                 mask_in_tag: str = "mask",
                 snr_map_out_tag: str = "snr",
                 CC_cube_out_tag: str = "CC_cube"):
        """
            Constructor of CrossCorrelationModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param RV: limit radial velocity
            :type RV: float
            :param dRV: step radial velocity
            :type dRV: float
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param data_wv_in_tag: Tag of the database entry that is read as input.
            :type data_wv_in_tag: str
            :param model_wv: wavelength array for the model.
            :type model_wv: ndarray
            :param model_abs: flux array for the model.
            :type model_abs: ndarray
            :param snr_map_out_tag: Tag of the database entry that is written as output.
            :type snr_map_out_tag: str
            :param CC_cube_out_tag: Tag of the database entry that is written as output.
            :type CC_cube_out_tag: str
            
            :return: None
            """
        
        super(CrossCorrelationModule, self).__init__(name_in)
        
        self.m_data_wv_in_port = self.add_input_port(data_wv_in_tag)
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_mask_in_port = self.add_input_port(mask_in_tag)
        
        self.m_snr_map_out_port = self.add_output_port(snr_map_out_tag)
        self.m_CC_cube_out_port = self.add_output_port(CC_cube_out_tag)
        
        self.m_RV = RV
        self.m_dRV = dRV
        self.m_wv_model = model_wv
        self.m_abs_model = model_abs
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        size = self.m_image_in_port.get_shape()[-1]
        
        wv = self.m_data_wv_in_port.get_all()
        mask = self.m_mask_in_port.get_all()
        CC_prep = self.m_image_in_port.get_all()
        
        
        def _CC(wv, spectrum, wv_model, model):
            rv, cc = crosscorrRV(w = wv, f = spectrum/np.sum(np.abs(spectrum)), tw = wv_model, tf = model,rvmin=-self.m_RV, rvmax = self.m_RV+self.m_dRV, drv = self.m_dRV,skipedge=0)
            #rv, cc = crosscorrRV(w = wv, f = spectrum, tw = wv_model, tf = model,rvmin=-self.m_RV, rvmax = self.m_RV+self.m_dRV, drv = self.m_dRV,skipedge=0)
            fit = np.polyfit(rv,cc, 1)
            val = np.polyval(fit, rv)
            cc = cc-val
            N_arr = np.concatenate((cc[:np.argmax(cc)-15], cc[np.argmax(cc)+15:]))
            snr = (np.max(cc)-np.mean(N_arr))/np.std(N_arr)
            return cc, rv[np.argmax(cc)], snr
        
        
        CC_cube = np.zeros((int(2*self.m_RV/self.m_dRV+1), size, size))
        rv = np.zeros((size, size))
        snr = np.zeros((size, size))
        
        start_time = time.time()
        for i in range(size):
            for j in range(size):
                progress(i*size+j, size**2, 'Running CrossCorrelationModule...', start_time)
                if mask[0,i,j]!=0:
                    CC_cube_prov, rv[i,j], snr[i,j] = _CC(wv, CC_prep[:,i,j], self.m_wv_model, self.m_abs_model)
                    CC_cube[:,i,j] = CC_cube_prov/np.sum(np.abs(CC_cube_prov))
        
        
        self.m_CC_cube_out_port.set_all(CC_cube)
        self.m_snr_map_out_port.set_all(snr)
        
        
        
        self.m_snr_map_out_port.copy_attributes(self.m_image_in_port)
        self.m_snr_map_out_port.add_history("CC prep", "CC prep")
        
        self.m_CC_cube_out_port.copy_attributes(self.m_image_in_port)
        self.m_CC_cube_out_port.add_history("CC prep", "CC prep")
        
        self.m_snr_map_out_port.close_port()


