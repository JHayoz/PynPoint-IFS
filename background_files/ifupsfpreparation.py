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

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
#from pynpoint.util.image import shift_image


class IFUStellarSpectrumModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(self,
                 name_in: str = "Select_range",
                 image_in_tag: str = "initial_spectrum",
                 wv_in_tag: str = "wavelengths",
                 spectrum_out_tag: str = "spectrum_selected",
                 wv_out_tag: str = "wavelengths",
                 num_pix: int = 20,
                 std_max: float=0.2,
                 norm_range: Tuple[float, float] = (2.14,2.145)):
        """
            Constructor of IFUStellarSpectrumModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param wv_in_tag: Tag of the database (wavelengths) entry that is read as input.
            :type wv_in_tag: str
            :param spectrum_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type spectrum_out_tag: str
            :param wv_out_tag: Tag of the database entry that is read as input.
            :type wv_out_tag: str
            :param num_pix: number of pixels that should be evaluated to build the stellar spectrum.
            :type num_pix: int
            :param std_max: maximum standard deviation for each specturm to remove outliers.
            :type std_max: float
            :param norm_range: initial and final wavelength range to be used to normalize the spectra. A region without dominant telluric features should be selected.
            :type norm_range: Tuple
            
            :return: None
            """
        
        super(IFUStellarSpectrumModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_spectrum_out_port = self.add_output_port(spectrum_out_tag)
        
        self.m_num_pix = num_pix
        self.m_std = std_max
        self.m_norm_range = norm_range
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """

        def collapse_frames(images):
            return np.mean(images, axis=0)
        
        def find_brightest_pixels(image, num_pix):
            im_new = np.copy(image)
            Count = 0
            max_pos = []
            while Count<num_pix:
                max_pos_i = np.unravel_index(np.argmax(im_new, axis=None), im_new.shape)
                max_pos.append(max_pos_i)
                Count+=1
                im_new[max_pos_i]=0
            return max_pos
        
        def _normalize_spectra(spectra_init, num_pix, norm_range):
            spectra2 = np.zeros_like(spectra_init)
            for p in range(num_pix):
                spectra2[p,:] = spectra_init[p,:]/np.median(spectra_init[p,norm_range[0]:norm_range[1]])
            return spectra2
        
        def _find_outliers(spectra_init, std):
            spectra2 = copy.copy(spectra_init)
            spectrum_f = np.zeros(nspectrum[0])
            for m in range(nspectrum[0]):
                new_arr=spectra2[:,m]
                while np.std(new_arr)>std:
                    m_i = np.mean(new_arr)
                    d_i = np.argmax(np.abs(new_arr-m_i))
                    new_arr = np.delete(new_arr,d_i)
                spectrum_f[m] = np.median(new_arr)
            return spectrum_f
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        wv = self.m_wv_in_port.get_all()
        n_before = 0
        while wv[n_before]<self.m_norm_range[0]:
            n_before+=1
        
        n_after = nspectrum[0]-1
        while wv[n_after]>self.m_norm_range[1]:
            n_after-=1
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUStellarSpectrumModule...', start_time)
            frames_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            cube_collapsed = collapse_frames(frames_i)
            pixels = find_brightest_pixels(cube_collapsed,self.m_num_pix)
            
            pix_spectrum = np.zeros((self.m_num_pix, nspectrum_i))
            for k in range(self.m_num_pix):
                pix_spectrum[k,:] = frames_i[:,pixels[k][0], pixels[k][1]]
            
            pix_spectrum_norm = _normalize_spectra(pix_spectrum, self.m_num_pix, (n_before, n_after))
            
            pix_spectrum_norm_outliers = _find_outliers(pix_spectrum_norm, self.m_std)
            
            self.m_spectrum_out_port.append(pix_spectrum_norm_outliers)
        
        self.m_spectrum_out_port.copy_attributes(self.m_image_in_port)
        self.m_spectrum_out_port.add_history("Stellar Spectrum", "num pixels = "+str(self.m_num_pix))
        self.m_spectrum_out_port.close_port()

