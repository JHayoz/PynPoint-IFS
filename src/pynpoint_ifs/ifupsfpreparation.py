"""
Pipeline modules for PSF preparation (for PSF reference or PSF subtraction)
"""

import sys
import time
import math
import warnings
import copy

from typing import Union, Tuple
from typeguard import typechecked

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from PyAstronomy.pyasl import dopplerShift

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import scale_image, shift_image

from pynpoint_ifs.ifu_utils import select_cubes,rebin


class IFUStellarSpectrumModule(ProcessingModule):
    """
    Module to extract the stellar spectrum.
    """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'Select_range',
        image_in_tag: str = 'initial_spectrum',
        wv_in_tag: str = 'wavelengths',
        spectrum_out_tag: str = 'spectrum_selected',
        wv_out_tag: str = 'wavelengths',
        num_pix: int = 20,
        std_max: float=0.2,
        norm_range: Tuple[float, float] = (2.14,2.145)
    ) -> None:
        """
        Parameters
        ----------
        
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        wv_in_tag : str
            Tag of the database (wavelengths) entry that is read as input.
        spectrum_out_tag : str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        wv_out_tag : str
            Tag of the database entry that is read as input.
        num_pix : int
            number of pixels that should be evaluated to build the stellar spectrum.
        std_max : float
            maximum standard deviation for each specturm to remove outliers.
        norm_range : Tuple
            initial and final wavelength range to be used to normalize the spectra. A region without dominant telluric features should be selected.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUStellarSpectrumModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_spectrum_out_port = self.add_output_port(spectrum_out_tag)
        
        self.m_num_pix = num_pix
        self.m_std = std_max
        self.m_norm_range = norm_range
    
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
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
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
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
        self.m_spectrum_out_port.add_history('Stellar Spectrum', 'num pixels = '+str(self.m_num_pix))
        self.m_spectrum_out_port.close_port()



class IFUSDIpreparationModule(ProcessingModule):
    """
    UNFINISHED, NOT FULLY THOUGHT THROUGH
    Module for preparing the cube to do Spectral Differential Imaging.
    To-do: test and correctly describe the parameters. Not sure if useful for ERIS IFS data.
    """

    __author__ = 'Jean Hayoz'

    @typechecked
    def __init__(
        self,
        name_in: str,
        image_in_tag: str,
        wv_in_tag: str,
        image_out_tag: str,
        method: str = 'up'
    ) -> None:
        """
        Parameters
        ----------
        
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from
            *image_in_tag*.
        wavelength : tuple(float, float)
            The central wavelengths of the line and continuum filter, (line, continuum), in
            arbitrary but identical units.
        width : tuple(float, float)
            The equivalent widths of the line and continuum filter, (line, continuum), in
            arbitrary but identical units.

        Returns
        -------
        NoneType
            None
        """

        super().__init__(name_in)

        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_method = method

    @typechecked
    def run(self) -> None:
        """
        Run method of the module. Normalizes the images for the different filter widths,
        upscales the images, and crops the images to the initial image shape in order to
        align the PSF patterns.

        Returns
        -------
        NoneType
            None
        """
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        wv = self.m_wv_in_port.get_all()
        if self.m_method=='up':
            wvl_factor_arr = wv[-1]/wv
        else:
            # wvl_factor_arr = wv/wv[0] # corresponds to scaling up the wrong way
            wvl_factor_arr = wv/wv[-1]
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUStellarSpectrumModule...', start_time)
            frames_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            scaled_frames = np.zeros_like(frames_i)
            if self.m_method == 'up':
                scaled_frames[-1,:,:] = frames_i[-1,:,:]
            else:
                #scaled_frames[0,:,:] = frames_i[0,:,:]
                scaled_frames[0,:,:] = frames_i[0,:,:]
            for wvl_i in range(len(wv)-1):
                if self.m_method == 'up':
                    index = wvl_i
                else:
                    index = wvl_i + 1
                image = frames_i[index]
                im_scale = scale_image(image, wvl_factor_arr[index], wvl_factor_arr[index])
                
                npix_del = im_scale.shape[-1] - image.shape[-1]
                
                if npix_del % 2 == 0:
                    npix_del_a = int(npix_del/2)
                    npix_del_b = int(npix_del/2)

                else:
                    npix_del_a = int((npix_del-1)/2)
                    npix_del_b = int((npix_del+1)/2)
                
                if npix_del_b == 0:
                    im_crop = im_scale[npix_del_a:, npix_del_a:]
                else:
                    im_crop = im_scale[npix_del_a:-npix_del_b, npix_del_a:-npix_del_b]
                
                if npix_del % 2 == 1:
                    im_crop = shift_image(im_crop, (-0.5, -0.5), interpolation='spline')
                scaled_frames[index,:,:] = im_crop[:,:]

            self.m_image_out_port.append(scaled_frames, data_dim=3)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        history = '(Maximum scaling: %.2f, Method: %s, Pixscale: ' % (wvl_factor_arr[-1],self.m_method)
        if self.m_method == 'down':
            pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
            new_pixscale = pixscale / wvl_factor_arr[-1]
            history += '%.4f)' % new_pixscale
            self.m_image_out_port.add_attribute('PIXSCALE',new_pixscale,static=True)
        else:
            history += '%.4f)' % self.m_image_in_port.get_attribute('PIXSCALE')
        
        self.m_image_out_port.add_history('IFUSDIpreparationModule', history)
        self.m_image_in_port.close_port()


