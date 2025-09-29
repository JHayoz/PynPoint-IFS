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
from pynpoint.util.image import scale_image, shift_image

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from PyAstronomy.pyasl import dopplerShift
from .ifu_utils import select_cubes,rebin


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

class IFUWavelengthCalibrationModule(ProcessingModule):
    """
    Module to calibrate the wavelength solution based on cross-correlation with telluric lines. The module extracts the RV shift in each spaxel by idientifying the peak of the CCF.
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'calibrate_datacube',
        cc_in_tag: str = 'cc_cube',
        drv_in_tag: str = 'drv',
        rvshift_out_tag: str = 'rvshift'
    ) -> None:
        """
        Constructor of IFUWavelengthCalibrationModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param cc_in_tag: Tag of the database entry that is read as input CCF
        :type cc_in_tag: str
        :param drv_in_tag: Tag of the database entry that is read as input radial velocity axis of the CCF
        :type drv_in_tag: str
        :param rvshift_out_tag: Tag of the database entry that is written as output, in terms of radial velocity shift between the data and the telluric lines.
        :type rvshift_out_tag: str
        
        :return: None
        """
        
        super(IFUWavelengthCalibrationModule, self).__init__(name_in)
        
        
        self.m_cc_in_port = self.add_input_port(cc_in_tag)
        self.m_drv_in_port = self.add_input_port(drv_in_tag)
        self.m_rvshift_out_port = self.add_output_port(rvshift_out_tag)
        
    
    def run(self):
        """
        Run method of the module.
        
        :return: None
        """

        def extract_rv(drv,cc,sigma = 5,drv_range=50):
            def gauss_fct(x,mu,std,a,b):
                return a*np.exp(-0.5*((x-mu)/(std))**2) + b
            smooth_cc = gaussian_filter(cc,sigma)
            max_i = np.argmax(smooth_cc)
            mask_range = np.logical_and(drv > drv[max_i] - drv_range,drv < drv[max_i] + drv_range)
            p0 = [drv[max_i],sigma,max(smooth_cc),0]
            xdata=drv[mask_range]
            ydata=smooth_cc[mask_range]
            try:
                popt,pcov = curve_fit(gauss_fct,xdata=xdata,ydata=ydata,p0=p0)
                return popt[0]
            except RuntimeError:
                print('PARAMETER NOT FOUND, RETURNING NONE')
                return None
        

        cc_cube = self.m_cc_in_port.get_all()
        drv = self.m_drv_in_port.get_all()
        len_cube,len_rv,len_x,len_y = np.shape(cc_cube)
        mean_rv_shift_arr = []
        for cube_i in range(len_cube):
            tell_rv = np.zeros((len_x,len_y))
            for i in range(len_x):
                for j in range(len_y):
                    rv_signal = extract_rv(drv=drv,cc=cc_cube[cube_i,:,i,j],sigma = 5,drv_range=50)
                    tell_rv[i,j] = rv_signal
            mask_nans = np.isnan(tell_rv)
            mean_rv_shift = np.nanmean(tell_rv[~mask_nans])
            tell_rv[mask_nans] = mean_rv_shift
            mean_rv_shift_arr += [mean_rv_shift]
            self.m_rvshift_out_port.append([tell_rv])
        
        self.m_rvshift_out_port.copy_attributes(self.m_cc_in_port)
        self.m_rvshift_out_port.add_history('Radial velocity shift in the cube', 'Mean RV shift: %.2f' % np.nanmean(mean_rv_shift_arr))
        self.m_rvshift_out_port.close_port()

class IFUWavelengthCorrectionModule(ProcessingModule):
    """
    Module to correct the wavelength solution based on an input RV shift.
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'correct_datacube',
        image_in_tag: str = 'datacube',
        wv_in_tag: str = 'wavelength',
        rvshift_in_tag: str = 'rvshift',
        image_out_tag: str = 'datacube_wvl_shifted',
        method: str = 'full'
    ) -> None:
        """
        Constructor of IFUWavelengthCalibrationModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param wv_in_tag: Tag of the database (wavelengths) entry that is read as input.
        :type wv_in_tag: str
        :param rvshift_in_tag: Tag of the database entry that is read as input RV shift for each pixel, either a common RV shift for each spaxel throughout the cube, or one per cube
        :type rvshift_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output, after dopplershifting the cube by the amount specified by the RV shift
        :type image_out_tag: str
        :param method: either 'full' or 'mean': if 'mean', applies the same RV shift for all spaxels, if 'full' applies a different RV shift for each spaxel
        :type method: str
        
        :return: None
        """
        
        super(IFUWavelengthCorrectionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_rvshift_in_port = self.add_input_port(rvshift_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_method = method
    
    def run(self):
        """
        Run method of the module.
        
        :return: None
        """
        wavelength = self.m_wv_in_port.get_all()
        datacubes = select_cubes(self.m_image_in_port.get_all(),wavelength)
        
        len_cube,len_wvl,len_x,len_y = np.shape(datacubes)

        rvshift_cube = self.m_rvshift_in_port.get_all()
        len_cube_rv,len_x_rv,len_y_rv = np.shape(rvshift_cube)
        if self.m_method not in ['full','mean']:
            print('WRONG METHOD. METHOD MUST BE FULL OR MEAN')
            self.m_image_out_port.close_port()
            assert(False)
        if len_x_rv != len_x or len_y_rv != len_x:
            print('INPUT IMAGE AND RV SHIFT HAVE DIFFERENT SHAPES!')
            self.m_image_out_port.close_port()
            assert(False)
        rvshift_cube_tmp = np.zeros((len_cube,len_x,len_y))
        if len_cube_rv != len_cube:
            if len_cube_rv != 1:
                print('INVALID SHAPE FOR RV SHIFT CUBE')
                print('The RV shift cube needs to have the same shape as the input image or a length of 1')
                self.m_image_out_port.close_port()
                assert(False)
            for cube_i in range(len_cube):
                rvshift_cube_tmp[cube_i,:,:] = rvshift_cube[0,:,:]
        else:
            for cube_i in range(len_cube):
                rvshift_cube_tmp[cube_i,:,:] = rvshift_cube[cube_i,:,:]
        
        mean_rv_shift = np.nanmean(rvshift_cube_tmp,axis=(1,2))
        for cube_i in range(len_cube):
            cube_shifted_out = np.zeros((len_wvl,len_x,len_y))
            print('Progress: %.2f' % ((cube_i)/(len_cube)*100),end='\r')
            for i in range(len_x):
                for j in range(len_y):
                    spectrum_cube = datacubes[cube_i,:,i,j]
                    if self.m_method == 'full':
                        rvshift_i = rvshift_cube_tmp[cube_i,i,j]
                    elif self.m_method == 'mean':
                        rvshift_i = mean_rv_shift[cube_i]
                    else:
                        self.m_image_out_port.close_port()
                        assert(False)
                    nflux1, wlprime1 = dopplerShift(wavelength, spectrum_cube, -rvshift_i, edgeHandling="firstlast")
                    cube_shifted_out[:,i,j] = nflux1
            self.m_image_out_port.append(cube_shifted_out)
        
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('Corrected the residual radial velocity shift in the cube', 'Method %s, mean RV: %.2f' % (self.m_method,np.nanmean(mean_rv_shift)))
        self.m_image_out_port.close_port()




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

class IFUTelluricsWavelengthCalibrationModule(ProcessingModule):
    """
    Module to correct the wavelength solution based on cross-correlation with a tellurics transmission spectrum
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        tellurics_wvl: np.array,
        tellurics_flux: np.array,
        name_in: str = 'measure_wavelength_shift',
        image_in_tag: str = 'datacube',
        wv_in_tag: str = 'wavelength',
        wavelength_shift_out_tag: str = 'wavelength_shift',
        cc_accuracy: int = 10
    ) -> None:
        """
        Constructor of IFUTelluricsWavelengthCalibrationModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param wv_in_tag: Tag of the database (wavelengths) entry that is read as input.
        :type wv_in_tag: str
        :param wavelength_shift_out_tag: Tag of the database entry that is written as output wavelength shift for each spaxel
        :type wavelength_shift_out_tag: str
        
        :return: None
        """
        
        super(IFUTelluricsWavelengthCalibrationModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_wavelength_shift_out_tag = self.add_output_port(wavelength_shift_out_tag)
        self.m_tellurics_wvl = tellurics_wvl
        self.m_tellurics_flux = tellurics_flux
        self.m_cc_accuracy = cc_accuracy
    
    def run(self):
        """
        Run method of the module.
        
        :return: None
        """
        wavelength = self.m_wv_in_port.get_all()
        mean_wvl_step = np.mean(wavelength[1:]-wavelength[:-1])
        datacubes = select_cubes(self.m_image_in_port.get_all(),wavelength)
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        len_cube,len_wvl,len_x,len_y = np.shape(datacubes)

        # rebin tellurics to have the same wavelength bins as the data
        wlen_tell_cr,transm_tell_cr = rebin(self.m_tellurics_wvl,self.m_tellurics_flux,wavelength,flux_err = None, method='datalike')
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUTelluricsWavelengthCalibrationModule...', start_time)
            print('')
            cube_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            shift_trsm = np.zeros((len_x,len_y))
            for xi in range(len_x):
                for yi in range(len_y):
                    print('Progress %.2f' % (100*(xi*len_y + yi + 1)/(len_x*len_y)) + '%',end='\r')
                    
                    spectrum = cube_i[:,xi,yi]
                    
                    #assume that the cube is already continuum-removed
                    #smooth_spectrum = gaussian_filter(spectrum,sigma=40)
                    #spectrum_cr = spectrum - smooth_spectrum
                    
                    tmp_offset, _, _ = phase_cross_correlation(
                        spectrum,
                        transm_tell_cr,
                        normalization=None,
                        upsample_factor=self.m_cc_accuracy,
                        overlap_ratio=0.3)
                    
                    shift_trsm[xi,yi] = tmp_offset[0]*mean_wvl_step
            self.m_wavelength_shift_out_tag.append(shift_trsm, data_dim=3)
            
        self.m_wavelength_shift_out_tag.copy_attributes(self.m_image_in_port)
        self.m_wavelength_shift_out_tag.add_history('Measured the wavelength shift in the cube', 'Accuracy: %i' % (self.m_cc_accuracy))
        self.m_wavelength_shift_out_tag.close_port()

