"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import copy
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA

from typing import List, Optional, Tuple, Union

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

from background_files.ifu_utils import select_cubes,overlap_2_arrays,drizzle_2_arrays,dodrizzle_n_arrays,drizzle_stellar_spectrum
#from pynpoint.util.image import shift_image


class IFUPSFSubtractionModule_Jean(ProcessingModule):
    """
    Module to subtract the stellar signal.
    """

    __author__ = 'Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'PSF_sub',
        gauss_sigma: float = 10.,
        outliers_sigma: float = 3.
    ) -> None:
        """
        Constructor of IFUPSFSubtractionModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_out_tag: str
        :param gauss_sigma: value for the smoothing of normalized cube
        :type gauss_sigma: float
        :param outliers_sigma: value for the identification of outliers
        :type outliers_sigma: float
        
        :return: None
        """
        
        super(IFUPSFSubtractionModule_Jean, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_outliers_sigma = outliers_sigma
    
    def run(self):
        """
        Run method of the module. Creates a smooth (smooth along the spectral dimension) stellar PSF, and subtracts it from the data.
        
        :return: None
        """
        def extract_star_spectrum_from_bright_px(datacube,nb_px = 10):
            img = np.mean(datacube,axis=0)
            indices = np.argpartition(img.reshape((-1)), -nb_px)[-nb_px:]
            brightest_pix = np.unravel_index(indices,img.shape)
            master_star_spectrum = np.mean(datacube.transpose()[brightest_pix],axis=0)
            return master_star_spectrum
            
        def replace_outliers(spectrum, sigma):
            spectrum2 = copy.copy(spectrum)
            sub = medfilt(spectrum2, 7)
            sp_sub = spectrum2-sub
            std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
            spectrum2 = np.where(np.abs(sp_sub)> sigma*std, sub, spectrum2)
            return spectrum2
        
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        size = self.m_image_in_port.get_shape()[1]
        
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUPSFSubtractionModule...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            
            # Create reference spectrum
            master_star_spectrum = extract_star_spectrum_from_bright_px(datacube)

            master_star_spectrum = replace_outliers(master_star_spectrum, sigma=self.m_outliers_sigma)
            
            # divide by star spectrum
            norm_datacube = datacube/master_star_spectrum[:,np.newaxis,np.newaxis]
            
            # smooth along wavelength
            smoothed_cube = np.ones_like(datacube)
            len_wvl,len_x,len_y = np.shape(datacube)
            for i in range(len_x):
                for j in range(len_y):
                    smooth_spectrum = gaussian_filter(norm_datacube[:,i,j],self.m_gauss_sigma)
                    outliers_replaced = replace_outliers(smooth_spectrum, sigma=self.m_outliers_sigma)
                    smoothed_cube[:,i,j] = gaussian_filter(outliers_replaced,self.m_gauss_sigma)


            # multiply again with star spectrum
            continuum_PSF = smoothed_cube*master_star_spectrum[:,np.newaxis,np.newaxis]

            # PSf subtraction
            residuals = datacube - continuum_PSF
            
            self.m_image_out_port.append(residuals)

        
            
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PSF sub",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_image_out_port.close_port()

class IFUPSFSubtractionModuleCugno(ProcessingModule):
    """
    Module to subtract the stellar signal.
    """

    __author__ = 'Gabriele Cugno, Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'PSF_sub',
        residuals_out_tag: str = 'PSF_sub_non_outliers',
        ref_spectrum_out_tag: str = 'PSF_sub_ref_spectrum',
        norm_datacube_out_tag: str = 'PSF_sub_normcube',
        smooth_datacube_out_tag: str = 'PSF_sub_smoothcube',
        psf_model_out_tag: str = 'PSF_sub_model',
        gauss_sigma: float = 10.,
        filter_method: str = 'gaussian',
        outliers_sigma: float = 3.,
        stellar_spectrum_method: str ='mean',
        spaxel_select_method: str = 'brightest_10',
        drizzle_factor_upsampling: int = 2,
        drizzle_factor_drop: float = 1.,
        wvl_shift_tag: str = None,
        wvl_in_tag: str = 'wavelength'
    ) -> None:
        """
        Constructor of IFUPSFSubtractionModuleCugno.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output for the PSF subtracted data, after removing outliers. Should be different from *image_in_tag*.
        :type image_out_tag: str
        :param residuals_out_tag: Tag of the database entry that is written as output for the PSF subtracted data, without removing outliers. Should be different from *image_in_tag*.
        :type residuals_out_tag: str
        :param ref_spectrum_out_tag: Tag of the database entry that is written as output for the master stellar spectrum by which the data is divided. Should be different from *image_in_tag*.
        :type ref_spectrum_out_tag: str
        :param norm_datacube_out_tag: Tag of the database entry that is written as output, for the normalized cube after division by the master stellar spectrum. Should be different from *image_in_tag*.
        :type norm_datacube_out_tag: str
        :param smooth_datacube_out_tag: Tag of the database entry that is written as output, for the normalized cube that is smoothed out along the spectral dimension. Should be different from *image_in_tag*.
        :type smooth_datacube_out_tag: str
        :param psf_model_out_tag: Tag of the database entry that is written as output, for the PSF model. Should be different from *image_in_tag*.
        :type psf_model_out_tag: str
        :param gauss_sigma: value for the smoothing of the normalized cube
        :type gauss_sigma: float
        :param filter_method: type of smoothing function, savitzky-golay or gaussian filter
        :type filter_method: str
        :param outliers_sigma: sigma to exclude outliers, using median absolute deviation
        :type outliers_sigma: float
        :param stellar_spectrum_method: method to extract the stellar spectrum: 'mean', 'median', 'drizzle'
        :type stellar_spectrum_method: str
        :param spaxel_select_method: method to select the spaxels with which to calculate the stellar spectrum: 'brightest_n', 'all'
        :type spaxel_select_method: str
        :param drizzle_factor_upsampling: if 'drizzle' method to extract the stellar spectrum, which factor to upsample the spectrum with
        :type drizzle_factor_upsampling: int
        :param drizzle_factor_drop: if 'drizzle' method to extract the stellar spectrum, which drop factor to use
        :type drizzle_factor_drop: float
        :param wvl_shift_tag: if 'drizzle' method to extract the stellar spectrum, tag of the database to read as wavelength shift. Should be a wavelength shift for each spaxel, in python coordinates (first axis shift for first axis of data, etc)
        :type wvl_shift_tag: str
        
        :return: None
        """
        
        super(IFUPSFSubtractionModuleCugno, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_residuals_out_port = self.add_output_port(residuals_out_tag)
        self.m_ref_spectrum_out_port = self.add_output_port(ref_spectrum_out_tag)
        self.m_norm_datacube_out_port = self.add_output_port(norm_datacube_out_tag)
        self.m_smooth_datacube_out_port = self.add_output_port(smooth_datacube_out_tag)
        self.m_psf_model_datacube_out_port = self.add_output_port(psf_model_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_filter_method = filter_method
        self.m_outliers_sigma = outliers_sigma
    
        self.m_stellar_spectrum_method = stellar_spectrum_method
        self.m_spaxel_select_method = spaxel_select_method
        self.m_d_f_upsmpl = drizzle_factor_upsampling
        self.m_d_f_drop = drizzle_factor_drop
        if not wvl_shift_tag is None:
            self.m_wvl_shift_port = self.add_input_port(wvl_shift_tag)
            self.m_wvl_in_port = self.add_input_port(wvl_in_tag)
        else:
            self.m_wvl_shift_port = None
            self.m_wvl_in_port = None
        
    def run(self):
        """
        Run method of the module.
        
        :return: None
        """
        
        def replace_outliers(spectrum, sigma, method='mad'):
            spectrum2 = copy.copy(spectrum)
            sub = medfilt(spectrum2, 13)
            sp_sub = spectrum2-sub
            if method=='mad':
                std = median_abs_deviation(sp_sub)
            elif method=='std':
                std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
            else:
                print('PLEASE USE METHOD==mad or std')
            spectrum2 = np.where(np.abs(sp_sub)> sigma*std, sub, spectrum2)
            return spectrum2
        
        def extract_bright_px(datacube,nb_px = 10):
            img = np.mean(datacube,axis=0)
            indices = np.argpartition(img.reshape((-1)), -nb_px)[-nb_px:]
            brightest_pix = np.unravel_index(indices,img.shape)
            return brightest_pix
            
        def extract_star_spectrum_from_sub_bright_px(datacube,sample = 100,nb_px=10):
            sample = sample
            img = np.mean(datacube,axis=0)
            # pick the brightest sample pixels
            indices = np.argpartition(img.reshape((-1)), -sample)[-sample:]
            # sort them
            sorted_subindices = np.argsort(img[np.unravel_index(indices,np.shape(img))])
            sorted_indices = indices[sorted_subindices]
            # pick the least bright nb_px pixels 
            brightest_pix = np.unravel_index(sorted_indices[:nb_px],np.shape(img))
            return brightest_pix

        def replace_outliers_MAD(spectrum,smooth_sigma=11,sigma=11,replace_method = 'smooth'):
            # identify the outliers by the median absolute deviation, after continuum-subtraction
            smooth = gaussian_filter(spectrum,smooth_sigma)
            sub = spectrum-smooth
            std = median_abs_deviation(sub)
            outliers = np.abs(sub) > sigma * std
            if replace_method == 'smooth':
                # define a function to replace the outliers
                smooth_non_outliers = gaussian_filter(spectrum[~outliers],smooth_sigma)
                x_axis = np.arange(0,len(spectrum))
                smooth_small_x = x_axis[~outliers]
                interped = interp1d(x=smooth_small_x,y=smooth_non_outliers,bounds_error=None,fill_value='extrapolate')
                smooth_small_interped = interped(x_axis)
            
                # correct the spectrum
                spectrum_corr = np.where(outliers,smooth_small_interped,spectrum)
            else:# replace_method == 'zero'
                spectrum_corr = np.where(outliers,0,spectrum)
            return spectrum_corr
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        size = self.m_image_in_port.get_shape()[1]
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUPSFSubtractionModuleCugno...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            
            len_wvl,len_x,len_y = np.shape(datacube)
            #clean_datacube =  np.zeros_like(datacube)
            #for i in range(len_x):
            #    for j in range(len_y):
            #        clean_datacube[:,i,j] = replace_outliers(datacube[:,i,j], sigma=self.m_outliers_sigma)
            
            # Normalize cube by total flux
            total_flux = np.sum(datacube,axis=0)
            total_flux_non_zeros = np.where(total_flux != 0, total_flux, 1)
            #np.where(clean_datacube != 0,np.sum(clean_datacube,axis=0),1)
            norm_datacube = datacube[:,:,:] / total_flux_non_zeros
            
            if self.m_spaxel_select_method == 'all':
                X,Y = np.meshgrid(np.arange(len_x),np.arange(len_y))
                selected_pix = (X.flatten(),Y.flatten())
            elif self.m_spaxel_select_method[:len('brightest_')] == 'brightest_':
                nb_px = int(self.m_spaxel_select_method[len('brightest_'):])
                print('EXTRACTING BRIGHTEST %i PIXELS' % nb_px)
                # find brightest pixels
                selected_pix = extract_bright_px(datacube,nb_px = nb_px)
            else:
                info = self.m_spaxel_select_method[len('sub_brightest_'):]
                sample,nb_px = map(int,info.split('_'))
                print('EXTRACTING SUB-BRIGHTEST %i PIXELS OUT OF %i' % (nb_px,sample))
                # find brightest pixels
                selected_pix = extract_star_spectrum_from_sub_bright_px(datacube,sample = sample,nb_px = nb_px)
                
            
            # Create reference spectrum
            #reference_spectrum = np.median(norm_datacube,axis=(1,2))
            # reference_spectrum = extract_star_spectrum_from_bright_px(norm_datacube,nb_px=10)
            #reference_spectrum = reference_spectrum/np.mean(reference_spectrum)
            if self.m_stellar_spectrum_method == 'mean':
                reference_spectrum_original = np.mean(np.transpose(norm_datacube,axes=(1,2,0))[selected_pix],axis=0)
                mask_invalid = np.logical_or(reference_spectrum_original==0,np.isnan(reference_spectrum_original))
                reference_spectrum_nonzeros = np.where(mask_invalid,1,reference_spectrum_original)
                reference_spectrum = reference_spectrum_nonzeros[:,np.newaxis,np.newaxis]
            elif self.m_stellar_spectrum_method == 'median':
                reference_spectrum_original = np.median(np.transpose(norm_datacube,axes=(1,2,0))[selected_pix],axis=0)
                mask_invalid = np.logical_or(reference_spectrum_original==0,np.isnan(reference_spectrum_original))
                reference_spectrum_nonzeros = np.where(mask_invalid,1,reference_spectrum_original)
                reference_spectrum = reference_spectrum_nonzeros[:,np.newaxis,np.newaxis]
            elif self.m_stellar_spectrum_method == 'sum':
                reference_spectrum_original = np.sum(np.transpose(norm_datacube,axes=(1,2,0))[selected_pix],axis=0)
                mask_invalid = np.logical_or(reference_spectrum_original==0,np.isnan(reference_spectrum_original))
                reference_spectrum_nonzeros = np.where(mask_invalid,1,reference_spectrum_original)
                reference_spectrum = reference_spectrum_nonzeros[:,np.newaxis,np.newaxis]
            elif self.m_stellar_spectrum_method == 'drizzle':
                spectra_axis = np.transpose(norm_datacube,axes=(1,2,0))[selected_pix]
                wavelength = self.m_wvl_in_port[:]
                wvl_shift = self.m_wvl_shift_port[i,:,:]
                wavelength_shifted = wavelength[:,np.newaxis,np.newaxis] - wvl_shift
                wavelength_axis = np.transpose(wavelength_shifted,axes=(1,2,0))[selected_pix]
                
                wvl_drizzled,spectra_drizzled,weights = drizzle_stellar_spectrum(
                                                            wavelength_ref=wavelength,
                                                            wavelength_axis=wavelength_axis,
                                                            spectra_axis=spectra_axis,
                                                            factor_upsampling=self.m_d_f_upsmpl,
                                                            drop_factor=self.m_d_f_drop,
                                                            verbose=True)
                stellar_master_model = interp1d(x = wvl_drizzled, y = spectra_drizzled, bounds_error=False, fill_value = 'extrapolate', kind = 'cubic')
                
                reference_spectrum = stellar_master_model(wavelength_shifted)
                reference_spectrum_original = stellar_master_model(wavelength)
            # divide by star spectrum
            div_datacube = norm_datacube[:,:,:]/reference_spectrum
            #div_datacube = datacube[:,:,:]/reference_spectrum_nonzeros[:,np.newaxis,np.newaxis]
            
            # smooth along wavelength
            smoothed_cube = np.ones_like(datacube)
            continuum_PSF = np.zeros_like(datacube)
            len_wvl,len_x,len_y = np.shape(datacube)
            for i in range(len_x):
                for j in range(len_y):
                    outliers_removed = replace_outliers_MAD(
                        div_datacube[:,i,j].copy(),
                        smooth_sigma=self.m_gauss_sigma,
                        sigma=self.m_outliers_sigma,
                        replace_method='smooth')
                    # outliers_removed = replace_outliers(div_datacube[:,i,j].copy(), sigma=self.m_outliers_sigma)
                    if self.m_filter_method == 'gaussian':
                        smoothed_cube[:,i,j] = gaussian_filter(outliers_removed,self.m_gauss_sigma)
                    elif self.m_filter_method == 'savgol':
                        smoothed_cube[:,i,j] = savgol_filter(x = outliers_removed,window_length = int(self.m_gauss_sigma),polyorder = 3)
                    else:
                        print('CHOOSE FILTER GAUSSIAN OR SAVGOL')
                    
                    # multiply again with star spectrum
                    if self.m_stellar_spectrum_method == 'drizzle':
                        ref_spectrum = reference_spectrum[:,i,j]
                    else:
                        ref_spectrum = reference_spectrum_nonzeros
                    continuum_PSF[:,i,j] = smoothed_cube[:,i,j]*ref_spectrum *total_flux_non_zeros[i,j]

            # PSF subtraction
            residuals = datacube - continuum_PSF
            clean_residuals =  np.zeros_like(residuals)
            for i in range(len_x):
                for j in range(len_y):
                    clean_residuals[:,i,j] = replace_outliers_MAD(
                                                residuals[:,i,j].copy(),
                                                smooth_sigma=self.m_gauss_sigma,
                                                sigma=self.m_outliers_sigma,
                                                replace_method='zero')
            
            
            self.m_residuals_out_port.append(residuals)
            self.m_image_out_port.append(clean_residuals)
            self.m_ref_spectrum_out_port.append(reference_spectrum_original)
            self.m_norm_datacube_out_port.append(div_datacube)
            self.m_smooth_datacube_out_port.append(smoothed_cube)
            self.m_psf_model_datacube_out_port.append(continuum_PSF)

        
            
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PSF sub",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_image_out_port.close_port()

        #self.m_ref_spectrum_out_port.copy_attributes(self.m_image_in_port)
        #self.m_ref_spectrum_out_port.add_history("PSF sub, reference spectrum",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_ref_spectrum_out_port.close_port()

        #self.m_norm_datacube_out_port.copy_attributes(self.m_image_in_port)
        #self.m_norm_datacube_out_port.add_history("PSF sub, normalized datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_norm_datacube_out_port.close_port()

        #self.m_smooth_datacube_out_port.copy_attributes(self.m_image_in_port)
        #self.m_smooth_datacube_out_port.add_history("PSF sub, smooth datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_smooth_datacube_out_port.close_port()

        #self.m_residuals_out_port.copy_attributes(self.m_image_in_port)
        #self.m_residuals_out_port.add_history("PSF sub, smooth datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_residuals_out_port.close_port()


class IFUPSFSubtractionModuleLandman(ProcessingModule):
    """
    Module to subtract the stellar signal.
    """

    __author__ = 'Gabriele Cugno, Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'PSF_sub',
        residuals_out_tag: str = 'PSF_sub_non_outliers',
        ref_spectrum_out_tag: str = 'PSF_sub_ref_spectrum',
        norm_datacube_out_tag: str = 'PSF_sub_normcube',
        smooth_datacube_out_tag: str = 'PSF_sub_smoothcube',
        psf_model_out_tag: str = 'PSF_sub_model',
        gauss_sigma: float = 10.,
        filter_method: str = 'gaussian',
        outliers_sigma: float = 3.,
        stellar_spectrum_method: str ='mean',
        spaxel_select_method: str = 'brightest_10'
    ) -> None:
        """
        Constructor of IFUPSFSubtractionModuleCugno.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output for the PSF subtracted data, after removing outliers. Should be different from *image_in_tag*.
        :type image_out_tag: str
        :param residuals_out_tag: Tag of the database entry that is written as output for the PSF subtracted data, without removing outliers. Should be different from *image_in_tag*.
        :type residuals_out_tag: str
        :param ref_spectrum_out_tag: Tag of the database entry that is written as output for the master stellar spectrum by which the data is divided. Should be different from *image_in_tag*.
        :type ref_spectrum_out_tag: str
        :param norm_datacube_out_tag: Tag of the database entry that is written as output, for the normalized cube after division by the master stellar spectrum. Should be different from *image_in_tag*.
        :type norm_datacube_out_tag: str
        :param smooth_datacube_out_tag: Tag of the database entry that is written as output, for the normalized cube that is smoothed out along the spectral dimension. Should be different from *image_in_tag*.
        :type smooth_datacube_out_tag: str
        :param psf_model_out_tag: Tag of the database entry that is written as output, for the PSF model. Should be different from *image_in_tag*.
        :type psf_model_out_tag: str
        :param gauss_sigma: value for the smoothing of the normalized cube
        :type gauss_sigma: float
        :param filter_method: type of smoothing function, savitzky-golay or gaussian filter
        :type filter_method: str
        :param outliers_sigma: sigma to exclude outliers, using median absolute deviation
        :type outliers_sigma: float
        :param stellar_spectrum_method: method to extract the stellar spectrum: 'mean', 'median', 'drizzle'
        :type stellar_spectrum_method: str
        :param spaxel_select_method: method to select the spaxels with which to calculate the stellar spectrum: 'brightest_n', 'all'
        :type spaxel_select_method: str
        :param drizzle_factor_upsampling: if 'drizzle' method to extract the stellar spectrum, which factor to upsample the spectrum with
        :type drizzle_factor_upsampling: int
        :param drizzle_factor_drop: if 'drizzle' method to extract the stellar spectrum, which drop factor to use
        :type drizzle_factor_drop: float
        :param wvl_shift_tag: if 'drizzle' method to extract the stellar spectrum, tag of the database to read as wavelength shift. Should be a wavelength shift for each spaxel, in python coordinates (first axis shift for first axis of data, etc)
        :type wvl_shift_tag: str
        
        :return: None
        """
        
        super(IFUPSFSubtractionModuleLandman, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_residuals_out_port = self.add_output_port(residuals_out_tag)
        self.m_ref_spectrum_out_port = self.add_output_port(ref_spectrum_out_tag)
        self.m_norm_datacube_out_port = self.add_output_port(norm_datacube_out_tag)
        self.m_smooth_datacube_out_port = self.add_output_port(smooth_datacube_out_tag)
        self.m_psf_model_datacube_out_port = self.add_output_port(psf_model_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_filter_method = filter_method
        self.m_outliers_sigma = outliers_sigma
    
        self.m_stellar_spectrum_method = stellar_spectrum_method
        self.m_spaxel_select_method = spaxel_select_method
        
    def run(self):
        """
        Run method of the module.
        
        :return: None
        """
        
        def replace_outliers(spectrum, sigma, method='mad'):
            spectrum2 = copy.copy(spectrum)
            sub = medfilt(spectrum2, 13)
            sp_sub = spectrum2-sub
            if method=='mad':
                std = median_abs_deviation(sp_sub)
            elif method=='std':
                std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
            else:
                print('PLEASE USE METHOD==mad or std')
            spectrum2 = np.where(np.abs(sp_sub)> sigma*std, sub, spectrum2)
            return spectrum2
        
        def extract_bright_px(datacube,nb_px = 10):
            img = np.mean(datacube,axis=0)
            indices = np.argpartition(img.reshape((-1)), -nb_px)[-nb_px:]
            brightest_pix = np.unravel_index(indices,img.shape)
            return brightest_pix
            
        def extract_star_spectrum_from_sub_bright_px(datacube,sample = 100,nb_px=10):
            sample = sample
            img = np.mean(datacube,axis=0)
            # pick the brightest sample pixels
            indices = np.argpartition(img.reshape((-1)), -sample)[-sample:]
            # sort them
            sorted_subindices = np.argsort(img[np.unravel_index(indices,np.shape(img))])
            sorted_indices = indices[sorted_subindices]
            # pick the least bright nb_px pixels 
            brightest_pix = np.unravel_index(sorted_indices[:nb_px],np.shape(img))
            return brightest_pix

        def replace_outliers_MAD(spectrum,smooth_sigma=11,sigma=11,replace_method = 'smooth'):
            # identify the outliers by the median absolute deviation, after continuum-subtraction
            smooth = gaussian_filter(spectrum,smooth_sigma)
            sub = spectrum-smooth
            std = median_abs_deviation(sub)
            outliers = np.abs(sub) > sigma * std
            if replace_method == 'smooth':
                # define a function to replace the outliers
                smooth_non_outliers = gaussian_filter(spectrum[~outliers],smooth_sigma)
                x_axis = np.arange(0,len(spectrum))
                smooth_small_x = x_axis[~outliers]
                interped = interp1d(x=smooth_small_x,y=smooth_non_outliers,bounds_error=None,fill_value='extrapolate')
                smooth_small_interped = interped(x_axis)
            
                # correct the spectrum
                spectrum_corr = np.where(outliers,smooth_small_interped,spectrum)
            else:# replace_method == 'zero'
                spectrum_corr = np.where(outliers,0,spectrum)
            return spectrum_corr
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        size = self.m_image_in_port.get_shape()[1]
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUPSFSubtractionModuleCugno...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            
            len_wvl,len_x,len_y = np.shape(datacube)
            #clean_datacube =  np.zeros_like(datacube)
            #for i in range(len_x):
            #    for j in range(len_y):
            #        clean_datacube[:,i,j] = replace_outliers(datacube[:,i,j], sigma=self.m_outliers_sigma)
            
            # Normalize cube by total flux
            total_flux = np.sum(datacube,axis=0)
            total_flux_non_zeros = np.where(total_flux != 0, total_flux, 1)
            #np.where(clean_datacube != 0,np.sum(clean_datacube,axis=0),1)
            norm_datacube = datacube[:,:,:] / total_flux_non_zeros
            
            if self.m_spaxel_select_method == 'all':
                X,Y = np.meshgrid(np.arange(len_x),np.arange(len_y))
                selected_pix = (X.flatten(),Y.flatten())
            elif self.m_spaxel_select_method[:len('brightest_')] == 'brightest_':
                nb_px = int(self.m_spaxel_select_method[len('brightest_'):])
                print('EXTRACTING BRIGHTEST %i PIXELS' % nb_px)
                # find brightest pixels
                selected_pix = extract_bright_px(datacube,nb_px = nb_px)
            else:
                info = self.m_spaxel_select_method[len('sub_brightest_'):]
                sample,nb_px = map(int,info.split('_'))
                print('EXTRACTING SUB-BRIGHTEST %i PIXELS OUT OF %i' % (nb_px,sample))
                # find brightest pixels
                selected_pix = extract_star_spectrum_from_sub_bright_px(datacube,sample = sample,nb_px = nb_px)
                
            
            # Create reference spectrum
            #reference_spectrum = np.median(norm_datacube,axis=(1,2))
            # reference_spectrum = extract_star_spectrum_from_bright_px(norm_datacube,nb_px=10)
            #reference_spectrum = reference_spectrum/np.mean(reference_spectrum)
            if self.m_stellar_spectrum_method == 'mean':
                reference_spectrum_original = np.mean(np.transpose(norm_datacube,axes=(1,2,0))[selected_pix],axis=0)
                mask_invalid = np.logical_or(reference_spectrum_original==0,np.isnan(reference_spectrum_original))
                reference_spectrum = np.where(mask_invalid,1,reference_spectrum_original)
            elif self.m_stellar_spectrum_method == 'median':
                reference_spectrum_original = np.median(np.transpose(norm_datacube,axes=(1,2,0))[selected_pix],axis=0)
                mask_invalid = np.logical_or(reference_spectrum_original==0,np.isnan(reference_spectrum_original))
                reference_spectrum = np.where(mask_invalid,1,reference_spectrum_original)
            else:
                print('ERROR')
            smooth_ref_spectrum = np.zeros_like(reference_spectrum)
            if self.m_filter_method == 'gaussian':
                smooth_ref_spectrum = gaussian_filter(reference_spectrum,self.m_gauss_sigma)
            elif self.m_filter_method == 'savgol':
                smooth_ref_spectrum = savgol_filter(x = reference_spectrum,window_length = int(self.m_gauss_sigma),polyorder = 3)
            else:
                print('CHOOSE FILTER GAUSSIAN OR SAVGOL')
            
            # smooth along wavelength
            smoothed_cube = np.ones_like(datacube)
            continuum_PSF = np.zeros_like(datacube)
            len_wvl,len_x,len_y = np.shape(datacube)
            for i in range(len_x):
                for j in range(len_y):
                    outliers_removed = replace_outliers_MAD(
                        datacube[:,i,j].copy(),
                        smooth_sigma=self.m_gauss_sigma,
                        sigma=self.m_outliers_sigma,
                        replace_method='smooth')
                    # outliers_removed = replace_outliers(div_datacube[:,i,j].copy(), sigma=self.m_outliers_sigma)
                    if self.m_filter_method == 'gaussian':
                        smoothed_cube[:,i,j] = gaussian_filter(outliers_removed,self.m_gauss_sigma)
                    elif self.m_filter_method == 'savgol':
                        smoothed_cube[:,i,j] = savgol_filter(x = outliers_removed,window_length = int(self.m_gauss_sigma),polyorder = 3)
                    else:
                        print('CHOOSE FILTER GAUSSIAN OR SAVGOL')
                    
                    # multiply again with star spectrum
                    if self.m_stellar_spectrum_method == 'drizzle':
                        ref_spectrum = reference_spectrum[:,i,j]
                    else:
                        ref_spectrum = reference_spectrum_original
                    continuum_PSF[:,i,j] = smoothed_cube[:,i,j]/smooth_ref_spectrum
            
            PSF_model = continuum_PSF*reference_spectrum[:,np.newaxis,np.newaxis]
            # PSF subtraction
            residuals = datacube - PSF_model
            clean_residuals =  np.zeros_like(residuals)
            for i in range(len_x):
                for j in range(len_y):
                    clean_residuals[:,i,j] = replace_outliers_MAD(
                                                residuals[:,i,j].copy(),
                                                smooth_sigma=self.m_gauss_sigma,
                                                sigma=self.m_outliers_sigma,
                                                replace_method='zero')
            
            
            self.m_residuals_out_port.append(residuals)
            self.m_image_out_port.append(clean_residuals)
            self.m_ref_spectrum_out_port.append(reference_spectrum)
            self.m_norm_datacube_out_port.append(continuum_PSF)
            self.m_smooth_datacube_out_port.append(smoothed_cube)
            self.m_psf_model_datacube_out_port.append(PSF_model)

        
            
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PSF sub",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_image_out_port.close_port()

        #self.m_ref_spectrum_out_port.copy_attributes(self.m_image_in_port)
        #self.m_ref_spectrum_out_port.add_history("PSF sub, reference spectrum",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_ref_spectrum_out_port.close_port()

        #self.m_norm_datacube_out_port.copy_attributes(self.m_image_in_port)
        #self.m_norm_datacube_out_port.add_history("PSF sub, normalized datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_norm_datacube_out_port.close_port()

        #self.m_smooth_datacube_out_port.copy_attributes(self.m_image_in_port)
        #self.m_smooth_datacube_out_port.add_history("PSF sub, smooth datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_smooth_datacube_out_port.close_port()

        #self.m_residuals_out_port.copy_attributes(self.m_image_in_port)
        #self.m_residuals_out_port.add_history("PSF sub, smooth datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_residuals_out_port.close_port()

class IFUPSFSubtractionModuleHaffert(ProcessingModule):
    """
    Module to subtract the stellar signal.
    """

    __author__ = 'Gabriele Cugno, Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'PSF_sub',
        residuals_out_tag: str = 'PSF_sub_non_outliers',
        ref_spectrum_out_tag: str = 'PSF_sub_ref_spectrum',
        norm_datacube_out_tag: str = 'PSF_sub_normcube',
        smooth_datacube_out_tag: str = 'PSF_sub_smoothcube',
        psf_model_out_tag: str = 'PSF_sub_model',
        gauss_sigma: float = 10.,
        filter_method: str = 'gaussian',
        outliers_sigma: float = 3.
    ) -> None:
        """
        Constructor of IFUPSFSubtractionModuleHaffert.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output for the PSF subtracted data, after removing outliers. Should be different from *image_in_tag*.
        :type image_out_tag: str
        :param residuals_out_tag: Tag of the database entry that is written as output for the PSF subtracted data, without removing outliers. Should be different from *image_in_tag*.
        :type residuals_out_tag: str
        :param ref_spectrum_out_tag: Tag of the database entry that is written as output for the master stellar spectrum by which the data is divided. Should be different from *image_in_tag*.
        :type ref_spectrum_out_tag: str
        :param norm_datacube_out_tag: Tag of the database entry that is written as output, for the normalized cube after division by the master stellar spectrum. Should be different from *image_in_tag*.
        :type norm_datacube_out_tag: str
        :param smooth_datacube_out_tag: Tag of the database entry that is written as output, for the normalized cube that is smoothed out along the spectral dimension. Should be different from *image_in_tag*.
        :type smooth_datacube_out_tag: str
        :param psf_model_out_tag: Tag of the database entry that is written as output, for the PSF model. Should be different from *image_in_tag*.
        :type psf_model_out_tag: str
        :param gauss_sigma: value for the smoothing of the normalized cube
        :type gauss_sigma: float
        :param filter_method: type of smoothing function, savitzky-golay or gaussian filter
        :type filter_method: str
        :param outliers_sigma: sigma to exclude outliers, using median absolute deviation
        :type outliers_sigma: float
        
        :return: None
        """
        
        super(IFUPSFSubtractionModuleHaffert, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_residuals_out_port = self.add_output_port(residuals_out_tag)
        self.m_ref_spectrum_out_port = self.add_output_port(ref_spectrum_out_tag)
        self.m_norm_datacube_out_port = self.add_output_port(norm_datacube_out_tag)
        self.m_smooth_datacube_out_port = self.add_output_port(smooth_datacube_out_tag)
        self.m_psf_model_datacube_out_port = self.add_output_port(psf_model_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_filter_method = filter_method
        self.m_outliers_sigma = outliers_sigma
    
    def run(self):
        """
        Run method of the module.
        
        :return: None
        """
        
        def replace_outliers(spectrum, sigma):
            spectrum2 = copy.copy(spectrum)
            sub = medfilt(spectrum2, 13)
            sp_sub = spectrum2-sub
            std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
            spectrum2 = np.where(np.abs(sp_sub)> sigma*std, sub, spectrum2)
            return spectrum2
        def extract_star_spectrum_from_bright_px(datacube,nb_px = 10):
            img = np.mean(datacube,axis=0)
            indices = np.argpartition(img.reshape((-1)), -nb_px)[-nb_px:]
            brightest_pix = np.unravel_index(indices,img.shape)
            master_star_spectrum = np.mean(datacube.transpose()[brightest_pix],axis=0)
            return master_star_spectrum
            
        def extract_star_spectrum_from_sub_bright_px(datacube,nb_px = 10):
            sample = 10*nb_px
            img = np.mean(datacube,axis=0)
            # pick the brightest sample pixels
            indices = np.argpartition(img.reshape((-1)), -sample)[-sample:]
            # sort them
            sorted_subindices = np.argsort(img[np.unravel_index(indices,np.shape(img))])
            sorted_indices = indices[sorted_subindices]
            # pick the least bright nb_px pixels 
            brightest_pix = np.unravel_index(sorted_indices[:nb_px],np.shape(img))
            master_star_spectrum = np.median(datacube.transpose()[brightest_pix],axis=0)
            return master_star_spectrum
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        size = self.m_image_in_port.get_shape()[1]
        
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUPSFSubtractionModuleCugno...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            len_wvl,len_x,len_y = np.shape(datacube)
            #clean_datacube =  np.zeros_like(datacube)
            #for i in range(len_x):
            #    for j in range(len_y):
            #        clean_datacube[:,i,j] = replace_outliers(datacube[:,i,j], sigma=self.m_outliers_sigma)
            
            # Normalize cube by total flux
            total_flux = np.sum(datacube,axis=0)
            total_flux_non_zeros = np.where(total_flux != 0, total_flux, 1)
            
            # np.where(clean_datacube != 0,np.sum(clean_datacube,axis=0),1)
            norm_datacube = datacube[:,:,:] / total_flux_non_zeros
            
            # Create reference spectrum
            reference_spectrum = np.median(norm_datacube,axis=(1,2))
            #reference_spectrum = extract_star_spectrum_from_sub_bright_px(datacube)
            #reference_spectrum = reference_spectrum/np.mean(reference_spectrum)
            
            # divide by star spectrum
            div_datacube = norm_datacube[:,:,:]/reference_spectrum[:,np.newaxis,np.newaxis]
            
            # smooth along wavelength
            smoothed_cube = np.ones_like(datacube)
            len_wvl,len_x,len_y = np.shape(datacube)
            for i in range(len_x):
                for j in range(len_y):
                    outliers_removed = replace_outliers(div_datacube[:,i,j].copy(), sigma=self.m_outliers_sigma)
                    if self.m_filter_method == 'gaussian':
                        smoothed_cube[:,i,j] = gaussian_filter(outliers_removed,self.m_gauss_sigma)
                    elif self.m_filter_method == 'savgol':
                        smoothed_cube[:,i,j] = savgol_filter(x = outliers_removed,window_length = int(self.m_gauss_sigma),polyorder = 3)
                    else:
                        print('CHOOSE FILTER GAUSSIAN OR SAVGOL')
            
            continuum_PSF = norm_datacube/smoothed_cube
            # multiply again with star spectrum
            # continuum_PSF = smoothed_cube[:,:,:]*reference_spectrum[:,np.newaxis,np.newaxis]

            # PSF subtraction
            residuals = datacube - continuum_PSF
            clean_residuals =  np.zeros_like(residuals)
            for i in range(len_x):
                for j in range(len_y):
                    clean_residuals[:,i,j] = replace_outliers(residuals[:,i,j], sigma=self.m_outliers_sigma)
            
            
            self.m_residuals_out_port.append(residuals)
            self.m_image_out_port.append(clean_residuals)
            self.m_ref_spectrum_out_port.append(reference_spectrum)
            self.m_norm_datacube_out_port.append(div_datacube)
            self.m_smooth_datacube_out_port.append(smoothed_cube)
            self.m_psf_model_datacube_out_port.append(continuum_PSF)

        
            
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PSF sub",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_image_out_port.close_port()

        #self.m_ref_spectrum_out_port.copy_attributes(self.m_image_in_port)
        #self.m_ref_spectrum_out_port.add_history("PSF sub, reference spectrum",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_ref_spectrum_out_port.close_port()

        #self.m_norm_datacube_out_port.copy_attributes(self.m_image_in_port)
        #self.m_norm_datacube_out_port.add_history("PSF sub, normalized datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_norm_datacube_out_port.close_port()

        #self.m_smooth_datacube_out_port.copy_attributes(self.m_image_in_port)
        #self.m_smooth_datacube_out_port.add_history("PSF sub, smooth datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_smooth_datacube_out_port.close_port()

        #self.m_residuals_out_port.copy_attributes(self.m_image_in_port)
        #self.m_residuals_out_port.add_history("PSF sub, smooth datacube",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_residuals_out_port.close_port()


class IFUContinuumRemovalModule(ProcessingModule):
    """
    Module to subtract the continuum of the spectrum.
    """

    __author__ = 'Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'PSF_sub',
        gauss_sigma: float = 10.,
        outliers_sigma: float = 3.
    ) -> None:
        """
        Constructor of IFUPSFSubtractionModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_out_tag: str
        :param gauss_sigma: value for the smoothing of the residuals
        :type gauss_sigma: float
        :param outliers_sigma: sigma to exclude outliers
        :type outliers_sigma: float
        
        :return: None
        """
        
        super(IFUContinuumRemovalModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_outliers_sigma = outliers_sigma
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        def replace_outliers(spectrum, sigma):
            spectrum2 = copy.copy(spectrum)
            sub = medfilt(spectrum2, 7)
            sp_sub = spectrum2-sub
            std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
            spectrum2 = np.where(np.abs(sp_sub)> sigma*std, sub, spectrum2)
            return spectrum2

        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        size = self.m_image_in_port.get_shape()[1]
        
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUContinuumRemovalModule...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            # smooth along wavelength
            smoothed_cube = np.ones_like(datacube)
            len_wvl,len_x,len_y = np.shape(datacube)
            for i in range(len_x):
                for j in range(len_y):
                    smooth_spectrum = gaussian_filter(datacube[:,i,j],self.m_gauss_sigma)
                    outliers_replaced = replace_outliers(smooth_spectrum, sigma=self.m_outliers_sigma)
                    smoothed_cube[:,i,j] = gaussian_filter(outliers_replaced,self.m_gauss_sigma)


            # PSf subtraction
            residuals = datacube - smoothed_cube
            
            self.m_image_out_port.append(residuals)

        
            
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Continuum sub",'sigma = %.2f' % self.m_gauss_sigma)
        self.m_image_out_port.close_port()

class IFUPCAPSFSubtractionModule(ProcessingModule):
    """
    Module to subtract the continuum of the spectrum.
    """

    __author__ = 'Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        wv_in_tag: str = 'wavelength',
        image_out_tag: str = 'PSF_sub',
        pca_out_tag: str = 'PCA_model',
        pca_number: int = 5,
        method: str = 'full'
    ) -> None:
        """
        Constructor of IFUPCAPSFSubtractionModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param wv_in_tag: Tag of the database entry that is read as input wavelength axis.
        :type wv_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_out_tag: str
        :param pca_number: number of PCA components to remove.
        :type pca_number: int
        :param method: method to model the stellar spectrum, either single, where each cube is analysed separately (x,y) -> F for each t, or full where all cubes are taken into account (t,x,y) -> F
        :type method: std
        
        :return: None
        """
        
        super(IFUPCAPSFSubtractionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_pca_out_port = self.add_output_port(pca_out_tag)
        
        self.pca_number = pca_number
        self.m_method = method
    
    def run(self):
        """
        Run method of the module. Convolves the images with a Gaussian kernel.
        
        :return: None
        """
        # to-do
        # allow possibility to shift images before combining, rotate as well
        # should use full time stack to build up model
        # then use it to subtract PSF
        # then shift and derotate images
        # mean or median combine the residuals


        # reshape the input data
        data = self.m_image_in_port.get_all()
        wavelength = self.m_wv_in_port.get_all()
        datacubes = select_cubes(data,wavelength)
        len_cube,len_wvl,len_x,len_y = np.shape(datacubes)

        if self.m_method == 'full':
            # reshape: time, spatial x and y all into one axis, second axis is spectrum
            # output: (time x X x Y, wvl)
            pix_list_all = np.transpose(datacubes,axes=(0,2,3,1)).reshape((-1,len_wvl))
            mask_nans = np.sum(np.isnan(pix_list_all),axis=1) > 0
            pix_list = pix_list_all[~mask_nans]
            
            # subtract the mean value first
            spec_mean = np.mean(pix_list,axis=0)
            pix_list_res = pix_list - spec_mean
            
            # calculate the PCA once, but truncate to the different number of components
            pca_sklearn = PCA(n_components=self.pca_number, svd_solver="arpack",whiten=True)
            pca_sklearn.fit(pix_list_res)
            pca_representation = pca_sklearn.transform(pix_list_res)
            model_psf_1d = pca_sklearn.inverse_transform(pca_representation)
            model_psf_1d_all = np.zeros_like(pix_list_all)
            model_psf_1d_all[~mask_nans] = model_psf_1d+spec_mean
            model_psf = np.transpose(model_psf_1d_all.reshape((len_cube,len_x,len_y,len_wvl)),axes=(0,3,1,2))
            residuals = np.zeros_like(datacubes)
            for cube_i in range(len_cube):
                psf_model = model_psf[cube_i,:,:,:]
                residuals[cube_i,:,:,:] = datacubes[cube_i,:,:,:] - psf_model[:,:,:]
            self.m_image_out_port.set_all(residuals.reshape((-1,len_x,len_y)))
            self.m_pca_out_port.set_all(model_psf.reshape((-1,len_x,len_y)))
        elif self.m_method == 'single':
            for cube_i in range(len_cube):
                print('Progress %.2f' % (cube_i*100/len_cube),end='\r')
                pix_list_all = np.transpose(datacubes[cube_i],axes=(1,2,0)).reshape((-1,len_wvl))
                mask_nans = np.sum(np.isnan(pix_list_all),axis=1) > 0
                pix_list = pix_list_all[~mask_nans]
                # subtract the mean value first
                spec_mean = np.mean(pix_list,axis=0)
                pix_list_res = pix_list - spec_mean
                print('Data ready',end='\r')
                # calculate the PCA once, but truncate to the different number of components
                pca_sklearn = PCA(n_components=self.pca_number, svd_solver="arpack",whiten=True)
                pca_sklearn.fit(pix_list_res)
                print('PCA fitted',end='\r')
                pca_representation = pca_sklearn.transform(pix_list_res)
                print('PCA transform',end='\r')
                model_psf_1d = pca_sklearn.inverse_transform(pca_representation)
                print('Transform inverted',end='\r')
                model_psf_1d_all = np.zeros_like(pix_list_all)
                model_psf_1d_all[~mask_nans] = model_psf_1d+spec_mean
                model_psf = np.transpose((model_psf_1d_all).reshape((len_x,len_y,len_wvl)),axes=(2,0,1))
                residuals = np.zeros_like(datacubes[cube_i])
                print('Model calculated',end='\r')
                psf_model = model_psf[:,:,:]
                residuals[:,:,:] = datacubes[cube_i,:,:,:] - psf_model[:,:,:]
                print('PSF subtracted',end='\r')
                self.m_image_out_port.append(residuals)
                print('Residuals added',end='\r')
                self.m_pca_out_port.append(psf_model)
                print('PSF model added',end='\r')
        elif self.m_method == 'normalize_full':
            
            total_flux = np.sum(datacubes,axis=1)
            lencube,lenx,leny = np.shape(total_flux)
            total_flux_non_zeros = np.where(total_flux != 0, total_flux, 1)
            norm_datacube = datacubes[:,:,:,:] / np.reshape(total_flux_non_zeros,(lencube,1,lenx,leny))
                
            pix_list_all = np.transpose(norm_datacube,axes=(0,2,3,1)).reshape((-1,len_wvl))
            mask_nans = np.sum(np.isnan(pix_list_all),axis=1) > 0
            pix_list = pix_list_all[~mask_nans]
            
            # subtract the mean value first
            spec_mean = np.mean(pix_list,axis=0)
            pix_list_res = pix_list - spec_mean
            
            # calculate the PCA once, but truncate to the different number of components
            pca_sklearn = PCA(n_components=self.pca_number, svd_solver="arpack",whiten=True)
            pca_sklearn.fit(pix_list_res)
            pca_representation = pca_sklearn.transform(pix_list_res)
            model_psf_1d = pca_sklearn.inverse_transform(pca_representation)
            model_psf_1d_all = np.zeros_like(pix_list_all)
            model_psf_1d_all[~mask_nans] = model_psf_1d+spec_mean
            model_psf = np.transpose(model_psf_1d_all.reshape((len_cube,len_x,len_y,len_wvl)),axes=(0,3,1,2))
            residuals = np.zeros_like(datacubes)
            for cube_i in range(len_cube):
                psf_model = model_psf[cube_i,:,:,:]*np.reshape(total_flux_non_zeros[cube_i],(1,lenx,leny))
                residuals[cube_i,:,:,:] = datacubes[cube_i,:,:,:] - psf_model[:,:,:]
                self.m_pca_out_port.append(psf_model)
            self.m_image_out_port.set_all(residuals.reshape((-1,len_x,len_y)))
        elif self.m_method == 'normalize_single':
            for cube_i in range(len_cube):
                datacube = datacubes[cube_i]
                total_flux = np.sum(datacube,axis=0)
                total_flux_non_zeros = np.where(total_flux != 0, total_flux, 1)
                norm_datacube = datacube[:,:,:] / total_flux_non_zeros
                
                pix_list_all = np.transpose(norm_datacube,axes=(1,2,0)).reshape((-1,len_wvl))
                mask_nans = np.sum(np.isnan(pix_list_all),axis=1) > 0
                pix_list = pix_list_all[~mask_nans]
                # subtract the mean value first
                spec_mean = np.mean(pix_list,axis=0)
                pix_list_res = pix_list - spec_mean
                
                # calculate the PCA once, but truncate to the different number of components
                pca_sklearn = PCA(n_components=self.pca_number, svd_solver="arpack",whiten=True)
                pca_sklearn.fit(pix_list_res)
                
                pca_representation = pca_sklearn.transform(pix_list_res)
                model_psf_1d = pca_sklearn.inverse_transform(pca_representation)
                model_psf_1d_all = np.zeros_like(pix_list_all)
                model_psf_1d_all[~mask_nans] = model_psf_1d+spec_mean
                model_psf = np.transpose((model_psf_1d_all).reshape((len_x,len_y,len_wvl)),axes=(2,0,1))
                residuals = np.zeros_like(datacube)
                
                psf_model = model_psf[:,:,:]*total_flux_non_zeros
                residuals[:,:,:] = datacubes[cube_i,:,:,:] - psf_model[:,:,:]
                self.m_image_out_port.append(residuals)
                self.m_pca_out_port.append(psf_model)
        else:
            print('CHOOSE METHOD = full or single')
            assert(False)
        """ it's also better to just go with one component
        for pca_n in pca_numbers:
            if pca_n == max_comp:
                continue
            pca_representation_trunc = np.zeros_like(pca_representation)
            pca_representation_trunc[:,:pca_n] = pca_representation[:,:pca_n]
            model_psf_1d = pca_sklearn.inverse_transform(pca_representation_trunc)
            model_psf = np.transpose(model_psf_1d.reshape((len_cube,len_x,len_y,len_wvl)),axes=(0,3,1,2))

            psf_model = np.mean(model_psf[cube_i,:,:,:] + spec_mean[:,np.newaxis,np.newaxis],axis=(0))
            residual = datacubes[cube_i,:,:,:] - model_psf[cube_i,:,:,:] - spec_mean[:,np.newaxis,np.newaxis]
            
            # it's actually better to save the residuals first, then shift and derotate separately, and finally stack separately
            if self.m_shift:
                # shift the residuals
                shift_params = self.m_shift_in_port.get_all()
                x_shift = shift_params[:,0]
                y_shift = shift_params[:,2]
                # make image bigger in case shift is too big
                biggest_shift = max[np.max(x_shift),np.max(y_shift)]
                newlenx,newleny=len_x + biggest_shift,len_y + biggest_shift
                res_reshaped = np.zeros((len_cube,len_wvl,newlenx,newleny))
                res_reshaped[:,:,new_lenx//2-len_x//2:new_lenx//2+len_x//2,new_leny//2-len_y//2:new_leny//2+len_y//2] = residual[:,:,:,:]
                res_shifted = np.zeros((len_cube,len_wvl,newlenx,newleny))
                for cube_i in range(len_cube):
                    for wvl_i in range(len_wvl):
                        x_shift_i = x_shift[wvl_i + cube_i*len_wvl]
                        y_shift_i = y_shift[wvl_i + cube_i*len_wvl]
                        res_shifted[cube_i,wvl_i,:,:] = shift(res_reshaped[cube_i,wvl_i,:,:],(-y_shift_i,-x_shift_i))
            else:
                res_shifted=residual
            if self.m_derotate:
                # derotate the residuals
                parang = -1.*self.m_image_in_port.get_attribute('PARANG')
                if len(parang) == len_cube:
                    parang_resh = np.array([parang[cube_i] for cube_i in range(len_cube) for wvl_i in range(len_wvl)])
                elif len(parang) == len_cube*len_wvl:
                    parang_resh = parang
                else:
                    print('SHAPE OF PARANG DOESNT MATCH THE SIZE OF THE DATACUBES')
                    assert(False)
                res_shifted_rot = np.zeros_like(res_shifted)
                for cube_i in range(len_cube):
                    for wvl_i in range(len_wvl):
                        angle = parang_resh[wvl_i + cube_i*len_wvl]
                        res_shifted_rot[cube_i,wvl_i,:,:] = rotate(res_shifted[cube_i,wvl_i,:,:],angle,reshape=False)
        """
        print('Pipeline finished',end='\r')
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        print('Attributes copied',end='\r')
        self.m_image_out_port.add_history("PCA PSF subtraction",'PC = %i' % self.pca_number)
        print('History added',end='\r')
        self.m_image_out_port.close_port()
        #self.m_pca_out_port.copy_attributes(self.m_image_in_port)
        #self.m_pca_out_port.add_history("PCA PSF subtraction",'PC = %i' % self.pca_number)
        self.m_pca_out_port.close_port()
        print('Ports closed',end='\r')

class IFUPCAModule(ProcessingModule):
    """
    Module to subtract the remaining detector/instrumental effects after PSF subtraction. In principle same as ADIPCA, but without ADI. Redundant since also implemented in IFUPCAPSFSubtractionModule with the method = 'full'
    """

    __author__ = 'Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        wv_in_tag: str = 'wavelength',
        image_out_tag: str = 'PSF_sub',
        pca_number: int = 5,
        method: str = 'full'
        #method_stack: str = 'mean',
        #shift: bool = False,
        #shift_in_tag: str = None,
        #derotate: bool = False
    ) -> None:
        """
        Constructor of IFUPCAModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param stellar_spectra_in_tag: Tag of the database entry that is read as input.
        :type stellar_spectra_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_out_tag: str
        :param gauss_sigma: value for the smoothing of the residuals
        :type gauss_sigma: float
        :param sigma: sigma to exclude outliers
        :type sigma: float
        
        :return: None
            """
        
        super(IFUPCAModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.pca_number = pca_number
        self.m_method = method
        #self.m_method_stack = method_stack
        #self.m_shift = shift
        #self.m_derorate = derotate
        #if self.m_shift:
        #    self.m_shift_in_port = self.add_input_port(shift_in_tag)
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        # to-do
        # allow possibility to shift images before combining, rotate as well
        # should use full time stack to build up model
        # then use it to subtract PSF
        # then shift and derotate images
        # mean or median combine the residuals


        # reshape the input data
        data = self.m_image_in_port.get_all()
        wavelength = self.m_wv_in_port.get_all()
        datacubes = select_cubes(data,wavelength)
        len_cube,len_wvl,len_x,len_y = np.shape(datacubes)
        
        
        # reshape: time, spatial x and y all into one axis, second axis is spectrum
        # reshape: spatial into one axis, spectral and time is other axis
        # output: (X x Y, time x wvl)
        pix_list = np.transpose(datacubes,axes=(2,3,1,0)).reshape((len_x*len_y*len_wvl,-1))
        
        #pix_list = np.transpose(datacubes,axes=(0,2,3,1)).reshape((-1,len_wvl))
        
        # subtract the mean value first
        picture_mean = np.mean(pix_list,axis=0)
        pix_list_res = pix_list - picture_mean
        
        # calculate the PCA once, but truncate to the different number of components
        pca_sklearn = PCA(n_components=self.pca_number, svd_solver="arpack",whiten=True)
        pca_sklearn.fit(pix_list_res)
        pca_representation = pca_sklearn.transform(pix_list_res)
        model_1d = pca_sklearn.inverse_transform(pca_representation)
        
        model = np.transpose((model_1d + picture_mean).reshape((len_x,len_y,len_wvl,len_cube)),axes=(3,2,0,1))
        residuals = np.zeros_like(datacubes)
        for cube_i in range(len_cube):
            psf_model = model[cube_i,:,:,:]
            residuals[cube_i,:,:,:] = datacubes[cube_i,:,:,:] - psf_model[:,:,:]
        self.m_image_out_port.set_all(residuals.reshape((-1,len_x,len_y)))
        
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PCA subtraction",'PC = %i' % self.pca_number)
        self.m_image_out_port.close_port()