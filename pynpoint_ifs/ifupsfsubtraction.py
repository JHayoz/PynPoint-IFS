"""
Pipeline modules for PSF subtraction.
"""

import sys
import time
import math
import warnings
import copy

from typing import List, Optional, Tuple, Union
from typeguard import typechecked

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from scipy.stats import median_abs_deviation
import numpy as np
from sklearn.decomposition import PCA

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

from pynpoint_ifs.ifu_utils import select_cubes,drizzle_stellar_spectrum,replace_outliers,extract_bright_px,replace_outliers_MAD,extract_star_spectrum_from_sub_bright_px

class IFUPSFSubtractionModuleCugno(ProcessingModule):
    """
    Module to subtract the stellar signal using High-Resolution Spectral Differential Imaging (HRSDI) as in Cugno et al. 2022
    """

    __author__ = 'Gabriele Cugno, Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'psf_subtraction_HRSDI',
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
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output for the PSF subtracted data, after removing outliers. Should be different from *image_in_tag*.
        residuals_out_tag : str
            Tag of the database entry that is written as output for the PSF subtracted data, without removing outliers. Should be different from *image_in_tag*.
        ref_spectrum_out_tag : str
            Tag of the database entry that is written as output for the master stellar spectrum by which the data is divided. Should be different from *image_in_tag*.
        norm_datacube_out_tag : str
            Tag of the database entry that is written as output, for the normalized cube after division by the master stellar spectrum. Should be different from *image_in_tag*.
        smooth_datacube_out_tag : str
            Tag of the database entry that is written as output, for the normalized cube that is smoothed out along the spectral dimension. Should be different from *image_in_tag*.
        psf_model_out_tag : str
            Tag of the database entry that is written as output, for the PSF model. Should be different from *image_in_tag*.
        gauss_sigma : float
            value for the smoothing of the normalized cube
        filter_method : str
            type of smoothing function, savitzky-golay or gaussian filter
        outliers_sigma : float
            sigma to exclude outliers, using median absolute deviation
        stellar_spectrum_method : str
            method to extract the stellar spectrum: 'mean', 'median', 'drizzle'
        spaxel_select_method : str
            method to select the spaxels with which to calculate the stellar spectrum: 'brightest_n', 'all'
        drizzle_factor_upsampling : int
            if 'drizzle' method to extract the stellar spectrum, which factor to upsample the spectrum with
        drizzle_factor_drop : float
            if 'drizzle' method to extract the stellar spectrum, which drop factor to use
        wvl_shift_tag : str
            if 'drizzle' method to extract the stellar spectrum, tag of the database to read as wavelength shift. Should be a wavelength shift for each spaxel, in python coordinates (first axis shift for first axis of data, etc)
        
        Returns
        -------
        NoneType
            None
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
        
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        
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

class IFUContinuumRemovalModule(ProcessingModule):
    """
    Module to subtract the continuum of the spectrum using a gaussian filter.
    """

    __author__ = 'Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'continuum_subtraction',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'PSF_sub',
        gauss_sigma: float = 10.,
        outliers_sigma: float = 3.
    ) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        gauss_sigma : float
            value for the smoothing of the residuals
        outliers_sigma : float
            sigma to exclude outliers
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUContinuumRemovalModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_outliers_sigma = outliers_sigma
    
    def run(self) -> None:
        """
        Run method of the module. Convolves the images with a Gaussian kernel.
        
        Returns
        -------
        NoneType
            None
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
    Module to subtract the PSF using spectral PCA as in Hayoz et al. 2024(5).
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
        method: str = 'full',
        use_mask: bool = False,
        mask_position: List = [],
        mask_radius: float = 6.
    ) -> None:
        """
        Parameters
        ----------
        name_in: str
            Unique name of the module instance.
        image_in_tag: str
            Tag of the database entry that is read as input.
        wv_in_tag: str
            Tag of the database entry that is read as input wavelength axis.
        image_out_tag: str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        pca_out_tag: str
            Tag of the database entry that is written as output for the PCA model.
        pca_number: int
            number of PCA components to remove.
        method: str
            method to model the stellar spectrum, either single, where each cube is analysed separately (x,y) -> F for each t, or full where all cubes are taken into account (t,x,y) -> F
        use_mask: bool
            whether to use a mask
        mask_position: List
            position of the mask
        mask_radius: float
            radius of the mask
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUPCAPSFSubtractionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_pca_out_port = self.add_output_port(pca_out_tag)
        
        self.pca_number = pca_number
        self.m_method = method

        self.m_use_mask = use_mask
        self.m_mask_position = mask_position
        self.m_mask_radius = mask_radius
    
    def run(self) -> None:
        """
        Run method of the module. Convolves the images with a Gaussian kernel.
        
        Returns
        -------
        NoneType
            None
        """
        # to-do
        # allow possibility to shift images before combining, rotate as well
        # should use full time stack to build up model
        # then use it to subtract PSF
        # then shift and derotate images
        # mean or median combine the residuals


        # reshape the input data
        print('Load all data')
        data = self.m_image_in_port.get_all()
        print('Done')
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
                print('Progress %.2f' % (cube_i*100/len_cube),end='\n')
                pix_list_all = np.transpose(datacubes[cube_i],axes=(1,2,0)).reshape((-1,len_wvl))
                if self.m_use_mask:
                    X,Y = np.meshgrid(np.arange(len_x),np.arange(len_y))
                    mask = np.sqrt((X - self.m_mask_position[0] - len_x/2)**2 + (Y - self.m_mask_position[1] - len_y/2)**2) <= self.m_mask_radius
                    mask_cube = np.zeros((len_wvl,len_x,len_y),dtype=bool)
                    for k in range(len_wvl):
                        mask_cube[k,:,:] = mask
                    pix_list_masked = np.transpose(datacubes[cube_i][~mask_cube].reshape((len_wvl,-1)))
                    print('Successfully masked the region')
                else:
                    pix_list_masked = pix_list_all
                
                mask_nans = np.sum(np.isnan(pix_list_masked),axis=1) > 0
                
                pix_list = pix_list_masked[~mask_nans]
                # subtract the mean value first
                spec_mean = np.mean(pix_list,axis=0)
                pix_list_res = pix_list - spec_mean
                # calculate the PCA once, but truncate to the different number of components
                pca_sklearn = PCA(n_components=self.pca_number, svd_solver="arpack",whiten=True)
                
                # check for nans
                assert(np.sum(np.isnan(pix_list_res)) == 0)
                print('Fitting PCA')
                pca_sklearn.fit(pix_list_res)
                pca_representation = pca_sklearn.transform(pix_list_res)
                model_psf_1d = pca_sklearn.inverse_transform(pca_representation)
                model_psf_1d_all = np.zeros_like(pix_list_all)
                model_psf_1d_all[~mask_nans] = model_psf_1d+spec_mean
                # model_psf_1d_all = model_psf_1d+spec_mean
                model_psf = np.transpose((model_psf_1d_all).reshape((len_x,len_y,len_wvl)),axes=(2,0,1))
                residuals = datacubes[cube_i,:,:,:] - model_psf[:,:,:]
                print('Appending residuals and model')
                self.m_image_out_port.append(residuals)
                self.m_pca_out_port.append(model_psf)
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
        pca_number: int = 5
    ) -> None:
        """
        Parameters
        ----------
        name_in: str
            Unique name of the module instance.
        image_in_tag: str
            Tag of the database entry that is read as input.
        stellar_spectra_in_tag: str
            Tag of the database entry that is read as input.
        image_out_tag: str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        pca_number: int
            number of principal components to remove
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUPCAModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.pca_number = pca_number
    
    def run(self) -> None:
        """
        Run method of the module. Convolves the images with a Gaussian kernel.
        
        Returns
        -------
        NoneType
            None
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