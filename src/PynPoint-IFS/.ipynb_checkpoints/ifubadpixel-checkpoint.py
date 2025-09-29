"""
Pipeline modules for the detection and interpolation of bad pixels.
"""

import copy
import warnings

from typing import Union, Tuple

import cv2
import numpy as np
import time

from numba import jit
from typeguard import typechecked
from scipy.ndimage import generic_filter,gaussian_filter
from scipy.interpolate import interp1d
from scipy.stats import median_abs_deviation

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress



class NanFilterModule(ProcessingModule):
    """
    Module to find NaNs in the image and substitute them.
    """
    
    __author__ = 'Gabriele Cugno'

    @typechecked
    def __init__(self,
                 name_in: str = "Substitute_NaNs",
                 image_in_tag: str = "im_arr",
                 image_out_tag: str = "im_arr_Nan",
                 local: bool = False,
                 local_size: int = 3):
        """
            Constructor of NanFilterModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            :param local: Boolean to decide if the substitution occurs with the median of the image or with the median estimated from the surrounding pixels.
            :type local: bool
            
            :return: None
            """
        
        super(NanFilterModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_local = local
        self.m_local_size = local_size


    def run(self):
        """
            Run method of the module. Look for NaNs or zeros and substitute them.
            
            :return: None
            """


        def clean_NaNs(image, ind):
            size0=np.shape(image)[0] # 3rd dim
            size1=np.shape(image)[1] #  x or y axis
            if self.m_local:
                im_new = generic_filter(image, np.nanmedian, size=self.m_local_size) # if local = true, we replace the pixel with the median of the surrounding pixels
            else:
                im_new = np.nanmedian(image) # if local = false
            mask_bads = np.logical_or(~np.isfinite(image),image==0.)
            image_out = np.where(mask_bads,im_new,image)
            
            return image_out


        self.apply_function_to_images(clean_NaNs,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running NanFilterModule...")



        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("NaN removed", "sub with mean ")
        self.m_image_out_port.close_port()

class OutlierCorrectionModule(ProcessingModule):
    """
    Module to find outliers along the spectral dimension in the cube and substitute them.
    """
    
    __author__ = 'Jean Hayoz'

    @typechecked
    def __init__(self,
                 name_in: str = 'substitute_NaNs',
                 image_in_tag: str = 'im_arr',
                 image_out_tag: str = 'im_arr_Nan',
                 outlier_sigma: float = 8.,
                 filter_sigma: float = 11.,
                 replace_method: str = 'smooth',
                 cpu: int = 10):
        """
        Constructor of OutlierCorrectionModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_out_tag: str
        :param outlier_sigma: number of sigmas used to tag outliers. The "sigma" is calculated by the median absolute deviation.
        :type outlier_sigma: float
        :param filter_sigma: standard deviation of the Gaussian filter used to smoothe the spectrum.
        :type filter_sigma: float
        :param replace_method: if 'smooth', then replaces outliers by the smoothed spectrum, else replaces outliers by 0
        :type replace_method: str
        :param cpu: if cpu > 1, then parallelise the operation using joblib
        :type cpu: int
        
        :return: None
        """
        
        super(OutlierCorrectionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_outlier_sigma = outlier_sigma
        self.m_filter_sigma = filter_sigma
        self.m_replace_method = replace_method
        self.m_cpu = cpu


    def run(self):
        """
        Run method of the module. Look for outliers and substitute them.
        
        :return: None
        """

        def replace_outliers_MAD(spectrum,filter_sigma=11,outlier_sigma=11,replace_method = 'smooth'):
            # identify the outliers by the median absolute deviation, after continuum-subtraction
            smooth = gaussian_filter(spectrum,filter_sigma)
            sub = spectrum-smooth
            std = median_abs_deviation(sub)
            outliers = np.abs(sub) > outlier_sigma * std
            if replace_method == 'smooth':
                # define a function to replace the outliers
                smooth_non_outliers = gaussian_filter(spectrum[~outliers],filter_sigma)
                x_axis = np.arange(0,len(spectrum))
                smooth_small_x = x_axis[~outliers]
                interped = interp1d(x=smooth_small_x,y=smooth_non_outliers,bounds_error=None,fill_value='extrapolate')
                smooth_small_interped = interped(x_axis)
            
                # correct the spectrum
                spectrum_corr = np.where(outliers,smooth_small_interped,spectrum)
            else:# replace_method == 'zero'
                spectrum_corr = np.where(outliers,0,spectrum)
            return spectrum_corr

        if self.m_replace_method not in ['smooth','zero']:
            print('WRONG REPLACE_METHOD: NEED TO BE smooth OR zero')
            assert(False)
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        size = self.m_image_in_port.get_shape()[1]
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running OutlierCorrectionModule...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            len_wvl,len_x,len_y = np.shape(datacube)
            x_axis,y_axis=np.arange(len_x),np.arange(len_y)
            coordinates = np.array(np.meshgrid(y_axis,x_axis))
            indices_list = np.transpose(coordinates,axes = (1,2,0)).reshape((-1,2))
            if self.m_cpu > 1:
                results_computation = Parallel(n_jobs=self.m_cpu,verbose=2)(delayed(replace_outliers_MAD)
                                                                                (datacube[:,pxi,pxj],
                                                                                filter_sigma=self.m_filter_sigma,
                                                                                outlier_sigma=self.m_outlier_sigma,
                                                                                replace_method=self.m_replace_method)
                                                                     for pxi,pxj in indices_list)
            else:
                results_computation = [replace_outliers_MAD(datacube[:,pxi,pxj],
                                                            filter_sigma=self.m_filter_sigma,
                                                            outlier_sigma=self.m_outlier_sigma,
                                                            replace_method=self.m_replace_method) for pxi,pxj in indices_list]
            clean_datacube = np.zeros_like(datacube)
            for comp_i,(pxi,pxj) in enumerate(indices_list):
                clean_datacube[:,pxi,pxj] = results_computation[comp_i]
            
            self.m_image_out_port.append(clean_datacube)
            

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history(
            'Outliers removed', 
            'Outlier sigma = %i, filter sigma = %i, replace method = %s' % (self.m_outlier_sigma,self.m_filter_sigma,self.m_replace_method))
        self.m_image_out_port.close_port()

