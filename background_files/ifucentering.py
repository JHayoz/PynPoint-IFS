"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import copy
from scipy.ndimage import generic_filter
from scipy.signal import medfilt

from typing import Union, Tuple

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import shift_image

from .ifu_utils import fit_airy,fit_gaussian,fit_center_custom,select_cubes


class IFUAlignCubesModule(ProcessingModule):
    """
    Module to align the central star within each cube.
    """
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        precision: float = 0.02,
        shift_all_in_tag: str = 'centering_all',
        shift_cube_in_tag: str = 'centering_cubes',
        interpolation: str ='spline',
        name_in: str = 'shift_no_center',
        image_in_tag: str = 'spectrum_NaN_small',
        image_out_tag: str = 'cubes_aligned'
    ) -> None:
        """
        Constructor of IFUAlignCubesModule.
        
        :param precision: 
        :type precision: float
        :param shift_all_in_tag: Tag of the database entry that is read as input with the fit results from the :class:'~pynpoint.processing.centering.FitCenterModule' for each frame.
        :type shift_all_in_tag: str
        :param shift_cube_in_tag: Tag of the database entry that is read as input with the fit results from the :class:'~pynpoint.processing.centering.FitCenterModule' for each median-combined cubes.
        :type shift_cube_in_tag: str
        :param interpolation: Interpolation type for shifting of the images ('spline', 'bilinear', or 'fft').
        :type interpolation: str
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_out_tag: str
        
        :return: None
        """
        
        super(IFUAlignCubesModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_shift_all_in_port = self.add_input_port(shift_all_in_tag)
        self.m_shift_cube_in_port = self.add_input_port(shift_cube_in_tag)
        
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_interpolation = interpolation
        self.m_precision = precision

        #m_image_in_port = image_in_tag
        #m_shift_all_in_port = shift_all_in_tag
        #m_shift_cube_in_port = shift_cube_in_tag

        #m_image_out_port = image_out_tag

        #m_interpolation = interpolation
        #m_precision = precision
    
    
    def run(self):
        """
            Run method of the module. Shifts the images by the difference between each individual and cube shift.
            
            :return: None
            """
        
        
        def image_shift(image, shift_yx, interpolation):
            return shift_image(image, shift_yx, interpolation)
        
        def clean_shifts(shift_arr, precision, k):

            x = range(len(shift_arr))
            #p = np.polyfit(x,shift_arr,2)
            y = medfilt(shift_arr,101)
            
            std = np.std(shift_arr-y)
            max_ = np.max(shift_arr-y)
            mean = np.mean(shift_arr-y)
            shift_temp = copy.copy(shift_arr)
            
            count=0
            while max_>precision and count< self.m_image_in_port.shape[1]:
                filter_shift = medfilt(shift_temp, 101)
                
                for i in range(len(x)):
                    if (shift_temp[i]-y[i] > mean+std) or (shift_temp[i]-y[i] < mean-std):
                        shift_temp[i]=filter_shift[i]
            
                #p = np.polyfit(x,shift_temp,2)
                y = medfilt(shift_temp,101)#np.polyval(p,x)
                std = np.std(shift_temp[3:-3]-y[3:-3])
                max_ = np.max(shift_temp[3:-3]-y[3:-3])
                mean = np.mean(shift_temp[3:-3]-y[3:-3])
                count+=1
            
            return shift_temp
            
        #nframes = self.m_image_in_port.get_attribute("NFRAMES")
        nframes = [1485, 1485, 1485, 1485, 1485, 1485, 1485]
        #size = self.m_image_in_port.get_shape()[1]
        size = 56
            
        shift_all_xy = -1.*self.m_shift_all_in_port[:, [0, 2]]

        shift_cubes_xy = -1.*self.m_shift_cube_in_port[:, [0, 2]]

        iminport=[]
        start_time = time.time()
        for i, nframes_i in enumerate(nframes):
            progress(i, len(nframes), 'Running IFUAlignCubesModule...', start_time)
            shift_xy_i = shift_all_xy[i*nframes_i:(i+1)*nframes_i]-shift_cubes_xy[i] ###
            shift_y_i = clean_shifts(shift_xy_i[:, 1], self.m_precision, i)
            shift_x_i = clean_shifts(shift_xy_i[:, 0],self.m_precision, i)
            for j in range(nframes_i):
                im = image_shift(self.m_image_in_port[i*nframes_i+j],(shift_y_i[j], shift_x_i[j]),self.m_interpolation) ## here it tries to enter an array in second position instead of tuple?
                iminport.append(im.reshape(1,size, size))

        m_image_out_port = iminport
                            
                            
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Align", "cube")
        self.m_image_out_port.close_port()



class FitCenterCustomModule(ProcessingModule):
    """
    Module to center the star within each cube, by averaging the cube along the wavelength direction, and fitting a stellar PSF model (gaussian or airy) to find the mean position of the star in the cube.
    """
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str,
        image_in_tag: str,
        wvl_in_tag: str,
        fit_out_tag: str,
        sign: str = 'positive',
        model: str = 'airy',
        filter_size: float = 3,
        box: int =1
    ) -> None:
        """
        :param name_in: unique name of the module
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input datacube
        :type image_in_tag: str
        :param wvl_in_tag: Tag of the database entry that is read as wavelength axis
        :type wvl_in_tag: str
        :param fit_out_tag: Tag of the database entry that is used as output for the fit parameters of the image center
        :type fit_out_tag: str
        :param sign: Sign of the model (positive or negative)
        :type sign: str
        :param model: Model to use for the fit (gaussian or airy)
        :type model: str
        :param filter_size: standard deviation of the gaussian filter used to smoothe the image
        :type filter_size: float
        :param box: (2box+1,2box+1) are the dimensions of the square, centered at the maximum of the image, used to estimate the parameters of the fit (the amplitude of the model)
        :type box: int
        
        :return: None
        """
        
        super(FitCenterCustomModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wvl_in_port = self.add_input_port(wvl_in_tag)
        
        self.m_fit_out_port  = self.add_output_port(fit_out_tag)
        
        self.m_sign = sign
        self.m_model = model
        self.m_filter_size = filter_size
        self.m_box = box
    
    
    def run(self):
        """
        Run method of the module. Fits the position of the star with respect to the middle of the frame
        
        Returns
        -------
        NoneType
            None
        """
        wvl_len = self.m_wvl_in_port.get_shape()[0]
        # is input an image or a datacube
        if self.m_image_in_port.get_shape()[0] > wvl_len:
            nframes = self.m_image_in_port.get_shape()[0]//wvl_len
            print('Collapsing wavelength')
        else:
            nframes = self.m_image_in_port.get_shape()[0]
            print('Considering this an imaging dataset')
        params = np.zeros((nframes,4))
        iminport=[]
        start_time = time.time()
        for img_i in range(nframes):
            progress(img_i, nframes, 'Running FitCenterCustomModule...', start_time)
            if self.m_image_in_port.get_shape()[0] > wvl_len:
                datacube = self.m_image_in_port[img_i*wvl_len:(img_i + 1)*wvl_len, :, :]
                size_x = np.shape(datacube)[1]
                size_y = np.shape(datacube)[2]
                image_used = np.mean(datacube,axis=0)
            else:
                image_used = self.m_image_in_port[img_i, :, :]
                size_x = np.shape(image_used)[0]
                size_y = np.shape(image_used)[1]
            x0,y0,a0,r0 = fit_center_custom(image=image_used,sigma=self.m_filter_size,box=self.m_box,method=self.m_model,sign=self.m_sign)
            params = np.array([x0-size_x/2,0,y0-size_y/2,0,r0,0,r0,0,a0,0,0,0,0,0],dtype=float) # consistent format with the parameters recorded by FitCenterModule
            for wvl_i in range(wvl_len):
                iminport.append(params)
        print(np.shape(iminport))
        self.m_fit_out_port.set_all(iminport, data_dim=2)
        print(' [DONE]')
        
        self.m_fit_out_port.copy_attributes(self.m_image_in_port)
        self.m_fit_out_port.add_history("Center", "cube")
        self.m_fit_out_port.close_port()