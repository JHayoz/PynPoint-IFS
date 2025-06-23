"""
Pipeline modules for frame alignment.
"""

import sys
import time
import math
import warnings
import copy

from typing import Union, Tuple
from typeguard import typechecked

from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
from photutils.centroids import centroid_1dg,centroid_2dg,centroid_com,centroid_quadratic,centroid_sources

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import shift_image

from pynpoint_ifs.ifu_utils import fit_airy,fit_gaussian,fit_center_custom,select_cubes


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
        Parameters
        ----------
        
        precision: float
            Precision with which to align the images via cross-correlation.
        shift_all_in_tag : str
            Tag of the database entry that is read as input with the fit results from the :class:'~pynpoint.processing.centering.FitCenterModule' for each frame.
        shift_cube_in_tag : str
            Tag of the database entry that is read as input with the fit results from the :class:'~pynpoint.processing.centering.FitCenterModule' for each median-combined cubes.
        interpolation : str
            Interpolation type for shifting of the images ('spline', 'bilinear', or 'fft').
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        
        Returns
        -------
        NoneType
            None
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
    
    
    def run(self) -> None:
        """
        Run method of the module. Shifts the images by the difference between each individual and cube shift.
        
        Returns
        -------
        NoneType
            None
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
        Parameters
        ----------
        
        name_in : str
            unique name of the module
        image_in_tag : str
            Tag of the database entry that is read as input datacube
        wvl_in_tag : str
            Tag of the database entry that is read as wavelength axis
        fit_out_tag : str
            Tag of the database entry that is used as output for the fit parameters of the image center
        sign : str
            Sign of the model (positive or negative)
        model : str
            Model to use for the fit (gaussian or airy)
        filter_size : float
            standard deviation of the gaussian filter used to smoothe the image
        box : int
            (2box+1,2box+1) are the dimensions of the square, centered at the maximum of the image, used to estimate the parameters of the fit (the amplitude of the model)
        
        Returns
        -------
        NoneType
            None
        """
        
        super(FitCenterCustomModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wvl_in_port = self.add_input_port(wvl_in_tag)
        
        self.m_fit_out_port  = self.add_output_port(fit_out_tag)
        
        self.m_sign = sign
        self.m_model = model
        self.m_filter_size = filter_size
        self.m_box = box
    
    
    def run(self) -> None:
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

class FitCentroidsModule(ProcessingModule):
    """
    Module to fit the position of several sources in the field of view using centroids.
    """
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str,
        image_in_tag: str,
        wvl_in_tag: str,
        fit_out_tag: str,
        collapse_wavelength: bool = True,
        centroid_model: str = 'centroid_2dg',
        initial_guess: np.ndarray = None,
        filter_sigma: float = 3,
        box: int = 10,
        crop: int = 0,
        plot: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        
        name_in : str
            unique name of the module
        image_in_tag : str
            Tag of the database entry that is read as input datacube
        wvl_in_tag : str
            Tag of the database entry that is read as wavelength axis
        fit_out_tag : str
            Tag of the database entry that is used as output for the fit parameters of the image center
        centroid_model : str
            Model to use for the fit (centroid_quadratic, centroid_2dg, centroid_1dg, centroid_com)
        initial_guess : np.ndarray
            Initial guess for the position of the sources in the image. The format should be [[x1,y1],[x2,y2],...,[xn,yn]] for n objects to fit. If None, then just fit one source.
        filter_size : float
            standard deviation of the gaussian filter used to smoothe the image
        box : int
            (2box+1,2box+1) are the dimensions of the square, centered at the maximum of the image, used to estimate the parameters of the fit (the amplitude of the model)
        
        Returns
        -------
        NoneType
            None
        """
        
        super(FitCentroidsModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wvl_in_port = self.add_input_port(wvl_in_tag)

        self.m_collapse_wavelength = collapse_wavelength
        self.m_initial_guess = initial_guess
        self.m_filter_sigma = filter_sigma
        self.m_box = box
        self.m_crop = crop
        self.m_plot = plot
        
        # determine number of sources to fit
        if initial_guess is None:
            self.m_number_sources = 1
            self.m_fit_out_port  = self.add_output_port(fit_out_tag)
        else:
            self.m_number_sources = len(self.m_initial_guess)
            self.m_fit_out_port = {}
            for source_i in range(self.m_number_sources):
                self.m_fit_out_port[source_i]  = self.add_output_port(fit_out_tag + ('_source_%i' % source_i))
        self.m_fit_out_port_relative = self.add_output_port(fit_out_tag + '_relative')
        # get the right centroid function
        self.m_centroid_model = centroid_model
        if centroid_model == 'centroid_1dg':
            self.m_centroid_model = centroid_1dg
        elif centroid_model == 'centroid_2dg':
            self.m_centroid_model = centroid_2dg
        elif centroid_model == 'centroid_com':
            self.m_centroid_model = centroid_com
        elif centroid_model == 'centroid_quadratic':
            self.m_centroid_model = centroid_quadratic
        else:
            raise RuntimeError(
                    f"The centroid_model argument must be one of centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic."
                )
        
    def run(self) -> None:
        """
        Run method of the module. Fits the position of the star with respect to the middle of the frame
        
        Returns
        -------
        NoneType
            None
        """
        lenwvl = self.m_wvl_in_port.get_shape()[0]
        lendatacube,lenx,leny = self.m_image_in_port.get_shape()

        # fit individual images or just the wavelength-averaged cubes
        if self.m_collapse_wavelength:
            nframes = self.m_image_in_port.get_shape()[0]//lenwvl
            message_history = 'Collapsed wavelength'
            print('Collapsing wavelength')
        else:
            nframes = self.m_image_in_port.get_shape()[0]
            message_history = 'Fit each image separately'
            print('Considering this an imaging dataset')
        
        params = np.zeros((nframes,self.m_number_sources,14))
        start_time = time.time()
        for img_i in range(nframes):
            progress(img_i, nframes, 'Running FitCentroidsModule...', start_time)
            if self.m_collapse_wavelength:
                image = np.mean(self.m_image_in_port[img_i*lenwvl:(img_i + 1)*lenwvl, :, :],axis=0)
            else:
                image = self.m_image_in_port[img_i, :, :]
            
            image_crop = np.zeros_like(image)
            if self.m_crop > 0:
                image_crop[self.m_crop:-self.m_crop,self.m_crop:-self.m_crop] = image[self.m_crop:-self.m_crop,self.m_crop:-self.m_crop]
            else:
                image_crop[:,:] = image[:,:]
            
            image_smooth = gaussian_filter(image_crop,self.m_filter_sigma)
            
            if self.m_number_sources == 1:
                x1, y1 = self.m_centroid_model(image_smooth)
                params[img_i,0,0] = x1-lenx/2
                params[img_i,0,2] = y1-leny/2
            else:
                xs, ys = centroid_sources(image_smooth, xpos=self.m_initial_guess[:,0], ypos=self.m_initial_guess[:,1], box_size=self.m_box,
                        centroid_func=self.m_centroid_model)
                for source_i in range(self.m_number_sources):
                    params[img_i,source_i,0] = xs[source_i]-lenx/2
                    params[img_i,source_i,2] = ys[source_i]-leny/2
            
            if self.m_plot:
                plt.figure()
                plt.imshow(image_smooth,vmin=np.nanpercentile(image_smooth,5),vmax=np.nanpercentile(image_smooth,95),origin='lower')
                for source_i in range(self.m_number_sources):
                    plt.plot(params[img_i,source_i,0]+lenx/2,params[img_i,source_i,2]+leny/2,marker='.',markersize=5,label='Source %i' % source_i)
                plt.legend()
                plt.title('Image %i' % img_i)
                plt.show()
            
        if self.m_number_sources == 1:
            self.m_fit_out_port.set_all(params[:,0,:], data_dim=2)
            self.m_fit_out_port.copy_attributes(self.m_image_in_port)
            self.m_fit_out_port.add_history("Fit Center", message_history)
            self.m_fit_out_port.close_port()
        else:
            for source_i in range(self.m_number_sources):
                self.m_fit_out_port[source_i].set_all(params[:,source_i,:], data_dim=2)
                self.m_fit_out_port[source_i].copy_attributes(self.m_image_in_port)
                self.m_fit_out_port[source_i].add_history("Fit Center", message_history)
                self.m_fit_out_port[source_i].close_port()
        # Add relative position for centering purpose
        mean_pos = np.mean(params[:,0,:],axis=0)
        self.m_fit_out_port_relative.set_all(params[:,0,:] - mean_pos, data_dim=2)
        self.m_fit_out_port_relative.copy_attributes(self.m_image_in_port)
        self.m_fit_out_port_relative.add_history("Fit Center", message_history)
        self.m_fit_out_port_relative.close_port()
        print(' [DONE]')