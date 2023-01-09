"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import copy
from scipy.ndimage.filters import generic_filter
from scipy.signal import medfilt

from typing import Union, Tuple

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import shift_image

class IFUAlignCubesModule(ProcessingModule):
    """
        Module to align the central star within each cube.
        """
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(self,
                 precision: float = 0.02,
                 shift_all_in_tag: str = "centering_all",
                 shift_cube_in_tag: str = "centering_cubes",
                 interpolation: str ="spline",
                 name_in: str ="shift_no_center",
                 image_in_tag: str ="spectrum_NaN_small",
                 image_out_tag: str ="cubes_aligned"):
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
    
    
    def run(self):
        """
            Run method of the module. Shifts the images by the difference between each individual and cube shift.
            
            :return: None
            """
        
        
        def _image_shift(image, shift_yx, interpolation):
            return shift_image(image, shift_yx, interpolation)
        
        def _clean_shifts(shift_arr, precision, k):
            x = range(len(shift_arr))
            #p = np.polyfit(x,shift_arr,2)
            y = medfilt(shift_arr,101)
            
            std = np.std(shift_arr-y)
            max_ = np.max(shift_arr-y)
            mean = np.mean(shift_arr-y)
            shift_temp = copy.copy(shift_arr)
            
            count=0
            while max_>precision and count<self.m_image_in_port.get_shape()[1]:
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
            
        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        size = self.m_image_in_port.get_shape()[1]
        print(self.m_image_in_port.get_shape())
            
        shift_all_xy = -1.*self.m_shift_all_in_port[:, [0, 2]]
        shift_cubes_xy = -1.*self.m_shift_cube_in_port[:, [0, 2]]
        
        start_time = time.time()
        for i, nframes_i in enumerate(nframes):
            progress(i, len(nframes), 'Running IFUAlignCubesModule...', start_time)
            shift_xy_i = shift_all_xy[i*nframes_i:(i+1)*nframes_i]-shift_cubes_xy[i]
            shift_y_i = _clean_shifts(shift_xy_i[:, 1], self.m_precision, i)
            shift_x_i = _clean_shifts(shift_xy_i[:, 0],self.m_precision, i)
            for j in range(nframes_i):
                #im = _image_shift(self.m_image_in_port[i*nframes_i+j],(shift_y_i[j], shift_x_i[j]),self.m_interpolation) # temp modif to remove error.
                im = _image_shift(self.m_image_in_port[i*nframes_i+j,:,:],(shift_y_i[j], shift_x_i[j]),self.m_interpolation)
                self.m_image_out_port.append(im.reshape(1,size, size))

                            
                            
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Align", "cube")
        self.m_image_out_port.close_port()

