"""
Pipeline modules for the detection and interpolation of bad pixels.
"""

import copy
import warnings

from typing import Union, Tuple

import cv2
import numpy as np

from numba import jit
from typeguard import typechecked
from scipy.ndimage.filters import generic_filter

from pynpoint.core.processing import ProcessingModule


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
                 local: bool = False):
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
    
    
    def run(self):
        """
            Run method of the module. Look for NaNs and substitute them.
            
            :return: None
            """

        print("hey")

        def clean_NaNs(image,img):

            size0=np.shape(image)[0]
            size1=np.shape(image)[1]
            if self.m_local:
                im_new = generic_filter(image, np.nanmedian, size=3)
            else:
                im_new = np.nanmedian(image)
            for i in range(size0):
                for j in range(size1):
                    if not np.isfinite(image[i,j]):
                        if self.m_local:
                            image[i,j]= im_new[i,j]
                        else:
                            image[i,j]= im_new


            return image


        self.apply_function_to_images(clean_NaNs,
                                      self.m_image_in_port,
                                      self.m_image_out_port,
                                      "Running NanFilterModule...")



        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("NaN removed", "sub with mean ")
        self.m_image_out_port.close_port()
