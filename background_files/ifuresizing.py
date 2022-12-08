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
from pynpoint.util.image import shift_image, rotate




class FoldingModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 image_out_tag = "im_2D"):
        """
            Constructor of FoldingModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(FoldingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
    def run(self):
        """
            Run method of the module. Fold a 3D cube into 2D detector images.
            
            :return: None
            """

        
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        size = self.m_image_in_port.get_shape()[1]
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running FoldingModule...', start_time)
            
            frames_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]

            
            im_2d = np.zeros((nspectrum_i, int(size*size)))
            for j in range(nspectrum_i):
                for k in range(int(size)):
                    im_2d[nspectrum_i-j-1,k*size:(k+1)*size]=frames_i[j,k,:]
            
            self.m_image_out_port.append(im_2d.reshape(1,nspectrum_i,int(size*size)))
        
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Folded", "2D image ")
        self.m_image_out_port.close_port()



class UnfoldingModule(ProcessingModule):
    """
        Module to eliminate regions of the spectrum.
        """
    
    def __init__(self,
                 name_in = "Select_range",
                 image_in_tag = "initial_spectrum",
                 image_out_tag = "im_2D"):
        """
            Constructor of UnfoldingModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            
            :return: None
            """
        
        super(UnfoldingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
    def run(self):
        """
            Run method of the module. Unfold 2D detector images in 3D cubes.
            
            :return: None
            """
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        size = int(np.sqrt(self.m_image_in_port.get_shape()[2]))
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running UnfoldingModule...', start_time)
            
            frames_i = np.zeros((nspectrum_i,size, size))
            im_2D = self.m_image_in_port[i,:,:]
            
            for j in range(nspectrum_i):
                for k in range(size):
                    frames_i[j,k,:] = im_2D[nspectrum_i-j-1,k*size:(k+1)*size]
            
            self.m_image_out_port.append(frames_i)

        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("Unfolded", "3D image ")
        self.m_image_out_port.close_port()

