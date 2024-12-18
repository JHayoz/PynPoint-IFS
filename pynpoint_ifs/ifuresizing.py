"""
Pipeline modules for resizing the cubes.
"""

import sys
import time
import math
import warnings
import copy

from typing import Union, Tuple
from typeguard import typechecked

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import shift_image, rotate

class FoldingModule(ProcessingModule):
    """
    Module to fold the cube from 3D science space to 2D detector space. Doesn't keep track of the actual position of the slits.
    """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'im_2D'
    ) -> None:
        """
        Parameters
        ----------
        name_in: str
            Unique name of the module instance.
        image_in_tag: str
            Tag of the database entry that is read as input.
        image_out_tag: str
            Tag of the database entry that is written as output. Should be different from *image_in_tag*.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(FoldingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
    def run(self) -> None:
        """
        Run method of the module. Fold a 3D cube into 2D detector images.
        
        Returns
        -------
        NoneType
            None
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
    Module to construct the cube from 2D detector space to 3D science space. Inverse of FoldingModule.
    """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'im_2D'
    ) -> None:
        """
        Parameters
        ----------
        name_in: str
            Unique name of the module instance.
        image_in_tag: str
            Tag of the database entry that is read as input.
        image_out_tag: str
            Tag of the database entry that is written as output. Should be different from *image_in_tag*.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(UnfoldingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
    
    def run(self):
        """
        Run method of the module. Unfold 2D detector images into 3D cubes.
        
        Returns
        -------
        NoneType
            None
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

class UpSampleModule(ProcessingModule):
    """
    Module to save the data to a higher (spatial) sampling rate by "dividing" the pixels, e.g. from (1) into (1 1, 1 1) if factor=2. Note: this does not conserve flux, namely the flux increases by factor^2 as pixels are copied.
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'upsample',
        image_in_tag: str= 'datacube',
        image_out_tag: str = 'datacube_upsampled',
        factor: int = 2
    ) -> None:
        """
        Parameters
        ----------
        name_in: str
            Unique name of the module instance.
        image_in_tag: str
            Tag of the database entry that is read as input.
        image_out_tag: str
            Tag of the database entry that is written as output. Should be different from *image_in_tag*.
        factor: int
            factor by which the sampling should be increased.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(UpSampleModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_factor = factor
    
    def run(self):
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        
        nframes,lenx,leny = self.m_image_in_port.get_shape()
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running UpSampleModule...', start_time)
            
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            datacube_resampled = np.zeros((nspectrum_i,self.m_factor*lenx,self.m_factor*leny))
            
            for factor_i in range(self.m_factor):
                for factor_j in range(self.m_factor):
                    datacube_resampled[:,factor_i::self.m_factor,factor_j::self.m_factor] = datacube[:,:,:]
            self.m_image_out_port.append(datacube_resampled)
            

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('Up sampled', 'Factor: %i' % self.m_factor)
        self.m_image_out_port.close_port()

class FinerGridInterpolationModule(ProcessingModule):
    """
    Module to interpolate the data on a finer grid, defined by the argument factor, which controls the increase in spatial sampling. Note: this does not conserve flux, namely the flux increases by factor^2 as nex pixels are created through interpolation.
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'upsample',
        image_in_tag: str = 'datacube',
        image_out_tag: str = 'datacube_upsampled',
        factor: int = 2
    ) -> None:
        """
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be different from *image_in_tag*.
        factor : int
            factor by which the sampling should be increased
        
        Returns
        -------
        NoneType
            None
        """
        
        super(FinerGridInterpolationModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_factor = factor
    
    def run(self):
        """
        Run method of the module. Unfold 2D detector images in 3D cubes.
        
        Returns
        -------
        NoneType
            None
        """
        
        nframes,lenx,leny = self.m_image_in_port.get_shape()
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running FinerGridInterpolationModule...', start_time)
            
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            lenwvl,lenx,leny=np.shape(datacube)
            x_points,y_points = np.arange(0,self.m_factor*lenx,self.m_factor),np.arange(0,self.m_factor*leny,self.m_factor)
            x_points_new,y_points_new = np.arange(self.m_factor*lenx),np.arange(self.m_factor*leny)
            X,Y = np.meshgrid(x_points_new,y_points_new,indexing='ij')
            
            datacube_resampled = np.zeros((nspectrum_i,self.m_factor*lenx,self.m_factor*leny))

            
            for wvl_i in range(lenwvl):
                grid_interpolator = RegularGridInterpolator(points=(x_points,y_points), values=datacube[wvl_i,:,:], method='linear', bounds_error=False,fill_value=None)
                datacube_resampled[wvl_i,:,:] = grid_interpolator((X,Y))
            self.m_image_out_port.append(datacube_resampled)
            

        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('Up sampled', 'Factor: %i' % self.m_factor)
        self.m_image_out_port.close_port()