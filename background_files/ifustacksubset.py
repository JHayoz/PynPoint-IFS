"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import copy

from typing import Union, Tuple
from joblib import Parallel, delayed

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import shift_image, rotate
from .ifu_utils import select_cubes

from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry

from background_files.ifuprocessingfunctions import do_PCA_sub,do_derotate_shift

class ApertureCombineModule(ProcessingModule):
    """
    Module to bin together spectral channels.
    """
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'aperture_combine',
        image_in_tag: str = 'raw',
        image_out_tag: str = 'aperture_combined',
        aperture_radius: float = 1.4,
        cpus: int = 1
    ):
        """
        Constructor of ApertureCombineModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param wv_in_tag: Tag of the database entry that is read as input.
        :type wv_in_tag: str
        :param image_root_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_root_out_tag: str
        :param wv_out_tag: Tag of the database entry that is written as output.
        :type wv_out_tag: str
        :param combine: method used to combine the frames
        :type combine: str
        
        :return: None
        """
        
        super(ApertureCombineModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_aperture_radius = aperture_radius
        self.m_cpus = cpus
    
    def run(self):
        """
        Run method of the module. Combines the frames of the cubes.
            
        :return: None
        """
        def calculate_photometry(apertures,image):
            return apertures.do_photometry(image)[0]
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        lencube,lenx,leny = self.m_image_in_port.get_shape()
        X,Y = np.meshgrid(np.arange(leny),np.arange(lenx))
        coordinates = np.dstack([X,Y])
        coordinates_flat = coordinates.reshape((-1,2))
        nb_apertures = len(coordinates_flat)
        apertures = CircularAperture(coordinates_flat,self.m_aperture_radius)
        start_time = time.time()
        if self.m_cpus > 1:
            print('Running ApertureCombineModule in parallel...')
            photometry_apertures = Parallel(n_jobs=self.m_cpus,verbose=2)(delayed(calculate_photometry)(
                apertures=apertures, image=self.m_image_in_port[i:i+1,:,:][0]) for i in range(lencube))
            datacube_photometry = np.array(photometry_apertures).reshape((-1,lenx,leny))
            self.m_image_out_port.set_all(datacube_photometry)
        else:
            for i, nspectrum_i in enumerate(nspectrum):
                
                cube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
                lenwvl = len(cube)
                photometry_apertures = np.zeros((lenwvl,nb_apertures))
                for wvl_i in range(lenwvl):
                    progress(i*lenwvl + wvl_i, len(nspectrum)*lenwvl, 'Running ApertureCombineModule...', start_time)
                    photometry_apertures[wvl_i,:] = apertures.do_photometry(cube[wvl_i])[0]
                datacube_photometry = photometry_apertures.reshape((-1,lenx,leny))
                self.m_image_out_port.append(datacube_photometry)
        
        self.m_image_out_port.add_history("Aperture combined", "radius = %.2f" % self.m_aperture_radius)
        self.m_image_out_port.close_port()


class CrossCorrelationPreparationModule(ProcessingModule):
    """
    Module to eliminate regions of the spectrum.
    """
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'select_range',
        image_in_tag: str = 'initial_spectrum',
        shift_cubes_in_tag: str ='centering_cubes',
        image_out_tag: str = 'im_2D',
        mask_out_tag: str ='mask',
        shift: bool = True,
        rotate: bool = True,
        stack: bool = False,
        combine: str = 'median'
    ) -> None:
        """
        Constructor of CrossCorrelationPreparationModule. Assumes one parallactic angle and one shift per 3D cube XxYxWVL.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param shift_cubes_in_tag: Tag of the database entry that is read as input.
        :type shift_cubes_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_out_tag: str
        :param mask_out_tag: Tag of the database entry that is written as output.
        :type mask_out_tag: str
        :param shift: shift the images to center them.
        :type shift: bool
        :param rotate: rotate the images to a common orientation.
        :type rotate: bool
        :param stack: stack the images to a unique cube.
        :type stack: bool
        
        :return: None
        """
        
        super(CrossCorrelationPreparationModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        if type(shift_cubes_in_tag)==str:
            self.m_shift_in_port = self.add_input_port(shift_cubes_in_tag)
        else:
            self.m_shift_in_port = {}
            for i in range(len(shift_cubes_in_tag)):
                self.m_shift_in_port[i]=self.add_input_port(shift_cubes_in_tag[i])
        
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_mask_out_port = self.add_output_port(mask_out_tag)
        self.m_shift_cubes_in_tag = shift_cubes_in_tag
        
        self.m_shift = shift
        self.m_rotate = rotate
        self.m_stack = stack
        self.m_combine = combine

    def run(self):
        """
        Run method of the module. Convolves the images with a Gaussian kernel.
        
        :return: None
        """
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        pixscale = self.m_image_in_port.get_attribute("PIXSCALE")
        parang = self.m_image_in_port.get_attribute("PARANG")
        size = self.m_image_in_port.get_shape()[-1]
        
        if type(self.m_shift_cubes_in_tag) == str:
            shift_init = self.m_shift_in_port.get_all()
            shift_x = shift_init[:,0]
            shift_y = shift_init[:,2]
        
        else:
            shift_x = np.zeros(len(nspectrum))
            shift_y = np.zeros(len(nspectrum))
            count = 0
            for i in range(len(self.m_shift_cubes_in_tag)):
                shift_init = self.m_shift_in_port[i].get_all()
                shift_x[count:count+len(shift_init)] = shift_init[:,0]
                shift_y[count:count+len(shift_init)] = shift_init[:,2]
                count += len(shift_init)

        shift = np.array([shift_y,shift_x]).T

        final_cube = []
    
        mask_arr = np.zeros((len(nspectrum), size,size))
        mask_arr_shift = np.zeros_like(mask_arr)
        mask_arr_rot = np.zeros_like(mask_arr)
    
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running CrossCorrelationPreparationModule...', start_time)
            
            cube_init = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            cube_rot,mask_arr_rot[i] = do_derotate_shift(
                cube=cube_init,
                shift=self.m_shift,
                shift_vector=shift[i],
                derotate=self.m_rotate,
                derotate_angle=parang[i])
            final_cube.append(cube_rot)
        
        final_cube = np.array(final_cube)

        mask_sum = np.sum(mask_arr_rot, axis=0)
        mask_final = np.where(mask_sum>=len(nspectrum)*0.8, True, False)
        mask_output = np.where(mask_sum>=len(nspectrum)*0.8, 1, 0)

        if self.m_stack==False:
            self.m_image_out_port.set_all(final_cube.reshape(np.shape(final_cube)[0]*np.shape(final_cube)[1], np.shape(final_cube)[3],np.shape(final_cube)[3]))
        else:
            cube_median = []
            for k in range(nspectrum[0]):
                # cube_wv = np.where(np.abs(mask_arr_rot)>0.1, final_cube[:,k,:,:], np.nan)
                cube_wv = final_cube[:,k,:,:]
                if self.m_combine == 'median':
                    combined_cube = np.nanmedian(cube_wv, axis=0)
                    cube_median.append(np.where(mask_final, combined_cube, np.nan))
                    # cube_median.append(combined_cube)
                elif self.m_combine == 'mean':
                    combined_cube = np.nanmean(cube_wv, axis=0)
                    cube_median.append(np.where(mask_final, combined_cube, np.nan))
                    # cube_median.append(combined_cube)
                elif self.m_combine == 'combine':
                    n_samples = np.sum(mask_arr_shift,axis=0)
                    n_samples_corr = np.where(n_samples != 0, n_samples, 1)
                    combined_cube = np.nansum(cube_wv, axis=0)/n_samples_corr
                    cube_median.append(combined_cube)
                else:
                    combined_cube = np.nanmedian(cube_wv, axis=0)
                    cube_median.append(np.where(mask_final, combined_cube, np.nan))
                    # cube_median.append(combined_cube)
                
            cube_median = np.array(cube_median)
            self.m_image_out_port.set_all(cube_median)
        
        self.m_mask_out_port.set_all(mask_output)
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("CC prep", "CC prep")
        if self.m_stack == True:
            self.m_image_out_port.add_attribute("NFRAMES",[nspectrum[0]], False)
        
        self.m_mask_out_port.copy_attributes(self.m_image_in_port)
        self.m_mask_out_port.add_history("CC prep", "CC prep")
        
        
        self.m_image_out_port.close_port()



class BinIFUModule(ProcessingModule):
    """
    Module to bin together spectral channels.
    """
    __author__ = 'Gabriele Cugno'
    
    def __init__(
        self,
        bin,
        name_in = 'bin_channels',
        wv_in_tag = ('wavelength_range',2000),
        image_in_tag = 'cubes_aligned',
        image_root_out_tag = 'bin_',
        wv_out_tag = 'wavelengths',
        combine = 'median'
    ):
        """
        Constructor of BinIFUModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param wv_in_tag: Tag of the database entry that is read as input.
        :type wv_in_tag: str
        :param image_root_out_tag: Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        :type image_root_out_tag: str
        :param wv_out_tag: Tag of the database entry that is written as output.
        :type wv_out_tag: str
        :param combine: method used to combine the frames
        :type combine: str
        
        :return: None
        """
        
        super(BinIFUModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag[0])
        self.m_wv_out_port = self.add_output_port(wv_out_tag)
        
        self.m_outports = []
        for k in range(int(wv_in_tag[1]/bin)):
            self.m_outports.append(self.add_output_port(image_root_out_tag+str(k)))
        
        self.m_nbins = bin
        self.m_combine = combine
    
    def run(self):
        """
        Run method of the module. Combines the frames of the cubes.
            
        :return: None
        """
        
        wv_i =self.m_wv_in_port.get_all()
        nbins_init = len(wv_i)
        num_bin_frames = int(nbins_init/self.m_nbins)
        
        wv_f = []
        for i in range(num_bin_frames):
            wv_f.append((wv_i[i*self.m_nbins]+wv_i[(i+1)*self.m_nbins])/2.)

        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        size = self.m_image_in_port.get_shape()[-1]
    
        start_time = time.time()
        for i, nframes_i in enumerate(nframes):
            progress(i, len(nframes), 'Running BinIFUModule...', start_time)
            cube = self.m_image_in_port[i*nframes_i : (i+1)*nframes_i]
        
            for k in range(num_bin_frames):
                if self.m_combine == 'median':
                    self.m_outports[k].append(np.median(cube[k*self.m_nbins : (k+1)*self.m_nbins], axis=0).reshape(1,size,size))
                if self.m_combine == 'sum':
                    self.m_outports[k].append(np.median(cube[k*self.m_nbins : (k+1)*self.m_nbins], axis=0).reshape(1,size,size))

        sys.stdout.write('Running BinIFUModule... [DONE]\n')
        sys.stdout.flush()
        
        
        for k in range(num_bin_frames):
            self.m_outports[k].copy_attributes(self.m_image_in_port)
            self.m_outports[k].add_attribute("NFRAMES",[len(nframes)], False)
            self.m_outports[k].add_history("Bin spectrum", "nbins = "+str(self.m_nbins))
        
        self.m_wv_out_port.set_all(wv_f)
        self.m_wv_out_port.add_history("Bin wavelength", "nbins = "+str(self.m_nbins))
        self.m_wv_out_port.close_port()

class StackCubesModule(ProcessingModule):
    """
    Module to stack the datacubes along time, either by mean- or median-stacking.
    """
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'stack_cube',
        wv_in_tag: str = 'wavelength_range',
        image_in_tag: str = 'cubes_aligned',
        image_out_tag: str = 'mastercube',
        combine: str = 'median'
    ) -> None:
        """
        Constructor of StackCubesModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param wv_in_tag: Tag of the database entry that is read as wavelength axis.
        :type wv_in_tag: str
        :param image_in_tag: Tag of the database entry that is read as input datacube.
        :type image_in_tag: str
        :param image_out_tag: Tag of the database entry that is written as output.
        :type image_out_tag: str
        :param combine: Method to stack the data, either by 'mean' or 'median' stacking.
        :type combine: str
        
        :return: None
        """
        
        super(StackCubesModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_combine = combine
    
    def run(self):
        """
        Run method of the module. Combines the frames of the cubes.
            
        :return: None
        """
        

        wavelength = self.m_wv_in_port.get_all()
        datacubes = select_cubes(self.m_image_in_port.get_all(),wavelength)

        if self.m_combine=='median':
            master_cube = np.nanmedian(datacubes,axis=(0))
        elif self.m_combine=='mean':
            master_cube = np.nanmean(datacubes,axis=(0))

        self.m_image_out_port.set_all(master_cube)
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_attribute("NFRAMES",[len(wavelength)], False)
        self.m_image_out_port.add_history('Master datacube','Method = %s' % self.m_combine)
        self.m_image_out_port.close_port()