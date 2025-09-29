"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings

from typing import Union, Tuple
from typeguard import typechecked

import numpy as np
from PyAstronomy.pyasl import dopplerShift

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

from pynpoint_ifs.ifu_utils import get_wavelength

class SelectWavelengthRangeModule(ProcessingModule):
    """
    Module to select spectral channels based on the wavelength.
    """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        range_f: Tuple[float, float],
        range_i: Tuple[float, float] = (1.92854,2.47171),
        name_in: str = "Select_range",
        image_in_tag: str = "initial_spectrum",
        image_out_tag: str = "spectrum_selected",
        wv_out_tag: str = "wavelengths"
    ) -> None:
        """
        Parameters
        ----------
        range_f : tuple(float,float)
            final wavelegth range for the selected frames
        range_i : tuple(float,float)
            initial wavelegth range for the cube
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        image_out_tag : str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        wv_out_tag : str
            Tag of the database entry for the wavelength that is written as output. Should be different from *image_in_tag*.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(SelectWavelengthRangeModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_wv_out_port = self.add_output_port(wv_out_tag)
        
        self.m_range_f = range_f
        self.m_range_i = range_i
    
    
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        
        
        nframes = self.m_image_in_port.get_attribute("NFRAMES")
        
        spectrum_arr = np.linspace(self.m_range_i[0], self.m_range_i[-1], nframes[0])
        n_before = 0
        while spectrum_arr[n_before]<self.m_range_f[0]:
            n_before+=1
        
        n_after = nframes[0]-1
        while spectrum_arr[n_after]>self.m_range_f[1]:
            n_after-=1
        
        start_time = time.time()
        for i, nframes_i in enumerate(nframes):
            progress(i, len(nframes), 'SelectWavelengthRangeModule...', start_time)
            frames_i = self.m_image_in_port[i*nframes_i+n_before:i*nframes_i+n_after,:,:]
            self.m_image_out_port.append(frames_i)

        
        nframes_final = np.ones(len(nframes), dtype=int)*len(spectrum_arr[n_before: n_after])
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_attribute("NFRAMES",nframes_final, False)
        self.m_image_out_port.add_history("Select Spectrum", "lambda range = "+str(self.m_range_f))
        
        self.m_wv_out_port.set_all(spectrum_arr[n_before: n_after])
        self.m_wv_out_port.add_history("Select Spectrum", "lambda range = "+str(self.m_range_f))
        self.m_wv_out_port.close_port()


class AutomaticallySelectWavelengthRangeModule(ProcessingModule):
    """
    Module to automatically select spectral channels based on the bounds of the spectra in the data (i.e. where the spectra end on the detector frames)
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = "Select_range",
        image_in_tag: str = "raw",
        image_out_tag: str = "raw_sel",
        wv_out_tag: str = "wavelength",
        header_crval: str = 'CRVAL3',
        header_cd: str = 'CD3_3'
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
        wv_out_tag : str
            Tag of the database entry for the wavelength that is written as output. Should be different from *image_in_tag*.
        header_crval: str 
            header keyword for the reference wavelength of the first wavelength bin
        header_cd : str
            header keyword for the bin width of the wavelength axis
        
        Returns
        -------
        NoneType
            None
        """
        
        super(AutomaticallySelectWavelengthRangeModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_wv_out_port = self.add_output_port(wv_out_tag)

        self.m_header_crval = header_crval
        self.m_header_cd = header_cd
    
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        
        
        files = np.array(self.m_image_in_port.get_attribute('FILES'),dtype=str)
        wlen_bounds = np.zeros((len(files),2))
        nframes = []
        for cube_i in range(len(files)):
            wavelength = get_wavelength(files[cube_i],header_crval=self.m_header_crval,header_cd=self.m_header_cd)
            lencube = len(wavelength)
            index_f0 = int(np.sum(nframes[:cube_i]))
            nframes += [lencube]
            data = self.m_image_in_port[index_f0:index_f0 + lencube,:,:]
            assert(len(data)==lencube)
            # identify bad regions
            non_nans = ~np.logical_or((data == 0.),np.isnan(data))
            # count them along wavelength
            non_nans_sum = np.sum(non_nans,axis=(1,2))
            # most of the detector is good, so the median number of non-nans is close to the number of good pixels
            # at the edges of the spectral axis, the number of good pixels increase quickly: 
            # so a difference of 100 to the median number of non-nans is a good estimate of the edge
            mask_good = non_nans_sum > np.median(non_nans_sum) - 100
            lower_bound = np.where(mask_good)[0][0] + 10
            higher_bound = np.where(mask_good)[0][-1] - 10
            wlen_bounds[cube_i,0] = wavelength[lower_bound]
            wlen_bounds[cube_i,1] = wavelength[higher_bound]
        nframes = np.array(nframes)
        remove_low = np.max(wlen_bounds[:,0])//0.00013*0.00013 # sometimes led to float errors if not making sure that this is a multiple of the wvl spacing
        remove_high_tmp = np.min(wlen_bounds[:,1])
        remove_high = remove_low + ((remove_high_tmp-remove_low)//0.00013)*0.00013
        print('Keeping data between wavelength %.3f and %.3f' % (remove_low,remove_high))
        new_nframes = []
        for cube_i,lencube in enumerate(nframes):
            index_f0 = int(np.sum(nframes[:cube_i]))
            wavelength = get_wavelength(files[cube_i],header_crval=self.m_header_crval,header_cd=self.m_header_cd)
            data = self.m_image_in_port[index_f0:index_f0 + lencube,:,:]
            assert(len(data)==len(wavelength))
            mask_keep = np.logical_and(wavelength >= remove_low,wavelength <= remove_high)
            new_lencube = np.sum(mask_keep)
            
            # Check that the length of the cubes are all the same
            if len(new_nframes) > 0:
                if new_lencube != new_nframes[-1]:
                    raise RuntimeError(
                        f'Cube %i is not the same size as the previous cube.' % cube_i
                    )
            new_nframes += [new_lencube]
            
            data_sel = data[mask_keep]
            
            self.m_image_out_port.append(data_sel)
            
        mask_keep = np.logical_and(wavelength >= remove_low,wavelength <= remove_high)
        new_wavelength_axis = wavelength[mask_keep]
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_attribute('NFRAMES',np.array(new_nframes), False)
        self.m_image_out_port.add_history("Select Spectrum", "lambda range = (%.5f,%.5f)" % (remove_low,remove_high))
        
        self.m_wv_out_port.set_all(new_wavelength_axis)
        self.m_wv_out_port.add_history("Select Spectrum", "lambda range = (%.5f,%.5f)" % (remove_low,remove_high))
        self.m_wv_out_port.close_port()


class CorrectWavelengthModule(ProcessingModule):
    """
    Module to dopplershift the wavelength axis
    """
    
    __author__ = 'Gabriele Cugno'
    
    @typechecked
    def __init__(
        self,
        name_in: str = "Correct_wl",
        wv_in_tag: str = "initial_spectrum",
        wv_out_tag: str = "spectrum_selected",
        shift_km_s: float = 0.
    ) -> None:
        """
        Parameters
        ----------
        name_in: str
            Unique name of the module instance.
        wv_in_tag: str
            Tag of the database entry that is read as input.
        wv_out_tag: str
            Tag of the database entry for the wavelength that is written as output. Should be different from *image_in_tag*.
        shift_km_s: float
            Radial velocity by which the wavelength axis should be Doppler-shifted.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(CorrectWavelengthModule, self).__init__(name_in)
        
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_wv_out_port = self.add_output_port(wv_out_tag)
    
        self.m_shift_km_s = shift_km_s
    
    
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        
        wv = self.m_wv_in_port.get_all()
        
        _, wl_shift = dopplerShift(wv, np.ones(len(wv)), self.m_shift_km_s)
        
        
        self.m_wv_out_port.set_all(wl_shift)
        self.m_wv_out_port.copy_attributes(self.m_wv_in_port)
        self.m_wv_out_port.add_history("Select Shifted", "RV = "+str(self.m_shift_km_s))
        self.m_wv_out_port.close_port()


