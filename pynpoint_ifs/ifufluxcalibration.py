"""
Pipeline modules for flux extraction and calibration.
"""
import time
import warnings

from typeguard import typechecked
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter,median_filter
from scipy.signal import savgol_filter,medfilt,resample
from scipy.interpolate import splrep, BSpline,interp1d
from scipy.stats import median_abs_deviation
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import AiryDisk2D,Gaussian2D
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry
from photutils.centroids import centroid_quadratic
import copy
from spectres import spectres
import skycalc_ipy
import pandas as pd

import pynpoint as pp
from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

from pynpoint_ifs.ifu_utils import rebin


class IFUSpectrumExtractionModule(ProcessingModule):
    """
    Module to extract the spectrum of an object using aperture photometry. Optionally the background can be extracted too.
    """
    
    __author__ = 'Jean Hayoz'

    @typechecked
    def __init__(self,
                 name_in: str = 'extract_spectrum',
                 image_in_tag: str = 'raw',
                 obj_position_in_tag: str = 'fit_param',
                 obj_spectra_out_tag: str = 'spectra_raw',
                 bk_spectra_out_tag: str = 'spectra_bk',
                 aperture_obj_radius: float = 4,
                 aperture_bk_radius: float = 4,
                 aperture_bk_nb: int = 4,
                 aperture_bk_dist: float = 8.,
                 aperture_bk_angle: float = 0.,
                 aperture_bk_position: str = 'around',
                 aperture_bk_combine: bool = True,
                 plot: bool = True
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
        outlier_sigma : float
            number of sigmas used to tag outliers. The "sigma" is calculated by the median absolute deviation.
        filter_sigma : float
            standard deviation of the Gaussian filter used to smoothe the spectrum.
        replace_method : str
            if 'smooth', then replaces outliers by the smoothed spectrum, else replaces outliers by 0
        cpu : int
            if cpu > 1, then parallelise the operation using joblib
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUSpectrumExtractionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_obj_position_in_port = self.add_input_port(obj_position_in_tag)
        self.m_obj_spectra_out_port = self.add_output_port(obj_spectra_out_tag)
        
        self.m_ap_obj_radius = aperture_obj_radius
        self.m_ap_bk_radius = aperture_bk_radius
        self.m_ap_bk_nb = aperture_bk_nb
        self.m_ap_bk_dist = aperture_bk_dist
        self.m_ap_bk_angle = aperture_bk_angle
        self.m_ap_position = aperture_bk_position
        self.m_ap_bk_combine = aperture_bk_combine
        self.m_plot = plot

        self.m_extract_bk = True
        if self.m_ap_bk_nb <= 0:
            self.m_extract_bk = False
        else:
            self.m_bk_spectra_out_port = self.add_output_port(bk_spectra_out_tag)
        
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        lenimages,lenx,leny = self.m_image_in_port.get_shape()
        coord_origin_shift = 0.5*(1-(lenx%2))
        lenwvl = np.max(nspectrum)
        lencube = lenimages//lenwvl
        fitparam_obj = self.m_obj_position_in_port.get_all()[:,(0,2)]
        if len(fitparam_obj) == lenimages:
            position_frames = True
        else:
            position_frames = False

        # create the background apertures
        if self.m_extract_bk:
            bg_apertures = np.array([
                self.m_ap_bk_dist*np.array([
                    np.cos(2*np.pi/self.m_ap_bk_nb*i+self.m_ap_bk_angle),
                    np.sin(2*np.pi/self.m_ap_bk_nb*i+self.m_ap_bk_angle)
                ]) for i in range(self.m_ap_bk_nb)])
        
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUSpectrumExtractionModule...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            obj_spectrum = np.zeros((lenwvl,))
            if self.m_extract_bk:
                bk_spectra = np.zeros((lenwvl,self.m_ap_bk_nb))
            
            for wvl_i in range(lenwvl):
                if position_frames:
                    # different position for every frame within a cube
                    obj_pos = fitparam_obj[i*nspectrum_i + wvl_i,:] + np.array([leny-1,lenx-1])/2 # - coord_origin_shift
                else:
                    # same position for every frame within a cube
                    obj_pos = fitparam_obj[i,:] + np.array([leny-1,lenx-1])/2 # - coord_origin_shift
                # background
                if self.m_extract_bk:
                    if self.m_ap_position=='around':
                        pos_bg_apertures = obj_pos + bg_apertures[:,:]
                    else:
                        pos_bg_apertures = np.zeros_like(obj_pos) + bg_apertures[:,:] + np.array([leny,lenx])/2
                    aperture_bk = CircularAperture(pos_bg_apertures,self.m_ap_bk_radius)
                    bk_spectra[wvl_i,:] = aperture_bk.do_photometry(datacube[wvl_i])[0]
                
                # object
                aperture_obj = CircularAperture(obj_pos,self.m_ap_obj_radius)
                obj_spectrum[wvl_i] = aperture_obj.do_photometry(datacube[wvl_i])[0]
                
                
            if self.m_plot:
                plt.figure()
                img = np.mean(datacube,axis=0)
                plt.imshow(img,origin='lower',vmin=np.nanpercentile(img,5),vmax=np.nanpercentile(img,95),cmap='afmhot')
                aperture_obj.plot(color='b')
                if self.m_extract_bk:
                    aperture_bk.plot(color='b')
                plt.show()
            
            self.m_obj_spectra_out_port.append([obj_spectrum])
            if self.m_extract_bk:
                mask_nans = np.isnan(np.sum(bk_spectra,axis=0))
                if self.m_ap_bk_combine:
                    bk_spectra_median = np.median(bk_spectra.transpose()[~mask_nans].transpose(),axis=1)/(self.m_ap_bk_radius**2)*(self.m_ap_obj_radius**2)
                    self.m_bk_spectra_out_port.append([bk_spectra_median])
                else:
                    self.m_bk_spectra_out_port.append([bk_spectra.transpose()])
            
        self.m_obj_spectra_out_port.copy_attributes(self.m_image_in_port)
        self.m_obj_spectra_out_port.add_history(
            'Spectrum extracted', 
            'Spectrum extraction parameters: obj R=%.1f, bk (%s) R=%.1f N=%i d=%.1f dphi=%.1f %s' % (self.m_ap_obj_radius,self.m_extract_bk,self.m_ap_bk_radius,self.m_ap_bk_nb,self.m_ap_bk_dist,self.m_ap_bk_angle,self.m_ap_position))
        
        self.m_obj_spectra_out_port.close_port()
        if self.m_extract_bk:
            self.m_bk_spectra_out_port.close_port()