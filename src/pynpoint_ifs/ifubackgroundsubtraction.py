"""
Pipeline modules for background subtraction.
"""

import time

from typeguard import typechecked
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import splrep,BSpline
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

from pynpoint_ifs.ifu_utils_hidden import create_planet_mask
from pynpoint_ifs.ifu_plotting import plot_circle

class IFUBackgroundSubtractionModule(ProcessingModule):
    """
    Module to model the background and subtract it. Each slit is fitted individually, the spectral dependence is smooth, and the flux level across a slit is a spline.
    """
    
    __author__ = 'Jean Hayoz'

    @typechecked
    def __init__(
        self,
        name_in: str = 'subtract_background',
        image_in_tag: str = 'raw',
        image_out_tag: str = 'raw_bksub',
        background_out_tag: str = 'bk',
        planet_shift_param_tag: str = 'planet_position',
        star_shift_param_tag: str = 'star_position',
        mask_size_planet: float = 6.,
        mask_size_star: float = 12.,
        filter_sigma: float = 40.,
        background_method: str = 'spline',
        plot: bool = False
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
        
        super(IFUBackgroundSubtractionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_background_out_port = self.add_output_port(background_out_tag)
        self.m_planet_shift_in_port = self.add_input_port(planet_shift_param_tag)
        if mask_size_star > 0:
            self.m_star_shift_in_port = self.add_input_port(star_shift_param_tag)
            self.star_mask = True
        else:
            self.star_mask = False
        

        self.mask_size_planet = mask_size_planet
        self.mask_size_star = mask_size_star
        self.filter_sigma = filter_sigma
        self.background_method = background_method
        self.m_plot = plot
    
    
    def run(self) -> None:
        """
        Run method of the module. Model the background and subtract it.
        
        Returns
        -------
        NoneType
            None
        """
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        lenimages,lenx,leny = self.m_image_in_port.get_shape()
        lenwvl = np.max(nspectrum)
        lencube = lenimages//lenwvl
        fitparam_planet = self.m_planet_shift_in_port.get_all()
        # check the size
        planet_pos = fitparam_planet[:,(0,2)]
        if self.star_mask:
            fitparam_star = self.m_star_shift_in_port.get_all()
            star_pos = fitparam_star[:,(0,2)]
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUBackgroundSubtractionModule...', start_time)
            datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            planet_mask = create_planet_mask(im_shape=np.shape(datacube),size=(self.mask_size_planet,planet_pos[i,1],planet_pos[i,0])).astype(bool)
            if self.star_mask:
                # adding star mask
                star_mask = create_planet_mask(im_shape=np.shape(datacube),size=(self.mask_size_star,star_pos[i,1],star_pos[i,0])).astype(bool)
                planet_mask = np.logical_and(planet_mask,star_mask)
                
            background = np.zeros_like(datacube)
            # background_spectrum_smooth = np.zeros((lenwvl,lenx))
            # background_slit_model = np.zeros((leny,lenx))
            for slit_i in range(lenx):
                slitlet = datacube[:,slit_i,:]
                mask_slit = planet_mask[:,slit_i,:]
                all_spectra = np.array([slitlet[:,col_i] for col_i in range(leny) if mask_slit[0,col_i] == 1])
                # get non-masked data
                background_spectrum = np.median(all_spectra,axis=0)
                # smooth out background over spectral dimension
                background_spectrum_smooth_tmp = gaussian_filter(background_spectrum,sigma=self.filter_sigma)
                # median background value
                background_spectrum_zero = np.median(background_spectrum_smooth_tmp)
                # get spatial variation
                background_slit = np.median(all_spectra,axis=1)
                # get column index of non-masked data
                column_indices_all = np.arange(leny)
                column_indices = column_indices_all[mask_slit[0,:]]
                # fit a spline to the spatial variation of the background
                background_slit_spl = splrep(x = column_indices, y = background_slit, s=2)
                background_slit_model_tmp = BSpline(*background_slit_spl)(column_indices_all)
                if self.background_method == 'spline':
                    background_slit_zero = np.median(background_slit_model_tmp)
                elif self.background_method == 'median':
                    background_slit_model_tmp = np.ones_like(background_slit_model_tmp)*np.median(background_slit_model_tmp)
                    background_slit_zero = np.median(background_slit_model_tmp)
            
                background[:,slit_i,:] = background_slit_model_tmp[np.newaxis,:]+(background_spectrum_smooth_tmp - background_spectrum_zero)[:,np.newaxis]
                # background_spectrum_smooth[:,slit_i] = background_spectrum_smooth_tmp
                # background_slit_model[:,slit_i] = background_slit_model_tmp
            
            result = datacube - background
            self.m_image_out_port.append(result)
            self.m_background_out_port.append(background)
            
            # plot the masks
            if self.m_plot:
                image = np.nanmean(datacube,axis=0)
                fig,axes = plt.subplots(ncols=2,figsize=(8,4))
                axes[0].imshow(image,vmin=np.nanpercentile(image,5),vmax=np.nanpercentile(image,95),origin='lower')
                image_clean = np.nanmean(result,axis=0)
                axes[1].imshow(image_clean,vmin=np.nanpercentile(image_clean,5),vmax=np.nanpercentile(image_clean,95),origin='lower')
                for ax_i in range(2):
                    plot_circle(axes[ax_i],position=planet_pos[i] + lenx/2,radius=self.mask_size_planet,color='w')
                    if self.star_mask:
                        plot_circle(axes[ax_i],position=star_pos[i] + lenx/2,radius=self.mask_size_star,color='w')
                plt.show()
            

        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history(
            'Background subtracted', 
            'Background parameters: mask size = %.4f, filter size = %.4f, star mask size = %.4f, star mask: %s' % (self.mask_size_planet,self.filter_sigma,self.mask_size_planet,self.star_mask))
        self.m_image_out_port.close_port()
        self.m_background_out_port.close_port()