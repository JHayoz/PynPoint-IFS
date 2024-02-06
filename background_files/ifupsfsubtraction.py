"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import copy
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from sklearn.decomposition import PCA

from typing import Union, Tuple

import numpy as np

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
#from pynpoint.util.image import shift_image


class IFUPSFSubtractionModule(ProcessingModule):
    """
        Module to subtract the stellar signal.
        """

    __author__ = 'Gabriele Cugno'
        
    @typechecked
    def __init__(self,
                 name_in: str = "Select_range",
                 image_in_tag: str = "initial_spectrum",
                 image_out_tag: str = "PSF_sub",
                 gauss_sigma: float = 10.,
                 sigma: Union[float, None] =6.,
                 savgol: Tuple[float, float] = (151,7),
                 run_pca: bool = True,
                 pc: int = 20):
        """
            Constructor of IFUPSFSubtractionModule.
            
            :param name_in: Unique name of the module instance.
            :type name_in: str
            :param image_in_tag: Tag of the database entry that is read as input.
            :type image_in_tag: str
            :param stellar_spectra_in_tag: Tag of the database entry that is read as input.
            :type stellar_spectra_in_tag: str
            :param image_out_tag: Tag of the database entry that is written as output. Should be
            different from *image_in_tag*.
            :type image_out_tag: str
            :param gauss_sigma: value for the smoothing of the residuals
            :type gauss_sigma: float
            :param sigma: sigma to exclude outliers
            :type sigma: float
            
            :return: None
            """
        
        super(IFUPSFSubtractionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        
        self.m_gauss_sigma = gauss_sigma
        self.m_sigma = sigma
        self.m_savgol = savgol
        self.m_pc = pc
        self.m_run_pca = run_pca
    
    def run(self):
        """
            Run method of the module. Convolves the images with a Gaussian kernel.
            
            :return: None
            """
        
        def find_outliers_sub(spectrum, sigma):
            spectrum2 = copy.copy(spectrum)
            sub = medfilt(spectrum2, 7)
            sp_sub = spectrum2-sub
            std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
            spectrum2 = np.where(np.abs(sp_sub)> sigma*std, sub, spectrum2)
            return spectrum2
        
        def PC_analysis(cube, pc):
            
            size_f = np.shape(cube)[1]
            psf = np.mean(cube,axis=0)
            res = cube - psf
            pix_list = np.transpose(res.reshape((len(cube),-1)))
            #pix_list = []
            #for i in range(size_f):
            #    for j in range(size_f):
            #        pix_list.append(cube[:,i,j]-np.mean(cube[:,i,j]))

            #pix_list = np.array(pix_list)

            pca_sklearn = PCA(n_components=pc, svd_solver="arpack")
            pca_sklearn.fit(pix_list)

            pca_rep = np.matmul(pca_sklearn.components_, pix_list.T).T
            model1 = pca_sklearn.inverse_transform(pca_rep)
            residuals = pix_list-model1
    
            im_back = np.zeros_like(cube)
            for i in range(size_f):
                for j in range(size_f):
                    im_back[:,i,j] = residuals[i*size_f+j]

            model_back = np.zeros_like(cube)
            for i in range(size):
                for j in range(size):
                    model_back[:,i,j] = model1[i*size_f+j]
    
            return im_back, model_back
        
        
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        size = self.m_image_in_port.get_shape()[1]
        
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUPSFSubtractionModule...', start_time)
            
            # Create reference spectrum
            cube_init = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            cube_norm = np.zeros_like(cube_init)
            cube_outliers = np.zeros_like(cube_init)
            list_spectra = []
            for i in range(size):
                for j in range(size):
                    if self.m_sigma is None:
                        cube_norm[:,i,j] = np.where(cube_init[:,i,j]!=0,cube_init[:,i,j]/np.nansum(np.abs(cube_init[:,i,j]), axis=0),cube_init[:,i,j])
                    else:
                        cube_outliers[:,i,j] = find_outliers_sub(cube_init[:,i,j], self.m_sigma)
                        cube_norm[:,i,j] = np.where(cube_outliers[:,i,j]!=0,cube_outliers[:,i,j]/np.nansum(np.abs(cube_outliers[:,i,j]), axis=0),cube_outliers[:,i,j])
                    list_spectra.append(cube_norm[:,i,j])

            list_spectra = np.array(list_spectra)
            if self.m_sigma is not None:
                list_spectra_bp = np.zeros_like(list_spectra)
                for k in range(nspectrum_i):
                    list_spectra_bp[:,k] = find_outliers_sub(list_spectra[:,k],2*self.m_sigma)
                reference_spectrum = np.nanmedian(list_spectra_bp, axis=0)
            else:
                reference_spectrum = np.nanmedian(list_spectra, axis=0)
            
            
            # Remove low-pass filtered images
            spectrum_div = np.ones_like(cube_init)
            spectrum_smooth = np.ones_like(cube_init)
            low_pass_cube = np.zeros_like(cube_init)
            for i in range(size):
                for j in range(size):
                    spectrum_div[:,i,j] = cube_norm[:,i,j]/reference_spectrum
                    spectrum_smooth[:,i,j] = savgol_filter(spectrum_div[:,i,j], self.m_savgol[0], self.m_savgol[1])
                    low_pass_cube[:,i,j] = spectrum_smooth[:,i,j]*reference_spectrum
            
            cube_star_sub = cube_norm - low_pass_cube
            if self.m_run_pca:
                # Apply PCA on image
                cube_pca, model_pca = PC_analysis(cube_star_sub, self.m_pc)
                self.m_image_out_port.append(cube_pca)
            else:
                self.m_image_out_port.append(cube_star_sub)

        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history("PSF sub", "PC = "+str(self.m_pc))
        self.m_image_out_port.close_port()
