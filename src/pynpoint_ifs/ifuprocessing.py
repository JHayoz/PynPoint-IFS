"""
Pipeline modules for running spectral PCA and combining the cubes.
"""

import time

from typeguard import typechecked

import numpy as np

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

from pynpoint_ifs.ifuprocessingfunctions import do_PCA_sub,do_derotate_shift

class IFUPostProcessingModule(ProcessingModule):
    """
    Module to subtract the PSF using spectral PCA and combine the data
    """

    __author__ = 'Jean Hayoz'
        
    @typechecked
    def __init__(
        self,
        name_in: str = 'post_processing',
        image_in_tag: str = 'initial_spectrum',
        image_out_tag: str = 'post_processed',
        mask_out_tag: str = 'mask',
        pca_number: int = 5,
        shift_cubes_in_tag: str ='centering_cubes',
        shift: bool = True,
        rotate: bool = True,
        stack: bool = False,
        combine: str = 'median'
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
        mask_out_tag : str
            Tag of the database entry that is written as output for the mask.
        pca_number : int
            number of principal components to remove
        shift_cubes_in_tag : str
            Tag of the database entry with the position towards which the images should be shifted (f.ex. the position of the star)
        shift : bool
            whether to shift the images to center them.
        rotate : bool
            whether to rotate the images to a common orientation.
        stack : bool
            whether stack the images to a unique cube.
        combine : str
            method used to stack the cubes. Can be 'mean', 'median', or 'combine' to consider each pixel separately (i.e. the result for a given pixel is the mean over all the images which contain this pixel)
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUPostProcessingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_mask_out_port = self.add_output_port(mask_out_tag)
        
        self.pca_number = pca_number
        
        if type(shift_cubes_in_tag)==str:
            self.m_shift_in_port = self.add_input_port(shift_cubes_in_tag)
        else:
            self.m_shift_in_port = {}
            for i in range(len(shift_cubes_in_tag)):
                self.m_shift_in_port[i]=self.add_input_port(shift_cubes_in_tag[i])
        
        self.m_shift_cubes_in_tag = shift_cubes_in_tag
        self.m_shift = shift
        self.m_rotate = rotate
        self.m_stack = stack
        self.m_combine = combine
        
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        
        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        
        residuals_list = []
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running PCA subtraction...', start_time)
            if nspectrum_i == 0:
                continue
            
            cube_init = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            
            # PCA subtraction
            residuals,model_psf = do_PCA_sub(cube=cube_init,pca_number=self.pca_number)
            residuals_list += [residuals]
        residuals_list = np.array(residuals_list)
        
        # cube averaging
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
        for i in range(len(residuals_list)):
            progress(i, len(residuals_list), 'Running Cube combination...', start_time)
            
            cube_init = residuals_list[i]
            
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
                cube_wv = final_cube[:,k,:,:]
                if self.m_combine == 'median':
                    combined_cube = np.nanmedian(cube_wv, axis=0)
                    cube_median.append(np.where(mask_final, combined_cube, np.nan))
                elif self.m_combine == 'mean':
                    combined_cube = np.nanmean(cube_wv, axis=0)
                    cube_median.append(np.where(mask_final, combined_cube, np.nan))
                elif self.m_combine == 'combine':
                    n_samples = np.sum(mask_arr_shift,axis=0)
                    n_samples_corr = np.where(n_samples != 0, n_samples, 1)
                    combined_cube = np.nansum(cube_wv, axis=0)/n_samples_corr
                    cube_median.append(combined_cube)
                else:
                    combined_cube = np.nanmedian(cube_wv, axis=0)
                    cube_median.append(np.where(mask_final, combined_cube, np.nan))
                
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