from pathlib import Path
from astropy.io import fits
import numpy as np
import os
import time
from typing import Union, Tuple
from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

class IFUFitsWritingModule(ProcessingModule):
    """
    Module to write the data under a tag to fits files with the original headers
    """
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'write',
        image_in_tag: str = 'image',
        wvl_in_tag: str = 'wavelength',
        output_dir: str = '',
        name_extension: str = 'SCIENCE',
        overwrite: bool = False
    ) -> None:
        """
        Constructor of IFUFitsWritingModule.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        
        
        :return: None
        """
        
        super(IFUFitsWritingModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wvl_in_port = self.add_input_port(wvl_in_tag)
        
        self.output_dir = output_dir
        self.name_extension = name_extension
        self.overwrite = overwrite

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def run(self):
        """
        Run method of the module.
        
        :return: None
        """

        nspectrum = self.m_image_in_port.get_attribute("NFRAMES")
        files = np.array(self.m_image_in_port.get_attribute('FILES'),dtype=str)
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUFitsWritingModule...', start_time)
            if nspectrum_i == 0:
                continue
            # get the original file
            file_path = Path(files[i])
            file_name = file_path.parent / (file_path.stem + '.fits') # correct for missing letters
            hdr = fits.getheader(file_name)
            arcfile = hdr['ARCFILE']
            new_path = Path(self.output_dir) / f'{self.name_extension}_{arcfile}'
            
            if os.path.exists(new_path) and not self.overwrite:
                continue
            # get the cube
            cube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]

            # get the wavelength axes
            wavelength = self.m_wvl_in_port[:]
            
            hdu_list =  fits.open(file_name)
            science_nframes,science_lenx,science_leny = np.shape(hdu_list[1].data)
            wvlparams = [hdu_list[1].header['CRVAL3'],hdu_list[1].header['CD3_3']]
            wavelength_axis = np.array([wvlparams[0] + i*wvlparams[1] for i in range(science_nframes)])
            header_original = hdu_list[1].header
            hdu_list.close()

            # add nans to fill up the cube
            # along wavelength
            add_left = np.sum(wavelength_axis < wavelength[0])
            add_right = np.sum(wavelength_axis > wavelength[-1])
            lenwvl,lenx,leny = np.shape(cube)
            new_cube = np.vstack([np.nan*np.ones((add_left,lenx,leny)),cube,np.nan*np.ones((add_right,lenx,leny))])
            lennewcube = len(new_cube)
            # along x
            add_left = (science_lenx - lenx)//2
            add_right = add_left
            new_cube = np.hstack([np.nan*np.ones((lennewcube,add_left,leny)),new_cube,np.nan*np.ones((lennewcube,add_right,leny))])
            lenxnewcube = len(new_cube[0])
            # along y
            add_left = (science_leny - leny)//2
            add_right = add_left
            new_cube = np.dstack([np.nan*np.ones((lennewcube,lenxnewcube,add_left)),new_cube,np.nan*np.ones((lennewcube,lenxnewcube,add_right))])

            
            new_imagehdu = fits.ImageHDU(data=new_cube,header=header_original)
            
            hdu_list =  fits.open(file_name)
            new_hdulist = fits.HDUList([hdu_list[0],new_imagehdu,hdu_list[2],hdu_list[3]])
            new_hdulist.writeto(new_path,overwrite=self.overwrite)
            
            hdu_list.close()