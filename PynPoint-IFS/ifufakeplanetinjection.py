import numpy as np
from scipy.ndimage import shift
from scipy.interpolate import interp1d
import time
from typeguard import typechecked
from typing import Union,List
from joblib import Parallel, delayed

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

class IFUFakePlanetInjectionModule(ProcessingModule):
    """
    Module to inject a fake planet into IFU data.
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'inject_planet',
        image_in_tag: str = 'raw',
        psf_ref_in_tag: str = 'PSF_reference',
        wvl_in_tag: str = 'wavelength_range',
        image_out_tag: str = 'raw_fake',
        star_position_param_tag: str = 'star_position',
        fake_planet_position: List[float] = [1.,0.],
        fake_planet_contrast: float = 0.,
        science_dit: float = 1.,
        reference_dit: float = 1.,
        star_spectrum: Union[np.ndarray, List[float]] = [1.],
        fake_planet_spectrum: Union[np.ndarray, List[float]] = [1.],
        photometric_filter_path: str = None,
        ncpus: int = 1
    ) -> None:
        """
        Parameters
        ----------
        
        name_in: str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        psf_ref_in_tag : str
            Tag of the database entry that is read as input for the PSF reference.
        wvl_in_tag : str
            Tag of the database for the wavelength axis
        image_out_tag : str
            Tag of the database entry that is written as output. Should be
        different from *image_in_tag*.
        star_position_param_tag : str
            Tag of the database with the position of the star
        fake_planet_position : List[float]
            Position of the fake planet to inject. The first element is the position angle measured in degree, 
        the second is the angular separation measured in arcsecs.
        fake_planet_contrast : float
            Contrast of the fake planet to inject measured in delta-magnitude.
        science_dit : float
            DIT of the science images
        reference_dit : float
            DIT of the reference images
        star_spectrum : Union[np.ndarray, List[float]]
            Stellar spectrum
        fake_planet_spectrum : Union[np.ndarray, List[float]]
            Spectrum of the planet to inject
        photometric_filter_path : str
            Path to the location of the filter transmission function used to measure the photometry. Needs to be within the wavelength interval of both the star and planet spectrum
        ncpus : int
            if larger than 1, uses joblib to parallelise the fake planet injection.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUFakePlanetInjectionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_psf_ref_in_port = self.add_input_port(psf_ref_in_tag)
        self.m_wvl_in_port = self.add_input_port(wvl_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_star_position_param_port = self.add_input_port(star_position_param_tag)

        self.m_fake_planet_position = fake_planet_position
        self.m_fake_planet_contrast = fake_planet_contrast
        self.m_science_dit = science_dit
        self.m_reference_dit = reference_dit
        self.m_star_spectrum = star_spectrum
        self.m_fake_planet_spectrum = fake_planet_spectrum
        self.m_photometric_filter_path = photometric_filter_path
        self.m_ncpus = ncpus
        
        
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        def shift_and_add(datacube,fake_planet,shift_vector):
            shifted_fake_planet = np.zeros_like(fake_planet)
            lenwvl = len(datacube)
            for wvl_i in range(lenwvl):
                shifted_fake_planet[wvl_i] = shift(fake_planet[wvl_i],(shift_vector[1],shift_vector[0]))
            shifted_cube = datacube + shifted_fake_planet
            return shifted_cube
        
        # Check the inputs
        if (len(self.m_star_spectrum) != len(self.m_fake_planet_spectrum)) or (len(self.m_star_spectrum) != self.m_wvl_in_port.get_shape()[0]) or (len(self.m_star_spectrum) != self.m_psf_ref_in_port.get_shape()[0]):
            raise RuntimeError(
                "The star, planet and wavelength axis of the data don't have the same length"
            )
        if (self.m_image_in_port.get_shape()[1] != self.m_psf_ref_in_port.get_shape()[1]) or (self.m_image_in_port.get_shape()[2] != self.m_psf_ref_in_port.get_shape()[2]):
            raise RuntimeError(
                "The PSF reference needs to have the same shape as the science data"
            )
        
        wavelength = self.m_wvl_in_port.get_all()
        lenwvl = len(wavelength)
        lencube,lenx,leny = self.m_image_in_port.get_shape()
        pixscale = self.m_image_in_port.get_attribute('PIXSCALE')
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        
        # First define the fake planet to inject
        
        # 1) normalise the spectra using synthetic photometry
        filter_wvl,filter_trsm = np.loadtxt(self.m_photometric_filter_path,delimiter=',')
        filter_trsm_interp = interp1d(x=filter_wvl,y=filter_trsm,bounds_error=False,fill_value=0)
        filter_trsm_ev = filter_trsm_interp(wavelength)
        norm_star_spectrum = self.m_star_spectrum / np.sum(self.m_star_spectrum * filter_trsm_ev)
        norm_planet_spectrum = self.m_fake_planet_spectrum / np.sum(self.m_fake_planet_spectrum * filter_trsm_ev)
        # 2) define the amplitude of the signal to inject
        contrast = 10**(-self.m_fake_planet_contrast/2.5)
        amplitude = contrast/self.m_reference_dit*self.m_science_dit
        # 3) get the PSF reference
        PSF_reference_cube = self.m_psf_ref_in_port.get_all()
        
        fake_planet = amplitude*PSF_reference_cube/norm_star_spectrum[:,np.newaxis,np.newaxis]*norm_planet_spectrum[:,np.newaxis,np.newaxis]
        
        # Define the position vector of the fake planet to inject
        angle_rad = self.m_fake_planet_position[1]/180*np.pi
        rel_position = self.m_fake_planet_position[0]/pixscale*np.array([-np.sin(angle_rad),np.cos(angle_rad)])

        # Get the position of the star in each frame
        star_position = self.m_star_position_param_port.get_all()[:,(0,2)]

        # Inject the fake planets
        start_time = time.time()
        if self.m_ncpus > 1:
            cube_injected = Parallel(n_jobs=self.m_ncpus,verbose=2)(delayed(shift_and_add)(
                datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:],
                fake_planet = fake_planet,
                shift_vector = star_position[i] + rel_position) for i,nspectrum_i in enumerate(nspectrum))
            data_injected = np.array(cube_injected).reshape((-1,lenx,leny))
            self.m_image_out_port.set_all(data_injected)
        else:
            for i, nspectrum_i in enumerate(nspectrum):
                progress(i, len(nspectrum), 'Running IFUFakePlanetInjectionModule...', start_time)
                datacube = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
                
                cube_injected = shift_and_add(datacube,fake_planet,shift_vector=star_position[i] + rel_position)
                # # define vector by which to shift the PSF reference
                # shift_vector = star_position[i] + rel_position
                # 
                # # shift PSF reference
                # shifted_fake_planet = np.zeros_like(PSF_reference_cube)
                # for wvl_i in range(lenwvl):
                #     shifted_fake_planet[wvl_i] = shift(fake_planet[wvl_i],(shift_vector[1],shift_vector[0]))
                # # inject the fake planet
                # cube_injected = datacube + shifted_fake_planet
                # store results
                self.m_image_out_port.append(cube_injected)
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history(
            'Fake planet injected', 
            'Fake planet parameters: position (r,rho) = (%.2f, %.2f), contrast = %.2f' % (self.m_fake_planet_position[0],self.m_fake_planet_position[1],self.m_fake_planet_contrast))
        self.m_image_out_port.close_port()