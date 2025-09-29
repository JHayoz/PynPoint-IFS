"""
Pipeline modules for calibration of wavelength. Not recommended, use instead SpyFFIER.
"""

import time
from typeguard import typechecked

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from PyAstronomy.pyasl import dopplerShift

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress

from pynpoint_ifs.ifu_utils import select_cubes,rebin

class IFUWavelengthCalibrationModule(ProcessingModule):
    """
    Module to calibrate the wavelength solution based on cross-correlation with telluric lines. The module extracts the RV shift in each spaxel by idientifying the peak of the CCF.
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'calibrate_datacube',
        cc_in_tag: str = 'cc_cube',
        drv_in_tag: str = 'drv',
        rvshift_out_tag: str = 'rvshift'
    ) -> None:
        """
        Parameters
        ----------
        
        name_in : str
            Unique name of the module instance.
        cc_in_tag : str
            Tag of the database entry that is read as input CCF
        drv_in_tag : str
            Tag of the database entry that is read as input radial velocity axis of the CCF
        rvshift_out_tag : str
            Tag of the database entry that is written as output, in terms of radial velocity shift between the data and the telluric lines.
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUWavelengthCalibrationModule, self).__init__(name_in)
        
        
        self.m_cc_in_port = self.add_input_port(cc_in_tag)
        self.m_drv_in_port = self.add_input_port(drv_in_tag)
        self.m_rvshift_out_port = self.add_output_port(rvshift_out_tag)
        
    
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """

        def extract_rv(drv,cc,sigma = 5,drv_range=50):
            def gauss_fct(x,mu,std,a,b):
                return a*np.exp(-0.5*((x-mu)/(std))**2) + b
            smooth_cc = gaussian_filter(cc,sigma)
            max_i = np.argmax(smooth_cc)
            mask_range = np.logical_and(drv > drv[max_i] - drv_range,drv < drv[max_i] + drv_range)
            p0 = [drv[max_i],sigma,max(smooth_cc),0]
            xdata=drv[mask_range]
            ydata=smooth_cc[mask_range]
            try:
                popt,pcov = curve_fit(gauss_fct,xdata=xdata,ydata=ydata,p0=p0)
                return popt[0]
            except RuntimeError:
                print('PARAMETER NOT FOUND, RETURNING NONE')
                return None
        

        cc_cube = self.m_cc_in_port.get_all()
        drv = self.m_drv_in_port.get_all()
        len_cube,len_rv,len_x,len_y = np.shape(cc_cube)
        mean_rv_shift_arr = []
        for cube_i in range(len_cube):
            tell_rv = np.zeros((len_x,len_y))
            for i in range(len_x):
                for j in range(len_y):
                    rv_signal = extract_rv(drv=drv,cc=cc_cube[cube_i,:,i,j],sigma = 5,drv_range=50)
                    tell_rv[i,j] = rv_signal
            mask_nans = np.isnan(tell_rv)
            mean_rv_shift = np.nanmean(tell_rv[~mask_nans])
            tell_rv[mask_nans] = mean_rv_shift
            mean_rv_shift_arr += [mean_rv_shift]
            self.m_rvshift_out_port.append([tell_rv])
        
        self.m_rvshift_out_port.copy_attributes(self.m_cc_in_port)
        self.m_rvshift_out_port.add_history('Radial velocity shift in the cube', 'Mean RV shift: %.2f' % np.nanmean(mean_rv_shift_arr))
        self.m_rvshift_out_port.close_port()

class IFUWavelengthCorrectionModule(ProcessingModule):
    """
    Module to correct the wavelength solution based on an input RV shift.
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'correct_datacube',
        image_in_tag: str = 'datacube',
        wv_in_tag: str = 'wavelength',
        rvshift_in_tag: str = 'rvshift',
        image_out_tag: str = 'datacube_wvl_shifted',
        method: str = 'full'
    ) -> None:
        """
        Parameters
        ----------
        
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        wv_in_tag : str
            Tag of the database (wavelengths) entry that is read as input.
        rvshift_in_tag : str
            Tag of the database entry that is read as input RV shift for each pixel, either a common RV shift for each spaxel throughout the cube, or one per cube
        image_out_tag : str
            Tag of the database entry that is written as output, after dopplershifting the cube by the amount specified by the RV shift
        method : str
            either 'full' or 'mean': if 'mean', applies the same RV shift for all spaxels, if 'full' applies a different RV shift for each spaxel
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUWavelengthCorrectionModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_rvshift_in_port = self.add_input_port(rvshift_in_tag)
        self.m_image_out_port = self.add_output_port(image_out_tag)
        self.m_method = method
    
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        wavelength = self.m_wv_in_port.get_all()
        datacubes = select_cubes(self.m_image_in_port.get_all(),wavelength)
        
        len_cube,len_wvl,len_x,len_y = np.shape(datacubes)

        rvshift_cube = self.m_rvshift_in_port.get_all()
        len_cube_rv,len_x_rv,len_y_rv = np.shape(rvshift_cube)
        if self.m_method not in ['full','mean']:
            print('WRONG METHOD. METHOD MUST BE FULL OR MEAN')
            self.m_image_out_port.close_port()
            assert(False)
        if len_x_rv != len_x or len_y_rv != len_x:
            print('INPUT IMAGE AND RV SHIFT HAVE DIFFERENT SHAPES!')
            self.m_image_out_port.close_port()
            assert(False)
        rvshift_cube_tmp = np.zeros((len_cube,len_x,len_y))
        if len_cube_rv != len_cube:
            if len_cube_rv != 1:
                print('INVALID SHAPE FOR RV SHIFT CUBE')
                print('The RV shift cube needs to have the same shape as the input image or a length of 1')
                self.m_image_out_port.close_port()
                assert(False)
            for cube_i in range(len_cube):
                rvshift_cube_tmp[cube_i,:,:] = rvshift_cube[0,:,:]
        else:
            for cube_i in range(len_cube):
                rvshift_cube_tmp[cube_i,:,:] = rvshift_cube[cube_i,:,:]
        
        mean_rv_shift = np.nanmean(rvshift_cube_tmp,axis=(1,2))
        for cube_i in range(len_cube):
            cube_shifted_out = np.zeros((len_wvl,len_x,len_y))
            print('Progress: %.2f' % ((cube_i)/(len_cube)*100),end='\r')
            for i in range(len_x):
                for j in range(len_y):
                    spectrum_cube = datacubes[cube_i,:,i,j]
                    if self.m_method == 'full':
                        rvshift_i = rvshift_cube_tmp[cube_i,i,j]
                    elif self.m_method == 'mean':
                        rvshift_i = mean_rv_shift[cube_i]
                    else:
                        self.m_image_out_port.close_port()
                        assert(False)
                    nflux1, wlprime1 = dopplerShift(wavelength, spectrum_cube, -rvshift_i, edgeHandling="firstlast")
                    cube_shifted_out[:,i,j] = nflux1
            self.m_image_out_port.append(cube_shifted_out)
        
        
        self.m_image_out_port.copy_attributes(self.m_image_in_port)
        self.m_image_out_port.add_history('Corrected the residual radial velocity shift in the cube', 'Method %s, mean RV: %.2f' % (self.m_method,np.nanmean(mean_rv_shift)))
        self.m_image_out_port.close_port()

class IFUTelluricsWavelengthCalibrationModule(ProcessingModule):
    """
    Module to correct the wavelength solution based on cross-correlation with a tellurics transmission spectrum
    """
    
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        tellurics_wvl: np.array,
        tellurics_flux: np.array,
        name_in: str = 'measure_wavelength_shift',
        image_in_tag: str = 'datacube',
        wv_in_tag: str = 'wavelength',
        wavelength_shift_out_tag: str = 'wavelength_shift',
        cc_accuracy: int = 10
    ) -> None:
        """
        Parameters
        ----------
        
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that is read as input.
        wv_in_tag : str
            Tag of the database (wavelengths) entry that is read as input.
        wavelength_shift_out_tag : str
            Tag of the database entry that is written as output wavelength shift for each spaxel
        
        Returns
        -------
        NoneType
            None
        """
        
        super(IFUTelluricsWavelengthCalibrationModule, self).__init__(name_in)
        
        self.m_image_in_port = self.add_input_port(image_in_tag)
        self.m_wv_in_port = self.add_input_port(wv_in_tag)
        self.m_wavelength_shift_out_tag = self.add_output_port(wavelength_shift_out_tag)
        self.m_tellurics_wvl = tellurics_wvl
        self.m_tellurics_flux = tellurics_flux
        self.m_cc_accuracy = cc_accuracy
    
    def run(self) -> None:
        """
        Run method of the module.
        
        Returns
        -------
        NoneType
            None
        """
        wavelength = self.m_wv_in_port.get_all()
        mean_wvl_step = np.mean(wavelength[1:]-wavelength[:-1])
        datacubes = select_cubes(self.m_image_in_port.get_all(),wavelength)
        nspectrum = self.m_image_in_port.get_attribute('NFRAMES')
        len_cube,len_wvl,len_x,len_y = np.shape(datacubes)

        # rebin tellurics to have the same wavelength bins as the data
        wlen_tell_cr,transm_tell_cr = rebin(self.m_tellurics_wvl,self.m_tellurics_flux,wavelength,flux_err = None, method='datalike')
        
        start_time = time.time()
        for i, nspectrum_i in enumerate(nspectrum):
            progress(i, len(nspectrum), 'Running IFUTelluricsWavelengthCalibrationModule...', start_time)
            print('')
            cube_i = self.m_image_in_port[i*nspectrum_i:(i+1)*nspectrum_i,:,:]
            shift_trsm = np.zeros((len_x,len_y))
            for xi in range(len_x):
                for yi in range(len_y):
                    print('Progress %.2f' % (100*(xi*len_y + yi + 1)/(len_x*len_y)) + '%',end='\r')
                    
                    spectrum = cube_i[:,xi,yi]
                    
                    #assume that the cube is already continuum-removed
                    #smooth_spectrum = gaussian_filter(spectrum,sigma=40)
                    #spectrum_cr = spectrum - smooth_spectrum
                    
                    tmp_offset, _, _ = phase_cross_correlation(
                        spectrum,
                        transm_tell_cr,
                        normalization=None,
                        upsample_factor=self.m_cc_accuracy,
                        overlap_ratio=0.3)
                    
                    shift_trsm[xi,yi] = tmp_offset[0]*mean_wvl_step
            self.m_wavelength_shift_out_tag.append(shift_trsm, data_dim=3)
            
        self.m_wavelength_shift_out_tag.copy_attributes(self.m_image_in_port)
        self.m_wavelength_shift_out_tag.add_history('Measured the wavelength shift in the cube', 'Accuracy: %i' % (self.m_cc_accuracy))
        self.m_wavelength_shift_out_tag.close_port()
