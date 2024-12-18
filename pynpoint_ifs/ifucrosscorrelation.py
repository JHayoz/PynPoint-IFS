"""
Pipeline modules for frame selection.
"""

import sys
import time
import math
import warnings
import copy

from typing import Union, Tuple

import numpy as np
from PyAstronomy.pyasl import crosscorrRV
from joblib import Parallel, delayed

from typeguard import typechecked

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress
from pynpoint.util.image import shift_image, rotate

from pynpoint_ifs.ifu_utils import select_cubes




class CrossCorrelationModule_Jean(ProcessingModule):
    """
    Module to cross-correlate continuum-subtracted IFS data with a spectral template.
    """
    __author__ = 'Jean Hayoz'
    
    @typechecked
    def __init__(
        self,
        name_in: str = 'crosscorr',
        RV: float = 2500.,
        dRV: float = 10.,
        range_CCF_RV: float = 50.,
        image_in_tag: str = 'data_mask',
        data_wv_in_tag: str = 'wavelength_range',
        model_wv: np.ndarray = None,
        model_abs: np.ndarray = None,
        SNR_cube_out_tag: str = 'snr_cube',
        CC_cube_out_tag: str = 'CC_cube',
        RV_data_out_tag: str = 'radial_velocity',
        cpus: int = 100
    ) -> None:
        """
        Constructor of CrossCorrelationModule_Jean.
        
        :param name_in: Unique name of the module instance.
        :type name_in: str
        :param RV: (-RV, RV) is the interval of radial velocity used to doppler-shift the template, in km/s
        :type RV: float
        :param dRV: spacing between each step of radial velocity
        :type dRV: float
        :param range_CCF_RV: region (in radial velocity) away from the peak, that is used to estimate the std of the CCF in order to calculate the SNR
        :type range_CCF_RV: float
        :param image_in_tag: Tag of the database entry that is read as input.
        :type image_in_tag: str
        :param data_wv_in_tag: Tag of the database entry that is read as wavelength axis.
        :type data_wv_in_tag: str
        :param model_wv: wavelength array for the model.
        :type model_wv: ndarray
        :param model_abs: flux array for the model.
        :type model_abs: ndarray
        :param SNR_cube_out_tag: Tag of the database entry that is written as output, saving the 2D SNR molecular map (x,y) -> SNR
        :type SNR_cube_out_tag: str
        :param CC_cube_out_tag: Tag of the database entry that is written as output, saving the 3D CCF (x,y,rv) -> CCF
        :type CC_cube_out_tag: str
        :param RV_data_out_tag: Tag of the database entry that is written as output radial velocity axis for the CCF
        :type RV_data_out_tag: str
        
        :return: None
        """
        
        super(CrossCorrelationModule_Jean, self).__init__(name_in)
        
        self.m_data_wv_in_port = self.add_input_port(data_wv_in_tag)
        self.m_image_in_port = self.add_input_port(image_in_tag)
        
        self.m_SNR_cube_out_port = self.add_output_port(SNR_cube_out_tag)
        self.m_CC_cube_out_port = self.add_output_port(CC_cube_out_tag)
        self.m_RV_data_out_port = self.add_output_port(RV_data_out_tag)
        
        self.m_RV = RV
        self.m_dRV = dRV
        self.range_CCF_RV = range_CCF_RV
        self.m_wv_model = model_wv
        self.m_abs_model = model_abs
        self.m_cpus = cpus
    
    def run(self):
        """
        Run method of the module. Cross-correlates the IFS data with a spectral template, and parallelises the operation (using joblib) for speed.
        
        :return: None
        """
        
        
        wavelength = self.m_data_wv_in_port.get_all()
        
        data = self.m_image_in_port.get_all()
        datacubes = select_cubes(data,wavelength)
        nb_cubes,len_wvl,len_x,len_y = np.shape(datacubes)

        model_wvl = self.m_wv_model
        model_flux_abs = self.m_abs_model

        # identify zeros in data where cross-correlation doesn't have to be calculated
        mask_zeros = np.sum(datacubes,axis=1) == 0
        indices_non_zeros = np.where(mask_zeros == False)
        indices_non_zeros_arr=np.vstack([indices_non_zeros[0],indices_non_zeros[1],indices_non_zeros[2]]).transpose()
        
        
        def calculate_CC_SNR(wv, spectrum, wv_model, model,m_RV,m_dRV,range_CCF_RV):
            rv, cc = crosscorrRV(
                w = wv, f = spectrum, 
                tw = wv_model, tf = model,
                rvmin=-m_RV, rvmax = m_RV+m_dRV, drv = m_dRV,skipedge=0)
            
            # SNR
            max_CC_i = np.argmax(cc)
            CCF_range_mask = np.abs(rv-rv[max_CC_i]) > range_CCF_RV
            if np.sum(CCF_range_mask) == 0:
                print('REGION AWAY FROM CCF PEAK IS EMPTY. CHOOSE SMALLER RV RANGE.')
                assert(False)
            CCF_range=cc[CCF_range_mask]
            std = np.std(CCF_range)
            mean = np.mean(CCF_range)
            snr = (cc[max_CC_i]-mean)/std
            
            return cc, snr
        
        
        RV_N = int(2*self.m_RV/self.m_dRV)+1
        rv = np.linspace(-self.m_RV,self.m_RV,int(2*self.m_RV/self.m_dRV)+1)
        self.m_RV_data_out_port.set_all(rv)
        
        total_it = len(indices_non_zeros_arr)
        start_time = time.time()
        print('Running CrossCorrelationModule...')
        print('Total iteration: %i' % total_it)
        results_computation = Parallel(n_jobs=self.m_cpus,verbose=2)(delayed(calculate_CC_SNR)(
            wavelength, spectrum=datacubes[cube_i,:,i,j], 
            wv_model=model_wvl, model=model_flux_abs,
            m_RV=self.m_RV,m_dRV=self.m_dRV,range_CCF_RV=self.range_CCF_RV) for cube_i,i,j in indices_non_zeros_arr)
        print('FINISHED!')
        result_CCF = np.zeros((nb_cubes,RV_N,len_x,len_y))
        result_SNR = np.zeros((nb_cubes,len_x,len_y))
        for iteration_i,(cube_i,i,j) in enumerate(indices_non_zeros_arr):
            result_CCF[cube_i,:,i,j] = results_computation[iteration_i][0]
            result_SNR[cube_i,i,j] = results_computation[iteration_i][1]

            
        
        self.m_CC_cube_out_port.set_all(result_CCF)
        self.m_SNR_cube_out_port.set_all(result_SNR)
        
        
        self.m_SNR_cube_out_port.copy_attributes(self.m_image_in_port)
        self.m_SNR_cube_out_port.add_history("CCF cube", "SNR cube")
        
        self.m_CC_cube_out_port.copy_attributes(self.m_image_in_port)
        self.m_CC_cube_out_port.add_history("CCF cube", "CCF cube")
        
        self.m_RV_data_out_port.copy_attributes(self.m_image_in_port)
        self.m_RV_data_out_port.add_history("CCF cube", "RV axis")
        
        self.m_SNR_cube_out_port.close_port()
        self.m_CC_cube_out_port.close_port()
        self.m_RV_data_out_port.close_port()


