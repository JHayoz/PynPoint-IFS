"""
Helper functions for flux extraction and calibration.
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

def extract_spectra(datacube,fit_param,radius,plot=False):
    print(len(fit_param))
    planet_spectra = []
    lencube,lenwvl,lenx,leny=np.shape(datacube)
    for cube_i in range(lencube):
        print('Progress: %.2f' % (100*cube_i/lencube),end='\r')
        cube = datacube[cube_i]
        
        spectrum = np.zeros((lenwvl))
        for wvl_i in range(lenwvl):
            img_i = cube[wvl_i]
            if len(fit_param) < 1400:
                pos_x,pos_y = fit_param[cube_i] + np.array([leny,leny])/2
            else:
                pos_x,pos_y = fit_param[cube_i*lenwvl + wvl_i] + np.array([leny,leny])/2
            aperture = CircularAperture((pos_x,pos_y),radius)
            spectrum[wvl_i] = aperture.do_photometry(img_i)[0]
        planet_spectra += [spectrum]
        if plot:
            img = np.mean(cube,axis=0)
            plt.figure()
            plt.imshow(img,origin='lower',vmin=np.percentile(img,5),vmax=np.percentile(img,95))
            aperture.plot(color='w')
            plt.show()
    return np.array(planet_spectra)
def extract_bg(datacubes,fit_param,radius,nb_apertures,dist,plot=True,angle_tilt=False,aperture_position='around'):
    angle_phase = 0
    if angle_tilt:
        angle_phase = 2*np.pi/nb_apertures/2
    bg_apertures = np.array([dist*np.array([np.cos(2*np.pi/nb_apertures*i+angle_phase),np.sin(2*np.pi/nb_apertures*i+angle_phase)]) for i in range(nb_apertures)])
    
    if aperture_position=='around':
        pos_bg_apertures = fit_param[np.newaxis,:,:] + bg_apertures[:,np.newaxis,:]
    else:
        pos_bg_apertures = np.zeros_like(fit_param[np.newaxis,:,:]) + bg_apertures[:,np.newaxis,:]
    
    bg_spectra = np.zeros((len(datacubes),nb_apertures,len(datacubes[0])))
    for pos_i,pos in enumerate(pos_bg_apertures):
        bg_spectra[:,pos_i,:] = extract_spectra(datacubes,pos,radius=radius,plot=plot)
    return bg_spectra
def median_combine_spectra(all_spectra):
    norm_spectra = all_spectra/np.mean(all_spectra,axis=1)[:,np.newaxis]*np.mean(all_spectra)
    extr_spectrum = np.median(norm_spectra,axis=0)
    extr_spectrum_err = np.std(norm_spectra,axis=0)
    return extr_spectrum,extr_spectrum_err
def remove_tellurics(wavelength,spectrum,wvl_trsm,transm,outlier_sigma=3,outlier_iterations=10,filter_med_size=20,filter_savg_size=101,plot=True):
    mask_trsm = np.isin(wvl_trsm,wavelength)
    atmo_trsm_div = spectrum/transm[mask_trsm]
    
    instrum_response_med = median_filter(atmo_trsm_div,size=filter_med_size)
    instrum_response_savg = savgol_filter(instrum_response_med,window_length=filter_savg_size,polyorder=3)
    outliers = np.zeros_like(atmo_trsm_div,dtype=bool)
    result = atmo_trsm_div.copy()
    for iter in range(outlier_iterations):
        instrum_response_med = median_filter(result,size=filter_med_size)
        instrum_response_savg = savgol_filter(instrum_response_med,window_length=filter_savg_size,polyorder=3)
        diff = result-instrum_response_savg
        std = np.std(diff)
        outliers = np.logical_or(outliers,np.abs(diff) > outlier_sigma*std)
        result = np.where(outliers,instrum_response_savg,result)
    if plot:
        plt.figure(figsize=(15,5))
        plt.scatter(wavelength[outliers],atmo_trsm_div[outliers],zorder=11,s=1,color='r')
        plt.plot(wavelength,result)
        plt.plot(wavelength,instrum_response_savg)
        plt.ylim((np.min(result),np.max(result)))
        plt.show()
    return result,instrum_response_savg
def get_stellar_model(teff,logg,feh,wavelength):
    star_params = [teff,logg,feh]
    stellar_model_params=pd.read_csv('/home/ipa/quanz/shared/eris/P112_atmospheres/stellar_models/BT-NextGen/BTNG_K_long_metadata/BTNG_K_long_v2_metadata.csv')
    mask_param = np.ones((len(stellar_model_params)),dtype=bool)
    params_range = {}
    for param_i,param in enumerate(['Teff']):
        unique_param = np.array(sorted(list(dict.fromkeys(stellar_model_params[param].values))))
        delta_param = unique_param[1]-unique_param[0]
        star_p = star_params[param_i]
        
        param_low = unique_param[np.where(unique_param-star_p < 0)[0][-1]]
        param_high = unique_param[np.where(unique_param-star_p > 0)[0][0]]
        params_range[param] = [param_low,param_high]
    mask_param = np.logical_and(mask_param,np.logical_and(stellar_model_params['Teff'] >= params_range['Teff'][0],stellar_model_params['Teff'] <= params_range['Teff'][1]))
    mask_param = np.logical_and(mask_param,stellar_model_params['log_g'] == np.round(logg/0.5)*0.5)
    mask_param = np.logical_and(mask_param,stellar_model_params['FeH'] == np.round(feh/0.5)*0.5)
    alpha_val = min(list(dict.fromkeys(stellar_model_params[mask_param]['alpha'].values)))
    mask_param = np.logical_and(mask_param,stellar_model_params['alpha'] == alpha_val)
    if np.sum(mask_param) == 1:
        (file_cold,T_cold) = stellar_model_params[mask_param].sort_values(by='Teff')[['FILE','Teff']].values[0]
        model_cold = pd.read_csv(file_cold)
        # model_cold_tmp = model_cold.drop(columns='Unnamed: 0').drop_duplicates()
        wlen_model_cold,flux_model_cold = rebin(model_cold['wlen'].values,model_cold['flux'].values,wavelength,method='linear')
        return wlen_model_cold,flux_model_cold
    [(file_cold,T_cold),(file_hot,T_hot)] = stellar_model_params[mask_param].sort_values(by='Teff')[['FILE','Teff']].values[:2]
    print(T_cold,T_hot)
    model_cold = pd.read_csv(file_cold)
    model_hot = pd.read_csv(file_hot)
    # model_hot_tmp = model_hot.drop(columns='Unnamed: 0').drop_duplicates()
    # model_cold_tmp = model_cold.drop(columns='Unnamed: 0').drop_duplicates()
    wlen_model_hot,flux_model_hot = rebin(model_hot['wlen'].values,model_hot['flux'].values,wavelength,method='linear')
    wlen_model_cold,flux_model_cold = rebin(model_cold['wlen'].values,model_cold['flux'].values,wavelength,method='linear')
    mask_hot_overlap = np.isin(wlen_model_hot,wlen_model_cold)
    mask_cold_overlap = np.isin(wlen_model_cold,wlen_model_hot)
    #model_hot_tmp_interp = interp1d(x=wlen_model_hot,y=flux_model_hot)
    #model_hot_tmp_interped = model_hot_tmp_interp(model_cold_tmp['wlen'])
    star_model_flux = flux_model_cold[mask_cold_overlap] + (teff-T_cold)/(T_hot-T_cold)*(flux_model_hot[mask_hot_overlap]-flux_model_cold[mask_cold_overlap])
    return wlen_model_cold[mask_cold_overlap],star_model_flux

def synthetic_photometry(wlen,flux,f):
    """
    f is filter transmission function
    output: synthetic photometry of flux through f
    """
    integrand1 = np.trapz([f(x)*flux[i] for i,x in enumerate(wlen)],wlen)
    integrand2 = np.trapz([f(x) for i,x in enumerate(wlen)],wlen)
    return integrand1/integrand2
def calc_median_filter(f,N_points):
    """
    f is a filter transmission function
    output: median of the filter
    """
    wvl = np.linspace(0.2,8,N_points)
    transmission = [f(xx) for xx in wvl]
    integral = np.trapz(transmission,wvl)
    wvl_i = 4
    cum_distr = 0.
    while cum_distr < integral/2 and wvl_i < len(wvl):
        cum_distr = np.trapz([f(xx) for xx in wvl[:wvl_i]],wvl[:wvl_i])
        wvl_i += 1
    if wvl_i == len(wvl):
        print('median wvl not found')
        return None
    return wvl[wvl_i]

def effective_width_filter(f,N_points):
    """
    f is filter transmission function
    output: width of transmission function if it were a rectangle of equivalent surface area
    """
    wvl = np.linspace(0.2,8,N_points)
    transmission = [f(xx) for xx in wvl]
    area = np.trapz(transmission,wvl)
    max_transm = max(transmission)
    return area/max_transm
def calibrate_stellar_model(wlen_model,flux_model,K_band_mag,wavelength,smooth=True):
    trsm_func = get_trsm_func()
    model_flux_K = synthetic_photometry(wlen_model,flux_model,trsm_func)
    vega_flux_K = 4.283e-10
    flux_model_norm = flux_model/model_flux_K*vega_flux_K*10**(-K_band_mag/2.5)
    if smooth:
        flux_model_norm_smooth = gaussian_filter(flux_model_norm,sigma=100)
    else:
        flux_model_norm_smooth=flux_model_norm
    mask_model_cov = np.isin(wlen_model,wavelength)
    wlen_model_reb_final,flux_model_norm_final = wlen_model[mask_model_cov],flux_model_norm_smooth[mask_model_cov]
    return wlen_model_reb_final,flux_model_norm_final
def get_trsm_func():
    phot_trsm = pd.read_csv('/home/ipa/quanz/user_accounts/jhayoz/Projects/ERIS_GTO_2023/analysis/flux_calibration/2MASS_2MASS.Ks.dat',header=None,delimiter=' ',names=['wlen','trsm'])
    phot_trsm['wlen'] = phot_trsm['wlen']/1e4
    trsm_func = interp1d(x=phot_trsm['wlen'],y=phot_trsm['trsm'],bounds_error=False,fill_value=0)
    return trsm_func

@typechecked
def create_planet_mask(
        im_shape: Tuple[int, int, int],
        size: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Function to create a mask for the central and outer image regions.
    
        Parameters
        ----------
        im_shape : tuple(int, int)
            Image size in both dimensions.
        size : tuple(float, float)
            Size (pix) of the mask, and position of the center (x px, y px), as measured from the center of the frame
    
        Returns
        -------
        np.ndarray
            Image mask.
        """
    
        mask = np.ones(im_shape)
        nwvl = im_shape[0]
        npix = im_shape[1]
    
        if size[0] is not None or size[1] is not None:
    
            if npix % 2 == 0:
                x_grid = y_grid = np.linspace(-npix / 2 + 0.5, npix / 2 - 0.5, npix)
            else:
                x_grid = y_grid = np.linspace(-(npix - 1) / 2, (npix - 1) / 2, npix)
            if nwvl % 2 == 0:
                z_grid = np.linspace(-nwvl / 2 + 0.5, nwvl / 2 - 0.5, nwvl)
            else:
                z_grid = np.linspace(-(nwvl - 1) / 2, (nwvl - 1) / 2, nwvl)
            xx_grid, zz_grid, yy_grid = np.meshgrid(x_grid, z_grid, y_grid )
            rr_grid = np.sqrt((xx_grid-size[1])**2 + (yy_grid-size[2])**2)
    
            if size[0] is not None:
                mask[rr_grid < size[0]] = 0.
    
        return mask