import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter,median_filter
from scipy.signal import savgol_filter,medfilt,resample
from scipy.interpolate import splrep, BSpline
import pynpoint as pp
import warnings
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
from scipy.stats import median_abs_deviation
import skycalc_ipy
from datetime import datetime
from pathlib import Path

# fit an 2D-airy model to some data
# x_guess,y_guess,amplitude_guess,radius_guess are parameter guesses for the fit
def fit_airy(box,x_guess,y_guess,amplitude_guess,radius_guess):
    
    x_length = box.shape[1]
    y_length = box.shape[0]        
    
    y, x = np.mgrid[:y_length, :x_length]
    
    p_init = AiryDisk2D(amplitude=amplitude_guess, x_0=x_guess, y_0=y_guess,radius=radius_guess)
    
    fit_p = fitting.LevMarLSQFitter()
    
    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, box,maxiter=20000)
    
    return p.x_0.value, p.y_0.value, p.amplitude.value, p.radius.value

# fit an 2D-Gaussian model to some data
# x_guess,y_guess,amplitude_guess,radius_guess are parameter guesses for the fit
def fit_gaussian(box,x_guess,y_guess,amplitude_guess,radius_guess):

    x_length = box.shape[1]
    y_length = box.shape[0]        
    
    y, x = np.mgrid[:y_length, :x_length]
    
    p_init = Gaussian2D(amplitude=amplitude_guess, x_mean=x_guess, y_mean=y_guess,x_stddev=radius_guess,y_stddev=radius_guess,theta=None)
    
    fit_p = fitting.LevMarLSQFitter()
    
    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, box,maxiter=20000)
    
    return p.x_mean.value, p.y_mean.value, p.amplitude.value, (p.x_fwhm + p.y_fwhm)/2

# smoothes using a Gaussian kernel, determines the max value of the smoothed image, then fits a 2D model to the non-smoothed data using 
# the smoothed data as initial value for the fit
# image is the input 2D data, sigma is the std deviation of the gaussian filter, box determines how many neighbouring pixels are considered
# to determine the amplitude, method is the function used for the fit, can be airy or gaussian
def fit_center_custom(image,sigma=3,box=1,method='airy',sign='positive'):
    size_y = len(image)
    size_x = len(image[0])
    if sigma > 0:
        img_smooth = gaussian_filter(image,sigma=sigma)
    else:
        img_smooth = image
    if sign=='positive':
        #max_x = np.nanargmax(np.nanmean(img_smooth,axis=0))
        #max_y = np.nanargmax(np.nanmean(img_smooth,axis=1))
        max_x,max_y = np.unravel_index(np.argmax(img_smooth),np.shape(img_smooth))
    else:
        #max_x = np.nanargmax(np.nanmean(-img_smooth,axis=0))
        #max_y = np.nanargmax(np.nanmean(-img_smooth,axis=1))
        max_x,max_y = np.unravel_index(np.argmax(-img_smooth),np.shape(img_smooth))
    if max_x<box or max_x > size_x - box or max_y<box or max_y > size_y - box:
        print('Star is outside of the FoV or the box is too large')
    amplitude = np.nanmean(image[max_x-box:max_x+box+1,max_y-box:max_y+box+1])
    if sign=='positive':
        image_used = img_smooth[:,:] - np.median(img_smooth)
    elif sign=='negative':
        image_used = -img_smooth[:,:] + np.median(img_smooth)
    else:
        print('Choose a valid sign: positive or negative')
    if method=='airy':
        x0,y0,a0,r0 = fit_airy(box=image_used,x_guess=max_y,y_guess=max_x,amplitude_guess=amplitude,radius_guess=sigma)
    elif method=='gaussian':
        x0,y0,a0,r0 = fit_gaussian(box=image_used,x_guess=max_y,y_guess=max_x,amplitude_guess=amplitude,radius_guess=sigma)
    else:
        print('Choose a valid method for the fit: airy or gaussian')
    if sign=='negative':
        a0 = -a0
    return x0,y0,a0,r0

# organises the IFU data as array of datacubes with axes (nframes, wavelength, x-axis, y-axis)
def select_cubes(data,wavelength):
    length_cube = len(wavelength)
    shape_cube = np.shape(data)
    return data.reshape((int(shape_cube[0]/length_cube),length_cube,shape_cube[1],shape_cube[2]))
def get_cubes(pipeline,data_tag,wvl_tag):
    data = pipeline.get_data(data_tag)
    wavelength = pipeline.get_data(wvl_tag)
    datacubes = select_cubes(data,wavelength)
    return datacubes,wavelength
def plot_2D_data(pipeline,data_tag,wvl_tag,plot='all'):
    slitlet_layout = np.array([9,8,10,7,11,6,12,5,13,4,14,3,15,2,16,1,32,17,31,18,30,19,29,20,28,21,27,22,26,23,25,24])
    datacubes,wavelength = get_cubes(pipeline,data_tag,wvl_tag)
    lencube,lenwvl,lenx,leny = np.shape(datacubes)
    images2D = np.reshape(datacubes,(lencube,lenwvl,-1))
    if plot == 'all':
        stop = lencube
    else:
        stop = int(plot)
    for img_i in range(lencube):
        if img_i >= stop:
            break
        img = images2D[img_i]
        plt.figure(figsize=(15,10))
        plt.imshow(img,vmin=np.percentile(img,5),vmax=np.percentile(img,95))
        plt.title('Cube %i' % img_i)
        plt.show()
    return images2D
# plot IFU data either as the wvl-averaged cubes (single) or wvl- and time-averaged image (full)
def plot_data(pipeline,data_tag,wvl_tag,method,contour=True,scatter_data=None):
    datacubes,wavelength = get_cubes(pipeline,data_tag,wvl_tag)
    lenwvl = len(wavelength)
    lencube=len(datacubes)
    if not (scatter_data is None):
        if len(scatter_data) == lencube:
            print('Keep same format scatter data')
            scatter_data_reformat = scatter_data
        elif len(scatter_data) == lenwvl*lencube:
            scatter_data_reformat = np.nanmean(np.array(scatter_data).reshape((lencube,lenwvl,2)),axis=1)
            print('Take the mean over the wavelength scatter data')
        elif len(scatter_data) == 2:
            scatter_data_reformat = [scatter_data for i in range(lencube)]
            print('Constant scatter data')
        else:
            print('PROBLEM WITH FORMAT OF SCATTER DATA')
    if method == 'full':
        images = [np.nanmean(datacubes,axis=(0,1))]
        if not (scatter_data is None):
            scatter_data_reformat = [np.nanmean(scatter_data_reformat,axis=0)]
    elif method == 'single':
        images = np.nanmean(datacubes,axis=1)
        if not (scatter_data is None):
            scatter_data_reformat = scatter_data_reformat
    else:
        print('CHOOSE BETWEEN METHOD=FULL OR METHOD=SINGLE')
    for img_i in range(len(images)):
        img = images[img_i]
        plt.figure()
        if contour:
            plt.contour(img,cmap='RdBu')
        plt.imshow(img,vmin=np.nanpercentile(img,5),vmax=np.nanpercentile(img,95),origin='lower')
        plt.title('Cube %i' % img_i)
        if not (scatter_data is None):
            plt.scatter(x=scatter_data_reformat[img_i][0],y=scatter_data_reformat[img_i][1],color='r')
        plt.show()
    return datacubes
# extract the wavelength info from an ERIS dataset reduced with esoreflex
# outputs the wavelength axis, and the number of frames (i.e. nb of wavelength channels) within each 3D cube
def get_wavelength_info(science_files):
    science_nframes = []
    for file in science_files:
        hdu = fits.open(file)
        hdr = hdu[1].header
        if 'CDELT3' in hdr.keys():
            dwvl = hdr['CDELT3']
        elif 'CD3_3' in hdr.keys():
            dwvl = hdr['CD3_3']
        wvl_params = [hdr['CRVAL3'],dwvl]
        data = hdu[1].data
        science_nframes += [len(data)]
        hdu.close()
    wavelength_axis = [wvl_params[0] + i*wvl_params[1] for i in range(science_nframes[0])]
    return wavelength_axis,science_nframes

# extract the wavelength info from an ERIS dataset reduced with esoreflex
# outputs the wavelength axis
def get_wavelength(file,header_crval='CRVAL3',header_cd='CD3_3'):
    path = Path(file)
    filepath = str(path.parent / path.stem) + '.fits'
    hdr = fits.getheader(filepath,1)
    if (not header_crval in hdr.keys()) or (not header_cd in hdr.keys()):
        raise RuntimeError(
            f"The header does not contain the keyword {header_crval} or {header_cd}"
        )
    rval = hdr[header_crval]
    rshift = hdr[header_cd]
    lencube = hdr['NAXIS3']
    # wavelength = np.arange(rval,rval+lencube*rshift,rshift)
    wavelength = np.array([rval + i*rshift for i in range(lencube)])
    return wavelength
# sort fits files according to the chronological order in which they were observed
def sort_files(file_arr):
    science_date = {}
    for file_i in range(len(file_arr)):
        filen = file_arr[file_i]
        hdr = fits.getheader(filen)
        science_date[filen] = hdr['ARCFILE']
    sorted_files = list(dict(sorted(science_date.items(), key=lambda item: item[1])).keys())
    return sorted_files

# function to replace outliers by comparing their distance to a (rolling) median filter in terms of median absolute deviation
# outliers are replaced by the value of the median filter
def replace_outliers(spectrum, sigma, filt_size = 31,iterate=1):
    spectrum2 = copy.copy(spectrum)
    for iter in range(iterate):
        smooth = median_filter(spectrum2,filt_size)
        sp_sub = spectrum2-smooth
        # std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
        std = median_abs_deviation(sp_sub)
        spectrum2 = np.where(np.abs(sp_sub)> sigma*std, smooth, spectrum2)
    return spectrum2

# Extract the spectrum using a photometric aperture centered on the brightest spaxel
def extract_star_spectrum(datacube):
    img = np.mean(datacube,axis=0)
    max_x,max_y = np.unravel_index(np.argmax(img),np.shape(img))
    ap = CircularAperture((max_y,max_x),5)
    spectrum = []
    for wvl_i in range(len(datacube)):
        spectrum += [ap.do_photometry(datacube[wvl_i])[0][0]]
    return spectrum

# Input: y is data that is parametrized by x
# Output: fits a spline to y for each piecewise continuous intervals of y, thus avoiding the problem caused by jumps
def piecewise_spline_fit(x,y,split_fit = True,s=0):
    if not split_fit:
        y_model = splrep(x, y, s=s)
        y_model_ev = BSpline(*y_model)(x)
    else:
        # identify intervals by computing the derivative
        derivative = y[1:]-y[:-1]
        derivative_model = splrep(x[:-1], derivative, s=s)
        derivative_model_ev = BSpline(*derivative_model)(x[:-1])
        model_err = derivative-derivative_model_ev
        std_deriv = np.std(model_err)
        
        # defines the bounds of the subset of frames when the model of the derivative is bad
        bounds = np.append(np.where(np.abs(model_err) > 3*std_deriv)[0],len(derivative))+1 
        
        # remove neighbouring bounds
        bounds_no_neighbour = bounds[np.append(bounds[1:]-bounds[:-1]>2,True)]
        
        # create the intervals
        y_model_ev = np.zeros((len(x)))
        index_x_i = 0
        for bound_i in range(len(bounds_no_neighbour)):
            x_indices = np.arange(index_x_i,bounds_no_neighbour[bound_i])
            x_data_interv = x[x_indices]
            y_data_interv = y[index_x_i:bounds_no_neighbour[bound_i]]
            
            y_model_interv_i = splrep(x_data_interv, y_data_interv, s=s)
            y_model_ev[x_indices] = BSpline(*y_model_interv_i)(x_data_interv)
            
            index_x_i = bounds_no_neighbour[bound_i]
    return y_model_ev
def snr_map(rv,cc,signal_range=(-100,100),std_interval = 100):
    lenrv,lenx,leny=np.shape(cc)
    snr=np.zeros((lenx,leny))
    mask_signal = np.logical_and(rv>=signal_range[0],rv<=signal_range[1])
    index0 = int(np.sum(rv<signal_range[0]))
    for i in range(lenx):
        for j in range(leny):
            signal = np.max(cc[:,i,j][mask_signal])
            argmax = np.argmax(cc[:,i,j][mask_signal])
            rvmax = rv[argmax + index0]
            mask_std = np.abs(rv-rvmax) >= std_interval
            std = np.std(cc[:,i,j][mask_std])
            if std == 0:
                snr[i,j] = 0
            else:
                snr[i,j] = signal / std
    return snr
def product_SNR_mol(CC_mol,rv,molecules,range_CCF_RV,signal_range=(-100,100)):
    # CC_mol: dict of CCFs
    # molecules: list of keys of CC_mol to be multiplied together
    # rv: 1-D axis of RV steps
    CC_tot = np.product([CC_mol[key] for key in molecules],axis=0)
    snr_from_CC = snr_map(rv,CC_tot,signal_range=signal_range,std_interval = range_CCF_RV)
    return snr_from_CC
def sort_dither_time(*args,obs_times):
    # sort arrays *args according to time
    df = pd.DataFrame()
    for arg_i,arg in enumerate(args):
        df[arg_i] = arg
    # df['Y'] = dither_y
    df['Date'] = obs_times
    df['Frame_nb'] = np.arange(len(obs_times))
    return df.sort_values(by='Date').values.transpose()
def define_groups(dither_x,dither_y,obs_times,pause_threshold=400):
    # groups defined close in time and at the same dither position
    groups = []
    for frame_i in range(len(dither_x)):
        if frame_i == 0:
            groups += [[0]]
        else:
            if (dither_x[frame_i] == dither_x[frame_i-1]) and (dither_y[frame_i] == dither_y[frame_i-1]) and (obs_times[frame_i] - obs_times[frame_i-1]) < pause_threshold:
                groups[-1] += [frame_i]
            else:
                groups += [[frame_i]]
    return groups
def get_groups(pipeline,tag,plot=True):
    # sort groups of frames along time and close in dither position
    dither_x = pipeline.get_attribute(tag,attr_name='DITHER_X',static=False).astype(str)
    dither_y = pipeline.get_attribute(tag,attr_name='DITHER_Y',static=False).astype(str)
    date = pipeline.get_attribute(tag,attr_name='DATE',static=False).astype(str)
    timestamps = []
    for date_i in date:
        timestamps += [datetime.fromisoformat(date_i[:-5]).timestamp()]
    obs_times = np.array(timestamps) - timestamps[0]
    dither_x_sorted,dither_y_sorted,date_sorted,frame_index = sort_dither_time(dither_x,dither_y,obs_times=obs_times).astype(float)
    if is_sorted(frame_index):
        print('Observations already sorted along time')
    else:
        print('CAREFUL: observations not sorted along time')
    if plot:
        plt.figure()
        plt.plot(date_sorted,dither_x_sorted,label='X')
        plt.plot(date_sorted,dither_y_sorted,label='Y')
        for frame_i in range(len(frame_index)):
            plt.annotate(int(frame_index[frame_i]),(date_sorted[frame_i],dither_x_sorted[frame_i]+0.05*((-1)**frame_i+1)),fontsize=10)
        plt.legend()
        plt.title('Dither positions along time')
        plt.show()
    groups = define_groups(dither_x,dither_y,obs_times,pause_threshold=400)
    # in case frames were not sorted along time
    groups_unsorted_frames = []
    for group_i in groups:
        new_group = []
        for el in group_i:
            new_group += [int(frame_index[el])]
        groups_unsorted_frames += [new_group]
    return groups,frame_index.astype(int)
def is_sorted(a):
    # checks whether an array is sorted
    return np.all(a[:-1] <= a[1:])
# lambda over D, with wvl in micrometers, and D in meters. Outputs lambda/D in arcseconds
def lambda_D(wvl,D):
    return 180./np.pi*60*60*wvl*1e-6/D

# function to down-sample a spectrum to a lower spectral resolution.
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
def rebin(wlen,flux,wlen_data,flux_err = None, method='linear'):
    #wlen larger than wlen_data
    
    
    #if method == 'linear':
    #extends wlen linearly outside of wlen_data using the spacing on each side
    if method == 'linear':
        stepsize_left = abs(wlen_data[1]-wlen_data[0])
        
        N_left = int((wlen_data[0]-wlen[0])/stepsize_left)-1
        wlen_left = np.linspace(wlen_data[0]-N_left*stepsize_left,
                                wlen_data[0],
                                N_left,
                                endpoint=False)
        
        stepsize_right = wlen_data[-1]-wlen_data[-2]
        
        N_right = int((wlen[-1]-wlen_data[-1])/stepsize_right)-1
        wlen_right = np.linspace(wlen_data[-1]+stepsize_right,
                                wlen_data[-1]+(N_right+1)*stepsize_right,
                                N_right,
                                endpoint=False)
        
        wlen_temp = np.concatenate((wlen_left,wlen_data,wlen_right))
    elif method == 'datalike':
        wlen_temp = wlen_data
    if flux_err is not None:
        assert(np.shape(flux_err)==np.shape(flux))
        flux_temp,flux_new_err = spectres(wlen_temp,wlen,flux,spec_errs = flux_err)
        return wlen_temp,flux_temp,flux_new_err
    else:
        flux_temp = spectres(wlen_temp,wlen,flux)
        return wlen_temp,flux_temp

# Loads a library of spectral templates into a dictionary, and remove the continuum to prepare for cross-correlation
def load_spectral_templates(wavelength):
    molecules = ['H2O','CO','C2H2','CH4','FeH','H2S','NH3','SiO','TiO','HCN']
    wlen_mol,flux_mol = {},{}
    for mol_i in range(len(molecules)):
        mol_template = np.loadtxt('/home/ipa/quanz/user_accounts/gcugno/SINFONI/GQLup/Templates/Spectrum_' + molecules[mol_i] + '.txt')
        t_wvl,t_flux = mol_template.transpose()
        mask_data = np.logical_and(t_wvl > wavelength[0]-0.1,t_wvl < wavelength[-1]+0.1)
        t1_wvl,t1_flux = t_wvl[mask_data],t_flux[mask_data]
        t2_wvl,t2_flux = rebin(t1_wvl,t1_flux,wavelength,flux_err = None, method='linear')
        smooth_tell = gaussian_filter(t2_flux,20)
        t2_flux_cont_rm = t2_flux - smooth_tell
        wlen_mol[molecules[mol_i]] = t2_wvl
        flux_mol[molecules[mol_i]] = t2_flux_cont_rm
    return wlen_mol,flux_mol

# Loads a library of spectral templates into a dictionary, and remove the continuum to prepare for cross-correlation
def load_custom_templates(paths,wavelength,sigma=60,remove_cont = True,method='linear'):
    wlen_mol,flux_mol = {},{}
    for path_i,path in enumerate(paths):
        filepath = Path(path)
        mol_template = pd.read_csv(filepath)
        molecule = str(filepath.stem)[str(filepath.stem).find('_')+1:]
        wlen,flux = rebin(mol_template['wlen'].values,mol_template['flux'].values,wavelength,flux_err = None, method=method)
        if remove_cont:
            smooth = gaussian_filter(flux,sigma)
            flux_final = flux - smooth
        else:
            flux_final = flux
        wlen_mol[molecule] = wlen
        flux_mol[molecule] = flux_final
    return wlen_mol,flux_mol

def smooth_star_center(pipeline,data_tag = 'star_position_onstar_after',wavelength_tag = 'wavelength_onstar_after',outlier=8, outlier_filter = 31,iterate=1,savgol_sigma=41,smoothing='savgol',spline_s=0.02,save=False,plot=True):
    data = pipeline.get_data(data_tag)
    wavelength = pipeline.get_data(wavelength_tag)
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(data[:,0])
        plt.plot(data[:,2])
        plt.title('Original fit params')
        plt.show()
    star_positions = np.reshape(np.vstack([data[:,0],data[:,2]]).transpose(),(-1,len(wavelength),2))
    
    star_positions_clean = np.zeros_like(star_positions)
    for img_i in range(len(star_positions)):
        star_positions_clean[img_i,:,0] = replace_outliers(star_positions[img_i,:,0], sigma=outlier,filt_size=outlier_filter)
        star_positions_clean[img_i,:,1] = replace_outliers(star_positions[img_i,:,1], sigma=outlier,filt_size=outlier_filter)
    
    photocenter_corr = np.zeros_like(star_positions_clean)
    for img_i in range(len(star_positions_clean)):
        #tmp = replace_outliers(np.array(photocenter_corr_x[img_i]), sigma=8)
        for coord_i in range(2):
            tmp = star_positions_clean[img_i,:,coord_i]
            if smoothing=='savgol':
                smooth = savgol_filter(tmp,savgol_sigma,3)
            else:
                smooth = median_filter(tmp,size=savgol_sigma)
            photocenter_corr[img_i,:,coord_i] = piecewise_spline_fit(x=np.arange(len(smooth)),y=smooth,split_fit=False,s=spline_s)
    if plot:
        for img_i in range(len(star_positions_clean)):
            plt.figure(figsize=(10,5))
            plt.plot(star_positions_clean[img_i,:,0]-star_positions_clean[img_i,0,0])
            plt.plot(photocenter_corr[img_i,:,0]-star_positions_clean[img_i,0,0])
            plt.plot(star_positions_clean[img_i,:,1]-star_positions_clean[img_i,0,1])
            plt.plot(photocenter_corr[img_i,:,1]-star_positions_clean[img_i,0,1])
            plt.title('Smoothed out fit for image %i' % img_i)
            plt.show()
    photocenter_corr_resh = photocenter_corr.reshape((-1,2))
    new_fit_params = np.zeros((len(photocenter_corr_resh),14))
    new_fit_params[:,0] = photocenter_corr_resh[:,0]
    new_fit_params[:,2] = photocenter_corr_resh[:,1]
    if save:
        out_port = pp.core.dataio.OutputPort(data_tag + '_smooth', data_storage_in=pipeline.m_data_storage)
        out_port.set_all(new_fit_params)
        in_port = pp.core.dataio.InputPort(data_tag, data_storage_in=pipeline.m_data_storage)
        out_port.copy_attributes(in_port)
        out_port.close_port()
        print('SAVED SMOOTHED PARAMETERS UNDER TAG: %s' % (data_tag + '_smooth'))
    return new_fit_params

def centroid_pointsource(pipeline,image_cube,filter_sigma=1,plot=True,save=True,save_fit_tag = 'star_position'):
    assert(len(np.shape(image_cube)) == 3)
    nb_images,len_x,len_y = np.shape(image_cube)
    planet_position = np.zeros((nb_images,14))
    for img_i in range(nb_images):
        image = image_cube[img_i]
        if filter_sigma > 0:
            image = gaussian_filter(image,1.)
        image_f = image-np.median(image)
        
        x1, y1 = centroid_quadratic(image_f)
        
        planet_position[img_i,0] = x1 - len_x/2
        planet_position[img_i,2] = y1 - len_y/2
        if plot:
            plt.figure()
            plt.imshow(image_f,vmin=np.percentile(image_f,30),vmax=np.percentile(image_f,99))
            plt.scatter(x=planet_position[img_i,0] + len_x/2,y=planet_position[img_i,2] + len_x/2,color='r')
            plt.show()
    if save:
        relative_position = planet_position - planet_position[0,:]
        out_port = pp.core.dataio.OutputPort(save_fit_tag, data_storage_in=pipeline.m_data_storage)
        out_port.set_all(planet_position)
        out_port.close_port()
        
        out_port = pp.core.dataio.OutputPort(save_fit_tag + '_relative', data_storage_in=pipeline.m_data_storage)
        out_port.set_all(relative_position)
        out_port.close_port()
    else:
        return planet_position

def get_sky_calc_model(obj_coord='23 07 28.9014701064 +21 08 02.109792078',date='2023-10-15T03:25:30',wres=20000):
    
    coord_target = SkyCoord('%s:%s:%s %s:%s:%s' % tuple(obj_coord.split(' ')),unit=(u.hourangle,u.degree))
    
    ra=coord_target.ra.value
    dec=coord_target.dec.value
    
    skycalc = skycalc_ipy.SkyCalc()
    skycalc.get_almanac_data(ra=ra, dec=dec,
                             date=date,
                             update_values=True)
    skycalc["msolflux"] = 130       # [sfu] For dates after 2019-01-31
    skycalc['wres'],skycalc['wmin'],skycalc['wmax'] = wres,1500,3000
    tbl = skycalc.get_sky_spectrum()
    wvl = tbl['lam'].data/1e3
    transm = tbl['trans'].data
    flux = tbl['flux'].data
    return wvl,transm,flux

def overlap_2_arrays(wvl_ref,wvl_axis,dlambda_ref,dlambda_axis):
    # overlap is matrix
    # overlap[i,j] is the overlap between wvl_ref[i] and wvl_axis[j]
    wvl_ref1 = wvl_ref - dlambda_ref/2
    wvl_ref2 = wvl_ref + dlambda_ref/2
    wvl_axis1 = wvl_axis - dlambda_axis/2
    wvl_axis2 = wvl_axis + dlambda_axis/2

    WVLREF1,WVLAXIS1 = np.meshgrid(wvl_ref1,wvl_axis1)
    WVLREF2,WVLAXIS2 = np.meshgrid(wvl_ref2,wvl_axis2)

    x1 = np.max(np.dstack([WVLREF1[:,:,np.newaxis],WVLAXIS1[:,:,np.newaxis]]),axis=2)
    x2 = np.min(np.dstack([WVLREF2[:,:,np.newaxis],WVLAXIS2[:,:,np.newaxis]]),axis=2)
    
    #x1 = np.array([[np.max([wvl_ref1[i],wvl_axis1[j]]) for i in range(len(wvl_ref1))] for j in range(len(wvl_axis1))])
    #x2 = np.array([[np.min([wvl_ref2[i],wvl_axis2[j]]) for i in range(len(wvl_ref2))] for j in range(len(wvl_axis2))])

    overlap = ((x2 > x1) * (x2 - x1)).transpose()

    return overlap

def drizzle_2_arrays(wvl_ref,spectrum_ref,weights_ref,wvl_axis,spectrum_axis,weights_axis,drop_factor = 1):
    dlambda_ref = np.hstack([wvl_ref[1:] - wvl_ref[:-1],wvl_ref[-1] - wvl_ref[-2]])
    dlambda_axis = np.hstack([wvl_axis[1:] - wvl_axis[:-1],wvl_axis[-1] - wvl_axis[-2]]) * drop_factor
    
    overlap_matrix = overlap_2_arrays(wvl_ref=wvl_ref,wvl_axis=wvl_axis,dlambda_ref=dlambda_ref,dlambda_axis=dlambda_axis)

    weights_result = weights_ref  + np.dot(overlap_matrix, weights_axis)
    spectrum_result = np.where(weights_result != 0., (weights_ref * spectrum_ref + np.dot(overlap_matrix, weights_axis*spectrum_axis))/weights_result,0)

    return wvl_ref,spectrum_result,weights_result

def dodrizzle_n_arrays(wvlref,wvl_new,spectra_new,drop_factor=1,verbose=False):
    wvl_upsampled = wvlref
    spectrum_upsampled = np.zeros_like(wvlref)
    weights_upsampled = np.zeros_like(wvlref)
    
    n_spec = len(wvl_new)
    for spec_i in range(n_spec):
        if verbose:
            print('Progress %.2f' % (100*(spec_i+1)/n_spec) + ' %',end='\r')
        wavelengthi,spectrumi = wvl_new[spec_i],spectra_new[spec_i]
        weightsi = np.ones_like(wavelengthi)
        wvl_upsampled,spectrum_upsampled,weights_upsampled = drizzle_2_arrays(wvl_ref=wvl_upsampled,spectrum_ref=spectrum_upsampled,weights_ref=weights_upsampled,wvl_axis=wavelengthi,spectrum_axis=spectrumi,weights_axis=weightsi,drop_factor=drop_factor)
    return wvl_upsampled,spectrum_upsampled,weights_upsampled

def drizzle_stellar_spectrum(wavelength_ref,wavelength_axis,spectra_axis,factor_upsampling=2,drop_factor=1,verbose=False):
    n_spectra = len(wavelength_axis)
    len_wvl = len(wavelength_ref)
    
    # upsampling
    wvl_up = np.zeros((n_spectra,factor_upsampling*len_wvl))
    spectra_up = np.zeros((n_spectra,factor_upsampling*len_wvl))
    for i in range(n_spectra):
        spectra_up[i,:],wvl_up[i,:] = resample(spectra_axis[i,:], num=len_wvl*factor_upsampling, t=wavelength_axis[i,:], axis=0, window=None, domain='time')
    spectra_tmp,wvl_ref_up = resample(np.ones_like(wavelength_ref), num=len_wvl*factor_upsampling, t=wavelength_ref, axis=0, window=None, domain='time')
    
    mean_dwvl = np.mean(wvl_ref_up[1:]-wvl_ref_up[:-1])
    wvl_ref_up_large = np.arange(np.min(wvl_up)//mean_dwvl+1,np.max(wvl_up)//mean_dwvl)*mean_dwvl
    
    wvl_drizzled,spectra_drizzled,weights = dodrizzle_n_arrays(wvl_ref_up_large,wvl_up,spectra_up,drop_factor=drop_factor,verbose=verbose)
    
    return wvl_drizzled,spectra_drizzled,weights

def check_oh_correction(pipeline,sequence_files_reduced,sigma=3):
    # checks the wavelength calibration based on the fit of the OH emission lines and identifies outlier frames for which the calibration maybe didn't work
    # sequence_files_reduced: dict with keys equal to the different datasets, and values the tags under which the reduced data is saved in the pipeline
    # sigma: outlier detection
    
    oh_correction = {}
    time = {}
    for which in sequence_files_reduced.keys():
        files = pipeline.get_attribute(sequence_files_reduced[which],attr_name='FILES',static=False)
        OH_correction_arr = []
        time_arr = []
        for file_i in files:
            hdr = fits.getheader(file_i)
            OH_correction_arr += [hdr['ESO QC LAMBDA SHIFT PIXEL']]
            timestamp=datetime.fromisoformat(hdr['ARCFILE'][5:-5]).timestamp()
            time_arr += [timestamp]
        oh_correction[which] = np.array(OH_correction_arr,dtype=np.float64)
        time[which] = np.array(time_arr,dtype=np.float64)

    time_arr = np.hstack(list(time.values()))
    oh_corr_arr = np.hstack(list(oh_correction.values()))
    df = pd.DataFrame()
    df['Time'] = np.array(time_arr,dtype=np.float64)-np.min(time_arr)
    df['OH_corr'] = np.array(oh_corr_arr,dtype=np.float64)
    datasets = list(sequence_files_reduced.keys())
    for which_i,which in enumerate(datasets):
        len_files_before = np.sum([len(time[datasets[i]]) for i in range(which_i)])
        len_files = np.sum([len(time[datasets[i]]) for i in range(which_i+1)])-1
        df.loc[len_files_before:len_files,'dataset'] = which
        df.loc[len_files_before:len_files,'file_id'] = np.arange(len(time[which]))
    
    time_sorted,oh_sorted,dataset_sorted,file_id_sorted = df.sort_values(by='Time')[['Time','OH_corr','dataset','file_id']].values.transpose()
    # fit linear function
    pols = np.polyfit(np.array(time_sorted,dtype=np.float64),np.array(oh_sorted,dtype=np.float64),deg=1)
    p = np.poly1d(pols)
    y_fit = p(np.array(time_sorted,dtype=np.float64))
    # identify outliers
    diff = oh_sorted-y_fit
    std = np.std(diff)
    mean = np.mean(diff)
    mask_bad = np.abs(diff) > sigma*std
    
    plt.figure()
    plt.plot(time_sorted,y_fit)
    for which in sequence_files_reduced.keys():
        mask = df['dataset'] == which
        plt.scatter(x=df[mask]['Time'],y=df[mask]['OH_corr'],label=which)
    plt.errorbar(x=time_sorted[mask_bad],y=oh_sorted[mask_bad],color='r',xerr=0.5*(time_sorted[1]-time_sorted[0]),yerr=0.1)
    plt.plot(time_sorted,y_fit-sigma*std,color='k',ls=':')
    plt.plot(time_sorted,y_fit+sigma*std,color='k',ls=':')
    plt.legend()
    plt.show()
    
    remove_frames = {}
    for which in sequence_files_reduced.keys():
        indices_mask = dataset_sorted[mask_bad] == which
        remove_frames[which] = file_id_sorted[mask_bad][indices_mask]
    
    return remove_frames
def get_ticks(img,star_pos,pixscale,tick_sep=0.5,minor_tick_sep = 0.1):
    lenx=len(img)
    ticks_x = np.arange(lenx)
    ticks_y = np.arange(lenx)
    
    ticks_x = np.arange(lenx)
    sky_x = -(ticks_x-star_pos[0])*pixscale
    low_x = np.round(np.min(sky_x)/tick_sep)*tick_sep
    high_x = np.round(np.max(sky_x)/tick_sep)*tick_sep
    nb_ticks = int((high_x-low_x)/tick_sep)+1
    ticks_x_labels = np.linspace(low_x,high_x,nb_ticks)
    tick_x_pos = (-ticks_x_labels)/pixscale + star_pos[0]
    
    nb_ticks = int((high_x-low_x)/minor_tick_sep)+1
    ticks_x_labels_minor = np.linspace(low_x,high_x,nb_ticks)
    tick_x_pos_minor = (-ticks_x_labels_minor)/pixscale + star_pos[0]
    
    ticks_y = np.arange(lenx)
    sky_y = (ticks_y-star_pos[1])*pixscale
    low_y = np.round(np.min(sky_y)/tick_sep)*tick_sep
    high_y = np.round(np.max(sky_y)/tick_sep)*tick_sep
    nb_ticks = int((high_y-low_y)/tick_sep)+1
    ticks_y_labels = np.linspace(low_y,high_y,nb_ticks)
    tick_y_pos = (ticks_y_labels)/pixscale + star_pos[1]
    
    nb_ticks = int((high_y-low_y)/minor_tick_sep)+1
    ticks_y_labels_minor = np.linspace(low_y,high_y,nb_ticks)
    tick_y_pos_minor = (ticks_y_labels_minor)/pixscale + star_pos[1]
    return ticks_x_labels,tick_x_pos,ticks_x_labels_minor,tick_x_pos_minor,ticks_y_labels,tick_y_pos,ticks_y_labels_minor,tick_y_pos_minor