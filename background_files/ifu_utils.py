import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter,medfilt
from scipy.interpolate import splrep, BSpline
import warnings
from astropy.modeling import models, fitting
from astropy.modeling.functional_models import AiryDisk2D,Gaussian2D
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.aperture import aperture_photometry
import copy
from spectres import spectres
from scipy.stats import median_abs_deviation

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

# plot IFU data either as the wvl-averaged cubes (single) or wvl- and time-averaged image (full)
def plot_data(pipeline,data_tag,wvl_tag,method,contour=True,scatter_data=None):
    data = pipeline.get_data(data_tag)
    wavelength = pipeline.get_data(wvl_tag)
    datacubes = select_cubes(data,wavelength)
    lenwvl = len(wavelength)
    lencube=len(datacubes)
    if not (scatter_data is None):
        if len(scatter_data) == lencube:
            scatter_data_reformat = scatter_data
        elif len(scatter_data) == lenwvl*lencube:
            scatter_data_reformat = np.mean(np.array(scatter_data).reshape((lencube,lenwvl,2)),axis=1)
        elif len(scatter_data) == 2:
            scatter_data_reformat = [scatter_data for i in range(lencube)]
        else:
            print('PROBLEM WITH FORMAT OF SCATTER DATA')
    if method == 'full':
        images = [np.mean(datacubes,axis=(0,1))]
        if not (scatter_data is None):
            scatter_data_reformat = [np.mean(scatter_data_reformat,axis=0)]
    elif method == 'single':
        images = np.mean(datacubes,axis=1)
        if not (scatter_data is None):
            scatter_data_reformat = scatter_data_reformat
    else:
        print('CHOOSE BETWEEN METHOD=FULL OR METHOD=SINGLE')
    for img_i in range(len(images)):
        img = images[img_i]
        plt.figure()
        if contour:
            plt.contour(img,cmap='RdBu')
        plt.imshow(img,vmin=np.percentile(img,5),vmax=np.percentile(img,95),origin='lower')
        if not (scatter_data is None):
            plt.scatter(x=scatter_data[img_i][0],y=scatter_data[img_i][1],color='r')
        plt.show()
    return datacubes
# extract the wavelength info from an ERIS dataset reduced with esoreflex
# outputs the wavelength axis, and the number of frames (i.e. nb of wavelength channels) within each 3D cube
def get_wavelength_info(science_files):
    science_nframes = []
    for file in science_files:
        hdu = fits.open(file)
        hdr = hdu[1].header
        wvl_params = [hdr['CRVAL3'],hdr['CDELT3']]
        data = hdu[1].data
        science_nframes += [len(data)]
        hdu.close()
    wavelength_axis = [wvl_params[0] + i*wvl_params[1] for i in range(science_nframes[0])]
    return wavelength_axis,science_nframes

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
def replace_outliers(spectrum, sigma, filt_size = 31):
    spectrum2 = copy.copy(spectrum)
    sub = medfilt(spectrum2,filt_size)
    sp_sub = spectrum2-sub
    # std = np.quantile(sp_sub,0.84)-np.quantile(sp_sub, 0.50)
    std = median_abs_deviation(sp_sub)
    spectrum_clean = np.where(np.abs(sp_sub)> sigma*std, sub, spectrum2)
    return spectrum_clean

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

# lambda over D, with wvl in micrometers, and D in meters. Outputs lambda/D in arcseconds
def lambda_D(wvl,D):
    return 180./np.pi*60*60*wvl*1e-6/D

# function to down-sample a spectrum to a lower spectral resolution.
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