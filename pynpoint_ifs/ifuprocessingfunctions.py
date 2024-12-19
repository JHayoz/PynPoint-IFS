"""
Help functions for ifuprocessing
"""

import copy

import numpy as np
from sklearn.decomposition import PCA

from pynpoint.util.image import shift_image, rotate


def do_PCA_sub(cube,pca_number):
    len_wvl,len_x,len_y = np.shape(cube)
    # put time at the back, then collapse along x and y
    pix_list_all = np.transpose(cube,axes=(1,2,0)).reshape((-1,len_wvl)) 
    
    mask_nans = np.sum(np.isnan(pix_list_all),axis=1) > 0
    pix_list = pix_list_all[~mask_nans]
    
    # subtract the mean value first
    spec_mean = np.mean(pix_list,axis=0)
    pix_list_res = pix_list - spec_mean
    
    # calculate the PCA once, but truncate to the different number of components
    pca_sklearn = PCA(n_components=pca_number, svd_solver="arpack",whiten=True)
    
    # fitting PCA
    pca_sklearn.fit(pix_list_res)
    # project unto basis
    pca_representation = pca_sklearn.transform(pix_list_res)
    # project back unto data
    model_psf_1d = pca_sklearn.inverse_transform(pca_representation)
    
    model_psf_1d_all = np.zeros_like(pix_list_all)
    model_psf_1d_all[~mask_nans] = model_psf_1d+spec_mean
    # separate again between x and y, and put time at front
    model_psf = np.transpose((model_psf_1d_all).reshape((len_x,len_y,len_wvl)),axes=(2,0,1)) 
    residuals = cube - model_psf
    
    return residuals,model_psf

def do_derotate_shift(cube,shift,shift_vector,derotate,derotate_angle):
    len_wvl,len_x,len_y = np.shape(cube)
    mask_arr = np.where(cube[0]==np.nan, False, True)
    
    if shift:
        mask_arr_shift = shift_image(mask_arr, (-shift_vector[0], -shift_vector[1]), "spline")
    else:
        mask_arr_shift = copy.copy(mask_arr)
    
    if derotate:
        mask_arr_rot = rotate(mask_arr_shift, -derotate_angle, reshape=False)
    else:
        mask_arr_rot = copy.copy(mask_arr_shift)
    
    cube_shift = np.zeros_like(cube)
    cube_rot = np.zeros_like(cube)
    
    for k in range(len_wvl):
        if shift:
            cube_shift[k] = shift_image(cube[k], (-shift_vector[0], -shift_vector[1]), "spline")
        else:
            cube_shift[k] = copy.copy(cube[k])
        
        if derotate:
            cube_rot[k] = rotate(cube_shift[k], -derotate_angle, reshape=False)
        else:
            cube_rot[k] = copy.copy(cube_shift[k])
    
    return cube_rot,mask_arr_rot