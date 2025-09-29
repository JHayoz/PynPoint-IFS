"""
Preprocessing pipeline
"""
import pynpoint as pp
from astropy.io import fits
import numpy as np

from pynpoint_ifs.ifuframeselection import *
from pynpoint_ifs.ifubadpixel import *
from pynpoint_ifs.ifucentering import *
from pynpoint_ifs.ifupsfpreparation import *
from pynpoint_ifs.ifupsfsubtraction import *
from pynpoint_ifs.ifupsfsubtraction import *
from pynpoint_ifs.ifuresizing import *
from pynpoint_ifs.ifustacksubset import *
from pynpoint_ifs.ifucrosscorrelation import *
from pynpoint_ifs.ifucrosscorrelation import *
from pynpoint_ifs.ifupcasubtraction import *
from pynpoint_ifs.ifuresizing import *
from pynpoint_ifs.ifu_utils import select_cubes,plot_data,sort_files,get_wavelength_info,piecewise_spline_fit,replace_outliers,load_spectral_templates,get_sky_calc_model,rebin

# automatically run the whole preprocessing pipeline
def run_preprocessing(pipeline, sequence_files, run_which = 'all', hdr_wvl_keywords = ['CRVAL3','CD3_3'],crop=4,outlier_sigma=20,bad_pixel_corr = True,bp_sigma=5.,bp_iterate=10,bp_box=9,start='reading'):
    possible_starts = ['reading','wvlsel','nan','crop','bpclean','outlier']
    if not start in possible_starts:
        raise RuntimeError(
                            f"The 'start' keyword must be one of {possible_starts}."
                        )
    if run_which == 'all':
        sequence = sequence_files.keys()
    else:
        sequence = [run_which]
    if start in possible_starts[:1]:
        for which in sequence:
            module = pp.FitsReadingModule(
                name_in = 'read_raw_' + which,
                filenames = sequence_files[which],
                image_tag = 'raw_' + which,
                check=False)
            pipeline.add_module(module)
            pipeline.run_module('read_raw_' + which)
    if start in possible_starts[:2]:
        for which in sequence:
            module = AutomaticallySelectWavelengthRangeModule(
                name_in = 'select_range_' + which,
                image_in_tag = 'raw_' + which,
                image_out_tag = 'raw_' + which + '_wvlsel',
                wv_out_tag = 'wavelength_' + which,
                header_crval = hdr_wvl_keywords[0],
                header_cd = hdr_wvl_keywords[1]
            )
            pipeline.add_module(module)
            pipeline.run_module('select_range_' + which)
        """
        # Old code
        science_nframes = {}
        wvl_params = {}
        wavelength_axis = {}
        for which in sequence:
            science_nframes[which] = []
            wvl_params[which] = []
            wavelength_axis[which] = []
            for file in sequence_files[which]:
                hdu = fits.open(file)
                hdr = hdu[1].header
                wvl_params[which] += [[hdr[hdr_wvl_keywords[0]],hdr[hdr_wvl_keywords[1]]]]
                data = hdu[1].data
                science_nframes[which] += [len(data)]
                hdu.close()
                
                wavelength_axis[which] = [
                    np.array([
                        wvl_params[which][j][0] + i*wvl_params[which][j][1] for i in range(science_nframes[which][j])
                    ]) for j in range(len(science_nframes[which]))
                ]
            pipeline.set_attribute(data_tag = 'raw_' + which,attr_name='NFRAMES',attr_value = np.array(science_nframes[which]),static=False)
        
        for which in sequence:
            module = SelectWavelengthRangeModule(
                name_in = 'select_range_' + which,
                image_in_tag = 'raw_' + which,
                image_out_tag = 'raw_' + which + '_wvlsel',
                wv_out_tag = 'wavelength_' + which,
                range_i = (wavelength_axis[which][0][0],wavelength_axis[which][0][-1]),
                range_f = (2.19696, 2.44149))
            pipeline.add_module(module)
            pipeline.run_module('select_range_' + which)
        """
    if start in possible_starts[:3]:
        for which in sequence:
            module = NanFilterModule(
                name_in = 'substitute_nans_' + which,
                image_in_tag = 'raw_'+ which + '_wvlsel',
                image_out_tag = 'raw_'+ which + '_wvlsel_nancorr',
                local=True,
                local_size = 5
                )
            pipeline.add_module(module)
            pipeline.run_module('substitute_nans_' + which)
    if start in possible_starts[:4]:
        for which in sequence:
            module = pp.RemoveLinesModule(
                lines = (crop,crop,crop,crop),
                name_in = 'crop_image_' + which,
                image_in_tag = 'raw_'+ which + '_wvlsel_nancorr',
                image_out_tag = 'raw_'+ which + '_wvlsel_nancorr_crop')
            
            pipeline.add_module(module)
            pipeline.run_module('crop_image_' + which)
    name_extension = '_wvlsel_nancorr_crop'
    if start in possible_starts[:5]:
        if bad_pixel_corr:
            for which in sequence:
                module = pp.BadPixelSigmaFilterModule(
                    name_in='bp_clean_' + which,
                    image_in_tag='raw_'+ which + '_wvlsel_nancorr_crop',
                    image_out_tag='raw_'+ which + '_wvlsel_nancorr_crop_bpclean',
                    map_out_tag=None,
                    box=bp_box,
                    sigma=bp_sigma,
                    iterate=bp_iterate)
                pipeline.add_module(module)
                pipeline.run_module('bp_clean_' + which)
    if bad_pixel_corr:
        name_extension += '_bpclean'
    if start in possible_starts[:6]:
        for which in sequence:
            module = OutlierCorrectionModule(
                name_in='outlier_clean_' + which,
                image_in_tag='raw_'+ which + name_extension,
                image_out_tag='raw_'+ which + name_extension + '_outlier',
                outlier_sigma = outlier_sigma,
                filter_sigma = 1.5,
                replace_method = 'smooth',
                cpu = 1)
            pipeline.add_module(module)
            pipeline.run_module('outlier_clean_' + which)

def run_preprocessing_order(
    pipeline, 
    sequence_files, 
    sequence_pipeline = ['reading','wvlsel','nan','crop','bpclean','outlier'], 
    run_which = 'all', 
    hdr_wvl_keywords = ['CRVAL3','CD3_3'],
    crop=4,
    outlier_sigma=20,
    bp_sigma=5.,
    bp_iterate=10,
    bp_box=9,
    start='reading'
):
    possible_starts = ['reading','wvlsel','nan','crop','bpclean','outlier']
    if not start in possible_starts:
        raise RuntimeError(
                            f"The 'start' keyword must be one of {possible_starts}."
                        )
    if run_which == 'all':
        sequence = sequence_files.keys()
    else:
        sequence = [run_which]
    
    pipeline_started = False
    file_name_extension = {
        'reading':'raw',
        'wvlsel':'wvlsel',
        'nan':'nancorr',
        'crop':'crop',
        'bpclean':'bpclean',
        'outlier':'outlier'
    }
    file_name = {which:'' for which in sequence}
    for seq_i,seq in enumerate(sequence_pipeline):
        if seq == start:
            pipeline_started = True
        
        if seq == 'reading':
            for which in sequence:
                
                addon = file_name_extension[seq] + '_' + which
                
                module = pp.FitsReadingModule(
                    name_in = 'read_raw_' + which,
                    filenames = sequence_files[which],
                    image_tag = file_name[which] + addon,
                    check=False
                )
                pipeline.add_module(module)
                pipeline.run_module('read_raw_' + which)
                
                file_name[which] += addon
                
                # add PARANG attribute in case it's missing
                attributes = pipeline.list_attributes(file_name[which])
                if 'PARANG' not in attributes.keys():
                    if 'PARANG_START' in attributes.keys() and 'PARANG_END' in attributes.keys():
                        parang = 0.5*(np.array(attributes['PARANG_START']) + np.array(attributes['PARANG_END']))
                    else:
                        parang = np.zeros((len(attributes['NFRAMES'])))
                    pipeline.set_attribute(data_tag = file_name[which],attr_name='PARANG',attr_value = parang,static=False)
        
        if seq == 'wvlsel':
            for which in sequence:
                
                addon = '_' + file_name_extension[seq]
                
                module = AutomaticallySelectWavelengthRangeModule(
                    name_in = 'select_range_' + which,
                    image_in_tag = file_name[which],
                    image_out_tag = file_name[which] + addon,
                    wv_out_tag = 'wavelength_' + which,
                    header_crval = hdr_wvl_keywords[0],
                    header_cd = hdr_wvl_keywords[1]
                )
                pipeline.add_module(module)
                pipeline.run_module('select_range_' + which)
                
                file_name[which] += addon
        
        if seq == 'nan':
            for which in sequence:
                
                addon = '_' + file_name_extension[seq]
                
                module = NanFilterModule(
                    name_in = 'substitute_nans_' + which,
                    image_in_tag = file_name[which],
                    image_out_tag = file_name[which] + addon,
                    local=True,
                    local_size = 5
                )
                pipeline.add_module(module)
                pipeline.run_module('substitute_nans_' + which)
                
                file_name[which] += addon
        
        if seq == 'crop':
            for which in sequence:
                
                addon = '_' + file_name_extension[seq]
                
                module = pp.RemoveLinesModule(
                    lines = (crop,crop,crop,crop),
                    name_in = 'crop_image_' + which,
                    image_in_tag = file_name[which],
                    image_out_tag = file_name[which] + addon,
                )
                pipeline.add_module(module)
                pipeline.run_module('crop_image_' + which)
                
                file_name[which] += addon
        
        if seq == 'bpclean':
            for which in sequence:
                
                addon = '_' + file_name_extension[seq]
                
                module = pp.BadPixelSigmaFilterModule(
                    name_in='bp_clean_' + which,
                    image_in_tag = file_name[which],
                    image_out_tag = file_name[which] + addon,
                    map_out_tag=None,
                    box=bp_box,
                    sigma=bp_sigma,
                    iterate=bp_iterate
                )
                pipeline.add_module(module)
                pipeline.run_module('bp_clean_' + which)
                
                file_name[which] += addon
        
        if seq == 'outlier':
            for which in sequence:
                
                addon = '_' + file_name_extension[seq]
                
                module = OutlierCorrectionModule(
                    name_in='outlier_clean_' + which,
                    image_in_tag = file_name[which],
                    image_out_tag = file_name[which] + addon,
                    outlier_sigma = outlier_sigma,
                    filter_sigma = 1.5,
                    replace_method = 'smooth',
                    cpu = 1
                )
                pipeline.add_module(module)
                pipeline.run_module('outlier_clean_' + which)
                
                file_name[which] += addon
    print('FINAL DATA TAGS:')
    for which in file_name.keys():
        print(which, file_name[which])