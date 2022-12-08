
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

import sys
#sys.path.append("/Users/Gabo/PynPoint/PynPoint")
#sys.path.append("/Users/Gabo/Tools/SpectRes-master/")
import os
from spectres import spectres
from pynpoint import Pypeline, FitsReadingModule, RemoveLinesModule, FitCenterModule, StackCubesModule, BadPixelSigmaFilterModule, ParangReadingModule, AddLinesModule, FitsWritingModule, TextWritingModule
from data_reduction_utils.ifuframeselection import SelectWavelengthRangeModule
from data_reduction_utils.ifubadpixel import NanFilterModule
from data_reduction_utils.ifucentering import IFUAlignCubesModule
from data_reduction_utils.ifupsfpreparation import IFUStellarSpectrumModule
from data_reduction_utils.ifupsfsubtraction import IFUPSFSubtractionModule
from data_reduction_utils.ifuresizing import FoldingModule
from data_reduction_utils.ifustacksubset import CrossCorrelationPreparationModule
from data_reduction_utils.ifucrosscorrelation import CrossCorrelationModule
from Mol_Map.config import path_init
from data_reduction_utils.ifupcasubtraction import IFUResidualsPCAModule
from data_reduction_utils.ifuresizing import UnfoldingModule
from Mol_Map.DataPreprocessing_utils import importWavelength_asList, image_deconstruct, image_reconstruct
import time
import concurrent.futures
import numpy as np
import pandas as pd
from photutils import CircularAperture
import matplotlib.pyplot as plt
from astropy.io import fits
import os
from Mol_Map.crosscorrNormVec import crosscorrRV_vec
from Mol_Map.config import path_init

## ACTIVE SUBDIR
subdir = path_init()
#subdir = "C:/Users/emily/Documents/ML_spectroscopy_thesis/"

# PATHS
code_path = subdir + "50_code/"
data_path = subdir + "30_data/DataSets/" #"C:/Users/emily/Documents/Mol_Map_project/30_data/DataSets/"
plot_path = subdir + "60_plots/"
plots_path = subdir + "60_plots/"
data_ccf = data_path + "csv_inputs/True_CCF_data/"



#fig.savefig('/Users/Gabo/SINFONI/Beta_Pic/Results/Spectra_H2O.png')
# Define Directories
working_place_in =  subdir+"30_data/DataSets/working_place"
input_place_in = subdir+"30_data/DataSets/Reflex_output/Science_test" # "/Users/Gabo/SINFONI/Beta_Pic/Reflex_output/Science_test"
output_place_in =  subdir+"30_data/DataSets/results"

pipeline = Pypeline(working_place_in, input_place_in, output_place_in)

Import_Science = FitsReadingModule(name_in = "Import_Science",
                                   input_dir = input_place_in,
                                   image_tag = "initial_spectrum",
                                   check=True
                                   )


Select_range = SelectWavelengthRangeModule(range_f = (2.088, 2.452),
                                           name_in = "Select_range",
                                           image_in_tag = "initial_spectrum",
                                           image_out_tag = "spectrum_selected",
                                           wv_out_tag = "wavelength_range"
                                           )


Substitute_NaNs = NanFilterModule(name_in = "Substitute_NaNs",
                                       image_in_tag = "spectrum_selected",
                                       image_out_tag = "spectrum_NaN")

"""
Small_image = RemoveLinesModule(lines = (4,4,4,4),
                             name_in = "Enlarge_image",
                             image_in_tag = "spectrum_NaN",
                             image_out_tag = "spectrum_NaN_small")


Centering_all = FitCenterModule(name_in = "Centering_all",
                                image_in_tag = "spectrum_NaN_small",
                                method='full',
                                fit_out_tag='centering_all',
                                mask_radii = (None, 1.0))

Coadd_cubes = StackCubesModule(name_in= "Coadd_cubes",
                             image_in_tag = "spectrum_NaN_small",
                             image_out_tag = "coadded_cubes",
                             combine='median')

Centering_cubes = FitCenterModule(name_in = "Centering_cubes",
                                image_in_tag = "coadded_cubes",
                                method='full',
                                fit_out_tag='centering_cubes',
                                mask_radii = (None, 1.0))

Shift_no_center = IFUAlignCubesModule(precision=0.02,
                                      shift_all_in_tag= "centering_all",
                                       shift_cube_in_tag = "centering_cubes",
                                       interpolation="spline",
                                       name_in="shift_no_center",
                                       image_in_tag="spectrum_NaN_small",
                                       image_out_tag="cubes_aligned")

Centering_test = FitCenterModule(name_in = "Centering_test",
                                  image_in_tag = "cubes_aligned",
                                  method='full',
                                  fit_out_tag='centering_test',
                                  mask_radii = (None, 1.0),
                                  guess=(0.5,0.2,3.,3.,5000,0.,0.))

bp = BadPixelSigmaFilterModule(name_in='bp',
                               image_in_tag="cubes_aligned",
                               image_out_tag="cubes_bp",
                               map_out_tag=None,
                               box=9,
                               sigma=3.,
                               iterate=4)

star_master = IFUStellarSpectrumModule(name_in="star_master",
                                       image_in_tag="cubes_aligned",
                                       wv_in_tag = "wavelength_range",
                                       spectrum_out_tag="stellar_spectrum",
                                       num_pix = 10,
                                       std_max = 0.1)


master_sub = IFUPSFSubtractionModule(name_in = "master_sub",
                                     image_in_tag="cubes_aligned",
                                     #stellar_spectra_in_tag = "stellar_spectrum",
                                     image_out_tag = "PSF_sub",
                                     #gauss_sigma=10,
                                     sigma=2.,
                                     #iteration = 2
                                     )

parang = ParangReadingModule(file_name = 'parang_sort.txt',
                             name_in = "parang",
                             input_dir = '/home/ipa/quanz/user_accounts/egarvin/Mol_Map_project/30_data/DataSets/results',
                             data_tag = "PSF_sub")

Folding = FoldingModule(name_in="Folding",
                        image_in_tag="PSF_sub",
                        image_out_tag = "im_2D")

PCA = IFUResidualsPCAModule(pc_number = 3,
                            name_in="PCA",
                            image_in_tag="im_2D",
                            image_out_tag = "im_2D_PCA")

Unfolding = UnfoldingModule(name_in="Unfolding",
                        image_in_tag="im_2D",
                        image_out_tag = "3D_PCA")

Large_image = AddLinesModule(lines = (20,20,20,20),
                                name_in = "Large_image",
                                image_in_tag = "3D_PCA",
                                image_out_tag = "3D_PCA_large")

CC_prep = CrossCorrelationPreparationModule(name_in="CC_prep",
                                            image_in_tag="3D_PCA_large",
                                            shift_cubes_in_tag="centering_cubes",
                                            image_out_tag="CC_prep",
                                            mask_out_tag="mask",
                                            #data_mask_out_tag = "data_mask"
                                            )

CrossCorr = CrossCorrelationModule(name_in = "CrossCorr",
                                   RV = 2500,
                                   dRV = 10,
                                   data_wv_in_tag = "wavelength_range",
                                   model_wv = wv_data,
                                   model_abs = H2O_residuals_LR,
                                   image_in_tag = "data_mask",
                                   mask_in_tag = "mask",
                                   snr_map_out_tag = "snr",
                                   CC_cube_out_tag = "CC_cube"
                                   )"""

'''snr = CrossCorrSNRModule(position=(74.5,34.5),
                         aperture=0.02,
                         name_in="snr",
                         image_in_tag="CC_cube",
                         snr_out_tag="snr_CC")
'''

Write_output = FitsWritingModule(name_in = "Write_output",
                                 file_name = "CC_cube.fits",
                                 output_dir = output_place_in,
                                 data_tag = "CC_cube")

Write_output_text = TextWritingModule(name_in = "Write_output_text",
                                 file_name = "stellar_spectrum.txt",
                                 output_dir = output_place_in,
                                 data_tag = "stellar_spectrum")


pipeline.add_module(Import_Science)
pipeline.add_module(Select_range)
pipeline.add_module(Substitute_NaNs)
"""pipeline.add_module(Small_image)
pipeline.add_module(Centering_all)
pipeline.add_module(Coadd_cubes)
pipeline.add_module(Centering_cubes)
pipeline.add_module(Shift_no_center)
pipeline.add_module(Centering_test)
pipeline.add_module(bp)
pipeline.add_module(star_master)
pipeline.add_module(master_sub)
pipeline.add_module(parang)
pipeline.add_module(Folding)
pipeline.add_module(PCA)
pipeline.add_module(Unfolding)
pipeline.add_module(Large_image)
pipeline.add_module(CC_prep)
pipeline.add_module(CrossCorr)"""
'''pipeline.add_module(snr)'''
"""pipeline.add_module(Write_output)
pipeline.add_module(Write_output_text)"""


pipeline.run()
print (pipeline.get_attribute("initial_spectrum","PIXSCALE", True))
print (pipeline.get_shape("stellar_spectrum"))

'''snr_list = pipeline.get_data("snr_CC")

fig, ax = plt.subplots(1,1, figsize=(15,7))
ax.plot(np.linspace(-2500, 2510,501),snr_list)
ax.set_xlabel('RV')
ax.set_ylabel('CCF')
fig.savefig(output_place_in+'snr_CCF.png')

max_ind = np.argmax(snr_list)
print np.linspace(-2500, 2500,501)[max_ind]
noise = np.concatenate((snr_list[:max_ind-10], snr_list[max_ind+10:]))

snr_tot = (snr_list[max_ind]-np.mean(noise))/np.std(noise)

print snr_tot'''

'''
quit()
python
execfile('Total_reduction.py')
'''
