
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append("/net/ipa-gate/export/ipa/quanz/user_accounts/egarvin/IFS_pipeline/")
#sys.path.append("/Users/Gabo/Tools/SpectRes-master/")
from spectres import spectres

from pynpoint import *

H2O = fits.open('/net/ipa-gate/export/ipa/quanz/user_accounts/egarvin/IFS_pipeline/30_data/spectral_templates/h2o_clean_norm.fits')[0].data
H2O_wv_HR = H2O[0,:]
H2O_abs_HR = H2O[1,:]

wv_data = np.loadtxt('/net/ipa-gate/export/ipa/quanz/user_accounts/egarvin/IFS_pipeline/30_data/betapic/sinfoni_Kband/wavelength_range.txt')
wv_step = wv_data[1]-wv_data[0]
for k in range(150):
    wv_data = np.concatenate((np.array([wv_data[0]-wv_step]), wv_data, np.array([wv_data[-1]+wv_step])))
H2O_abs_LR = spectres(wv_data,H2O_wv_HR,H2O_abs_HR)

#wv_model_data_res = np.arange(H2O_wv[30], H2O_wv[-30], (wv_data[-1]-wv_data[0])/len(wv_data))
#H2O_data_res = spectres(wv_data,H2O_wv_HR,H2O_abs_HR)

H2O_fit_HR = np.polyfit(H2O_wv_HR, H2O_abs_HR, 6)
H2O_cnt_HR = np.polyval(H2O_fit_HR, H2O_wv_HR)
H2O_fit_LR = np.polyfit(wv_data, H2O_abs_LR, 6)
H2O_cnt_LR = np.polyval(H2O_fit_LR, wv_data)
#H2O_fit_data_res = np.polyfit(wv_model_data_res, H2O_data_res, 3)
#H2O_cnt_data_res = np.polyval(H2O_fit_data_res, wv_model_data_res)

H2O_residuals_HR = H2O_abs_HR - H2O_cnt_HR
H2O_residuals_LR = H2O_abs_LR - H2O_cnt_LR
#H2O_cnt_res_data_range = H2O_data_range-H2O_cnt_data_range
#H2O_cnt_res_data_res = H2O_data_res-H2O_cnt_data_res ######


fig, (ax,bx) = plt.subplots(2,1,figsize=(18,6))
ax.plot(H2O_wv_HR, H2O_abs_HR, color='lightblue', alpha=0.5)
ax.plot(wv_data, H2O_abs_LR, color='blue')
#ax.plot(wv_model_data_res, CO_data_res, color='darkviolet')
ax.plot(H2O_wv_HR, H2O_cnt_HR,color='r')
ax.set_xlim(2.,2.5)

bx.plot(H2O_wv_HR, H2O_residuals_HR, color='lightblue', alpha=0.5)
bx.plot(wv_data,H2O_residuals_LR, color='blue')
bx.set_xlim(2.,2.5)
fig.savefig('/net/ipa-gate/export/ipa/quanz/user_accounts/egarvin/IFS_pipeline/30_data/betapic/sinfoni_Kband/plots_scratch/Spectra_H2O.png')
