


dir_path_data = data_path + "/True_HCI_data"
ls_data = os.listdir(dir_path_data)

ls_planetFilename = []
for i in range(0, len(ls_data)):
    ls_planetFilename.append(ls_data[i][4:][:-5])

template_characteristics = {'Temp': 1700, 'Surf_grav': 4.1, 'H2O': 1, 'CO': 0}
dir_file_mol_template = data_path + 'csv_inputs/Molecular_Templates_df.csv'

MT_df = pd.read_csv(dir_file_mol_template, index_col=0)
template_planetHCI = MT_df[MT_df["tempP"] == template_characteristics['Temp']][MT_df["loggP"] == template_characteristics['Surf_grav']][MT_df["H2O"] == template_characteristics['H2O']][MT_df["CO"] == template_characteristics['CO']]


H2O = template_planetHCI #fits.open(subdir+'30_data/Data_Planets/BPic/spectrum_PSF_sub0.fits')[0].data
H2O_wv_HR = template_planetHCI.drop(['tempP', 'loggP', 'H2O', 'CO'], axis=1).columns.astype(float) #H2O[0,:]
H2O_abs_HR = template_planetHCI.drop(['tempP', 'loggP', 'H2O', 'CO'], axis=1).astype(float)  #H2O[1,:]


wv_data = np.loadtxt(subdir+'30_data/Data_Planets/BPic_meta/wavelength_range_2.088-2.451.txt')
wv_step = wv_data[1]-wv_data[0]
for k in range(150):
    wv_data = np.concatenate((np.array([wv_data[0]-wv_step]), wv_data, np.array([wv_data[-1]+wv_step])))
H2O_abs_LR = spectres(wv_data,np.array(H2O_wv_HR),np.array(H2O_abs_HR))

#wv_model_data_res = np.arange(H2O_wv[30], H2O_wv[-30], (wv_data[-1]-wv_data[0])/len(wv_data))
#H2O_data_res = spectres(wv_data,H2O_wv_HR,H2O_abs_HR)

H2O_fit_HR = np.polyfit(np.array(H2O_wv_HR), np.array(H2O_abs_HR).flatten(), 6)
H2O_cnt_HR = np.polyval(H2O_fit_HR, H2O_wv_HR)
H2O_fit_LR = np.polyfit(wv_data, H2O_abs_LR.flatten(), 6)
H2O_cnt_LR = np.polyval(H2O_fit_LR, wv_data)
#H2O_fit_data_res = np.polyfit(wv_model_data_res, H2O_data_res, 3)
#H2O_cnt_data_res = np.polyval(H2O_fit_data_res, wv_model_data_res)

H2O_residuals_HR = H2O_abs_HR - H2O_cnt_HR
H2O_residuals_LR = H2O_abs_LR - H2O_cnt_LR
#H2O_cnt_res_data_range = H2O_data_range-H2O_cnt_data_range
#H2O_cnt_res_data_res = H2O_data_res-H2O_cnt_data_res ######


fig, (ax,bx) = plt.subplots(2,1,figsize=(18,6))
ax.plot(H2O_wv_HR, np.array(H2O_abs_HR).flatten(), color='lightblue', alpha=0.5)
ax.plot(wv_data, H2O_abs_LR.flatten(), color='blue')
#ax.plot(wv_model_data_res, CO_data_res, color='darkviolet')
ax.plot(H2O_wv_HR, H2O_cnt_HR,color='r')
ax.set_xlim(2.,2.5)

bx.plot(np.array(H2O_wv_HR), np.array(H2O_residuals_HR).flatten(), color='lightblue', alpha=0.5)
bx.plot(wv_data,H2O_residuals_LR.flatten(), color='blue')
bx.set_xlim(2.,2.5)
fig.show()