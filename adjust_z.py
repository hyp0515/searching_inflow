from astropy.io import fits
import numpy as np
from useful_functions import *
import matplotlib.pyplot as plt
import tqdm

spectra_data = fits.open('/Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits')
color_data = fits.open('/Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_with_Flux.fits')
spectra = Spectrum(spectra_data, color_data)




blue_crit = (spectra.color_criteria(criterion='g-z<1.25', exclude=False)) | (spectra.color_criteria(criterion='g-r<0.75', exclude=False))
# blue_crit = (spectra.color_criteria(criterion='g-r<0.75', exclude=False))
qso_crit = spectra.subtype_criteria(subtype='QSO', exclude=True)

crit = blue_crit & qso_crit
# crit = qso_crit

spectra.subset(criteria=crit)


spectra.shrink_dataset(10)
# # print(f"Number of spectra after criteria: {spectra.n_spectra}")

Fit = FitSpectrum(spectra)
# dz_arr  = np.zeros(spectra.n_spectra)
# id_arr  = np.zeros(spectra.n_spectra)
# print("Adjusting redshifts...")
# for i, target_id in tqdm.tqdm(enumerate(Fit.targetID)):
#     dz = Fit.adjust_z(id=target_id, mode='base_2')
#     dz_arr[i] = dz
#     id_arr[i] = target_id
# fit_method = Fit.adjust_z_mode

# np.savez('./adjusted_z_results.npz', delta_z=dz_arr, obj_id=id_arr, fit_method=fit_method)

# # Fit.read_adjust_z_results(fname='./adjusted_z_results.npz')
# status = []
# params_1 = []
# params_2 = []
# id_1 = []
# id_2 = []
# line_types_1 = []
# line_types_2 = []
# print("Searching NaD lines...")
# for i, target_id in tqdm.tqdm(enumerate(Fit.targetID)):
#     popt, s = Fit.search_NaD(id=target_id)
#     line_type = Fit.searched_NaD[i]
#     if (s is True) and (len(popt) != 9):
#         params_1.append(popt)
#         id_1.append(target_id)
#         line_types_1.append(line_type)
#     elif (s is True) and (len(popt) == 9):
#         params_2.append(popt)
#         id_2.append(target_id)
#         line_types_2.append(line_type)
        




# np.savez('./NaD_fit_results.npz', params_1=params_1, id_1=id_1, line_types_1=line_types_1,
#          params_2=params_2, id_2=id_2, line_types_2=line_types_2)


dz = np.load('./adjusted_z_results.npz')['delta_z']
id_arr = np.load('./adjusted_z_results.npz')['obj_id']
fit_method = np.load('./adjusted_z_results.npz')['fit_method']

i = np.searchsorted(id_arr, 39627806283404930)
print(f"Delta z for {id_arr[i]}: {dz[i]}, fit method: {fit_method[i]}")

z = Fit.z_pipe[Fit.id2index(39627806283404930)] + dz[i]

params_1 = np.load('./NaD_fit_results.npz', allow_pickle=True)['params_1']
id_1 = np.load('./NaD_fit_results.npz', allow_pickle=True)['id_1']
line_types_1 = np.load('./NaD_fit_results.npz', allow_pickle=True)['line_types_1']

popt = params_1[np.searchsorted(id_1, 39627806283404930)]

sigma_1, offset, conti_a, conti_b = popt[0], popt[-3], popt[-2], popt[-1]
amp_D1, amp_D2 = popt[1], popt[2]

offset_comp = [(amp_D1, (lines_vac['NaD'][0]+offset)*(1+z), sigma_1), (amp_D2, (lines_vac['NaD'][1]+offset)*(1+z), sigma_1)]
fitted_model = model(desi_wavelength, gaussian_parms=offset_comp, conti_parms=(conti_a, conti_b))

region = (np.min(lines_vac['NaD'])*(1+z)- 50, np.max(lines_vac['NaD'])*(1+z) + 50)

plt.plot(desi_wavelength, fitted_model, color='red', label='Fitted Model')
plt.step(desi_wavelength, Fit.coadd_data[Fit.id2index(39627806283404930)], where='mid', color='blue', label='Spectrum')
plt.xlim(region)
plt.show()