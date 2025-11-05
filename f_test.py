from astropy.io import fits
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import os
import sys
from pathlib import Path
from useful_functions import *

# OII_rest    = lines_vac['OII']
# NII_rest    = lines_vac['NII']
# Hbeta_rest  = lines_vac['Hbeta']
# Halpha_rest = lines_vac['Halpha']
# SII_rest    = lines_vac['SII']
# CaII_rest   = lines_vac['CaII']
# NaD_rest    = lines_vac['NaD']


# spectra_data = fits.open('/Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits')
# color_data = fits.open('/Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_with_Flux.fits')
# spectra = Spectrum(spectra_data, color_data)

# blue_crit = (spectra.color_criteria(criterion='g-z<1.25', exclude=False)) | (spectra.color_criteria(criterion='g-r<0.75', exclude=False))
# qso_crit = spectra.subtype_criteria(subtype='QSO', exclude=True)
# spectra.subset(criteria=blue_crit & qso_crit)
# # spectra.shrink_dataset(100)

# Fit = FitSpectrum(spectra)
# Fit.shift_to_rest_frame()

# targetid = []
# p_values = []
# for i, id in tqdm.tqdm(enumerate(Fit.targetID)):
#     targetid.append(id)
#     i = Fit.id2index(id)
#     try:
#         params_s, status_s = Fit.fit_flux(id=id, lam0=[*Halpha_rest, *NII_rest, *SII_rest], two_component=False, fit_z=True)
#         params_d, status_d = Fit.fit_flux(id=id, lam0=[*Halpha_rest, *NII_rest, *SII_rest], two_component=True, fit_z=True)
#         fitted_model_s = model(Fit.data_stack[i, 0, :], gaussian_parms=params_s['gaussian_params'], conti_parms=params_s['conti_params'])
#         fitted_model_d = model(Fit.data_stack[i, 0, :], gaussian_parms=params_d['gaussian_params'], conti_parms=params_d['conti_params'])
#         crop_region = [lines_vac['Halpha'][0]-30, lines_vac['Halpha'][0]+30]
        
#         f_stats, p_value = Fit.calculate_f_test(id, params_s=params_s, params_d=params_d, region=crop_region)
#         p_values.append(p_value)
#     except:
#         p_values.append(np.nan)

# np.savez('./p_value.npz', TargetId=targetid, p_values=p_values)

p_values = np.load('./p_value.npz')['p_values']

plt.hist(p_values)
plt.show()