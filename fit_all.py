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
from copy import deepcopy
import pickle


OII_rest    = lines_vac['OII']
NII_rest    = lines_vac['NII']
Hbeta_rest  = lines_vac['Hbeta']
Halpha_rest = lines_vac['Halpha']
SII_rest    = lines_vac['SII']
CaII_rest   = lines_vac['CaII']
NaD_rest    = lines_vac['NaD']
crop_region = [lines_vac['Halpha'][0]-30, lines_vac['Halpha'][0]+30]

spectra_data = fits.open('/Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits')
color_data = fits.open('/Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_with_Flux.fits')
spectra = Spectrum(spectra_data, color_data)

blue_crit = (spectra.color_criteria(criterion='g-z<1.25', exclude=False)) & (spectra.color_criteria(criterion='g-r<0.75', exclude=False))
qso_crit = spectra.subtype_criteria(subtype='QSO', exclude=True)
spectra.subset(criteria=blue_crit & qso_crit)
# spectra.shrink_dataset(20)
Fit = FitSpectrum(spectra)

Fit.shift_to_rest_frame()


# save_path = './fitted_results.pkl'
# if os.path.exists(save_path):
#     os.remove(save_path)


# for i in tqdm.tqdm(range(Fit.n_spectra)):
#     dz = Fit.adjust_z(id=Fit.targetID[i], mode='base_2')
#     Fit.shift_to_rest_frame(i=i)
    
#     params, status = Fit.search_NaD(id=spectra.targetID[i], two_component=True)
    
#     params_s, status_s = Fit.fit_flux(id=spectra.targetID[i], lam0=[*Halpha_rest, *NII_rest, *SII_rest], two_component=False, fit_z=False, e_or_a='e')
#     params_d, status_d = Fit.fit_flux(id=spectra.targetID[i], lam0=[*Halpha_rest, *NII_rest, *SII_rest], two_component=True, fit_z=False, e_or_a='e')
#     if status_d is True: 
#         chisq_d = Fit.calculate_chisq(spectra.targetID[i], params=params_d, region=crop_region)
#     else:
#         chisq_d = None
        
#     if status_s is True: 
#         chisq_s = Fit.calculate_chisq(spectra.targetID[i], params=params_s, region=crop_region)
#     else:
#         chisq_s = None
        
#     if status_s is True and status_d is True: 
#         f_test, p_value = Fit.calculate_f_test(spectra.targetID[i], params_s=params_s, params_d=params_d, region=crop_region)
#     else:
#         f_test, p_value = None, None

#     # store a record (avoids mutating the original params object and is easy to inspect/serialize)
#     record = {
#         'id': Fit.targetID[i],
#         'dz': dz,
#         'adjust_z_method': Fit.adjust_z_mode[i],
#         'NaD_status': status,
#         'NaD_params': deepcopy(params),   # keep a deep copy so later changes to Fit won't alter stored params
#         'NaD_type': Fit.searched_NaD[i],
#         'chisq_s': chisq_s,
#         'chisq_d': chisq_d,
#         'f_test': f_test,
#         'p_value': p_value,
#     }

#     # Optional: persist incrementally to disk (append a pickle entry per object)
#     with open(save_path, 'ab') as fout:
#         pickle.dump(record, fout)

# records = []
# if os.path.exists(save_path):
#     with open(save_path, 'rb') as fin:
#         while True:
#             try:
#                 records.append(pickle.load(fin))
#             except EOFError:
#                 break

# print(f"Loaded {len(records)} records")
# if records:
#     # example inspection
#     print("First record keys:", list(records[0].keys()))



save_path = './fitted_results_wo_z_adjust.pkl'
if os.path.exists(save_path):
    os.remove(save_path)


for i in tqdm.tqdm(range(Fit.n_spectra)):
    params, status = Fit.search_NaD(id=spectra.targetID[i], two_component=True)
    # store a record (avoids mutating the original params object and is easy to inspect/serialize)
    record = {
        'id': Fit.targetID[i],
        'dz': None,
        'adjust_z_method': None,
        'NaD_status': status,
        'NaD_params': deepcopy(params),   # keep a deep copy so later changes to Fit won't alter stored params
        'NaD_type': Fit.searched_NaD[i],
        'chisq_s': None,
        'chisq_d': None,
        'f_test': None,
        'p_value': None,
    }

    # Optional: persist incrementally to disk (append a pickle entry per object)
    with open(save_path, 'ab') as fout:
        pickle.dump(record, fout)

records = []
if os.path.exists(save_path):
    with open(save_path, 'rb') as fin:
        while True:
            try:
                records.append(pickle.load(fin))
            except EOFError:
                break

print(f"Loaded {len(records)} records")
if records:
    # example inspection
    print("First record keys:", list(records[0].keys()))