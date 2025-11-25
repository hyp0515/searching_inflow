import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import f
import requests
from pathlib import Path
import warnings
from multiprocessing import Pool, cpu_count
import functools
import tqdm
from typing import Dict, List, Tuple, Optional


c = const.c.cgs.value * 1e-4  # speed of light in km/s
desi_wavelength = np.arange(3600, 9824 + .8, .8) # DESI's observe wavelength

# https://astronomy.nmsu.edu/drewski/tableofemissionlines.html
lines_air = {
    'Halpha': [6562.819],
    'Hbeta': [4861.333],
    'OIII': [4958.911, 5006.843],
    'OII': [3726.032, 3728.815],
    'SII': [6716.440, 6730.810],
    'NII': [6548.050, 6583.460],
    'CaII': [8498.020, 8542.090, 8662.140],
    'NaD': [5890.004, 5895.985],
    'HeI': [5875.624]
}

def air2vac(wave):
    """
    Convert air wavelengths to vacuum wavelengths using the formula from
    Ciddor (1996) and Morton's (1991) values for standard air.

    Parameters:
    wave : array-like
        Wavelengths in air (in Angstroms).

    Returns:
    array-like
        Wavelengths in vacuum (in Angstroms).
    """
    wave = np.asarray(wave)
    sigma2 = (1e4 / wave) ** 2  # Convert to microns^-2
    factor = 1 + 0.0000834254 + (0.02406147 / (130 - sigma2)) + (0.00015998 / (38.9 - sigma2))
    return wave * factor

lines_vac = {key: air2vac(np.array(value)) for key, value in lines_air.items()}

OII_rest    = lines_vac['OII']
NII_rest    = lines_vac['NII']
Hbeta_rest  = lines_vac['Hbeta']
Halpha_rest = lines_vac['Halpha']
SII_rest    = lines_vac['SII']
CaII_rest   = lines_vac['CaII']
NaD_rest    = lines_vac['NaD']

def vac2air(wave):
    """
    Convert vacuum wavelengths to air wavelengths using the formula from
    Ciddor (1996) and Morton's (1991) values for standard air.

    Parameters:
    wave : array-like
        Wavelengths in vacuum (in Angstroms).

    Returns:
    array-like
        Wavelengths in air (in Angstroms).
    """
    wave = np.asarray(wave)
    sigma2 = (1e4 / wave) ** 2  # Convert to microns^-2
    factor = 1 + 0.0000834254 + (0.02406147 / (130 - sigma2)) + (0.00015998 / (38.9 - sigma2))
    return wave / factor

def lam2vel(lam, lam0, z):
    """
    Convert wavelength to velocity in km/s.

    Parameters:
    lam : array-like
        Observed wavelength (in Angstroms).
    lam0 : float
        Rest-frame wavelength (in Angstroms).
    z : float
        Redshift.

    Returns:
    array-like
        Velocity in km/s.
    """
    resting_lam = lam / (1 + z)
    return c * (resting_lam - lam0) / lam0

def smooth_spectrum(flux, sigma):
    """
    Apply Gaussian smoothing to the input flux array.

    Parameters:
        flux (array): The input flux values.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        array: Smoothed flux values.
    """
    return gaussian_filter1d(flux, sigma)

def gaussian(lam, amp, lam0, sigma):
    """
    Generate a Gaussian profile.
    """
    return amp * np.exp(-0.5 * ((lam - lam0) / sigma) ** 2)

def model(lam, gaussian_parms=None, conti_parms=(0, 0)):
    
    flux = 0

    if gaussian_parms is not None:
        for p in gaussian_parms:
            amp, lam0, sigma = p
            gauss = gaussian(lam, amp, lam0, sigma)
            flux += gauss

    conti_a, conti_b = conti_parms
    conti = conti_a * lam + conti_b

    return flux + conti

def image_link(RA, DEC, save_image=False, fname=None):
    if save_image:
        side_arcmin = 0.5
        scale = np.round(side_arcmin / 3, 6)
        url = f'https://www.legacysurvey.org/viewer/cutout.jpg?ra={RA}&dec={DEC}&pixscale={scale}&layer=ls-dr10-grz&size=200'
        with requests.get(url, stream=True, timeout=(10, 30)) as r:
            r.raise_for_status()
            with open(fname if fname else "cutout.jpg", "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        print(f"Saved {fname if fname else 'cutout.jpg'}")
    return f'https://www.legacysurvey.org/viewer?ra={RA}&dec={DEC}&layer=ls-dr10-grz&zoom=14'

def spectrum_link(targetID):
    return f'https://www.legacysurvey.org/viewer/desi-spectrum/dr1/targetid{targetID}'


################################################################################################################
class Spectrum:

    def __init__(self, spectra_data, color_data):
        
        sd1 = spectra_data[1].data
        sd2 = spectra_data[2].data
        sd3 = spectra_data[3].data
        sd4 = spectra_data[4].data
        
        cd1 = color_data[1].data

        targetID_spectra = sd1['TARGETID']
        targetID_color   = cd1['TARGETID']

        # robust + fast alignment
        order = np.argsort(targetID_color)
        pos = np.searchsorted(targetID_color[order], targetID_spectra)
        rearranged_indices = order[pos]

        self.targetID = np.asarray(targetID_spectra)
        self.z_pipe   = np.asarray(sd1['Z'])
        self.z        = self.z_pipe.copy()
        self.RA       = np.asarray(sd1['RA'])
        self.DEC      = np.asarray(sd1['DEC'])

        self.coadd_data = np.asarray(sd2[:, 0, :], dtype=np.float32)
        self.ivar       = np.asarray(sd3[:, 0, :], dtype=np.float32)
        self.mask       = np.asarray(sd4[:, 0, :], dtype=np.float32)

        self.spectype      = color_data[1].data['SPECTYPE'][rearranged_indices]
        
        g = cd1['FLUX_G']; r = cd1['FLUX_R']; z = cd1['FLUX_Z']
        w1 = cd1['FLUX_W1']; w2 = cd1['FLUX_W2']
        color_flux = np.column_stack((g, r, z, w1, w2))
        with np.errstate(divide='ignore', invalid='ignore'):
            color_mag_all = 22.5 - 2.5 * np.log10(color_flux)
        self.color_mag = color_mag_all[rearranged_indices]
        
        
        self.n_spectra = len(self.targetID)
        self.adjust_z_mode = None   # make these lazy if large & rarely used
        self.searched_NaD  = None
        self.target_label  = None
        self._id_to_idx = {int(tid): i for i, tid in enumerate(self.targetID)}
    
    def add_attribute(self, name, value):
        setattr(self, name, value)
        
    def del_attribute(self, name):
        if hasattr(self, name):
            delattr(self, name)
    
    def _apply_index(self, idx):
        # idx may be a boolean mask or integer array
        self.targetID   = self.targetID[idx]
        self.z_pipe     = self.z_pipe[idx]
        self.z          = self.z[idx]
        self.RA         = self.RA[idx]
        self.DEC        = self.DEC[idx]
        self.coadd_data = self.coadd_data[idx]
        self.ivar       = self.ivar[idx]
        self.mask       = self.mask[idx]
        if hasattr(self, "color_mag"):      self.color_mag = self.color_mag[idx]
        if hasattr(self, "smoothed_flux"):  self.smoothed_flux = self.smoothed_flux[idx]
        if hasattr(self, "target_label") and self.target_label is not None:
            if isinstance(idx, np.ndarray) and idx.dtype == bool:
                self.target_label = [lbl for lbl, keep in zip(self.target_label, idx) if keep]
            else:
                self.target_label = [self.target_label[i] for i in np.atleast_1d(idx)]

        self.n_spectra = len(self.targetID)
        self._id_to_idx = {int(tid): i for i, tid in enumerate(self.targetID)}
    
    def shrink_dataset(self, step: int):
        self._apply_index(slice(None, None, step))

    def subset(self, criteria):
        self._apply_index(criteria)

    #
    # Translate targetID to index
    #
    def id2index(self, targetID):
        idx = self._id_to_idx.get(int(targetID))
        if idx is None:
            raise ValueError(f"targetID {targetID} not found in the dataset.")
        else:
            return idx

    #
    # Color criteria
    #
    def color_criteria(self, blue=True, criterion=None, exclude=False):
        if blue is None:
            operators = {
                '>=': lambda x, y: x >= y,
                '<=': lambda x, y: x <= y,
                '>': lambda x, y: x > y,
                '<': lambda x, y: x < y,
            }
            if not any(op in criterion for op in operators.keys()):
                print(f"Criterion '{criterion}' not recognized. Available criteria: {list(operators.keys())}")
                return np.array([False] * self.n_spectra)
            
            
            left_color, right_color = criterion.split('-')
            if any(op in right_color for op in operators.keys()):
                operator = next(op for op in operators.keys() if op in right_color)
                right_color, val = right_color.split(operator)
            else:
                print(f"Criterion '{criterion}' not recognized. Available criteria: {list(operators.keys())}")
                return np.array([False] * self.n_spectra)
            left_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[left_color]]
            right_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[right_color]]
            criteria = operators[operator](left_mag - right_mag, float(val))
        else:
            distance = c*self.z_pipe / 70  # in Mpc
            magnitude_distance = 5 * (np.log10(distance * 1e6) - 1)  # in pc
            mag = self.color_mag[:, 1] - magnitude_distance

            M_sun_r = 4.64  # Absolute magnitude of the Sun in r-band
            luminosity_r = 10**(-0.4 * (mag - M_sun_r))
            separation_values = 1.3 * np.exp((np.log10(luminosity_r) - 13) / 1.2) + 0.45
            criteria = (self.color_mag[:, 0] - self.color_mag[:, 1]) <= separation_values
            
        if exclude:
            criteria = ~criteria
        return criteria

    def subtype_criteria(self, subtype='QSO', exclude=True):
        
        if subtype.upper() not in ['QSO', 'LRG', 'ELG', 'BGS', 'MWS']:
            print(f"Subtype '{subtype}' not recognized. Available subtypes: ['QSO', 'LRG', 'ELG', 'BGS', 'MWS']")
            return np.array([False] * self.n_spectra)
        
        # simple vectorized comparison
        is_subtype = (self.spectype == subtype.upper())

        if exclude:
            is_subtype = ~is_subtype
        return is_subtype

    #
    # Add label to the spectrum
    #
    def add_label(self, label_type='QSO', criteria=None, i=None):
        """Add a label to the spectrum.

        Args:
            i (int, optional): Index of the spectrum. Defaults to None.
            label_type (str, optional): Type of label to add. Defaults to 'QSO'.
            label_at (int, optional): Index to add the label at. Defaults to None. This is only used when label_type is uncategorized.
        """
        self.target_label = [set() for _ in range(self.n_spectra)] if self.target_label is None else self.target_label
        if i is not None:
            self.target_label[i].add(label_type)
        elif criteria is not None:
            self.target_label[criteria].add(label_type)

    def clean_label(self, label=None):
        if (label is None):
            self.target_label = [set() for _ in range(self.n_spectra)]
        elif (label is not None):
            for j in range(self.n_spectra):
                if label in self.target_label[j]:
                    self.target_label[j].remove(label)
    
    def label_filter(self, labels, exclude=True):
        is_label = np.array([any(lbl in set(x) for lbl in labels) for x in self.target_label])
        if exclude: is_label = ~is_label
        self._apply_index(is_label)
    
    def stack_data(self):
        n_spectra = self.n_spectra
        coadd_data = self.coadd_data
        ivar = self.ivar
        mask = self.mask

        lam = np.tile(desi_wavelength, (n_spectra, 1))
        fitted_model = np.zeros_like(lam)
        data_stack = np.column_stack((lam, coadd_data, ivar, mask, fitted_model))
        data_stack = data_stack.reshape(n_spectra, 5, -1)
        self.add_attribute('data_stack', data_stack)

    def mask_bad(self):
        if not hasattr(self, 'data_stack'):
            self.stack_data()

        n_spectra = self.n_spectra
        data_stack = self.data_stack
        new_grid = desi_wavelength.copy()

        # Vectorized approach for interpolation
        # Create a copy of flux and ivar to modify
        flux_interp = data_stack[:, 1, :].copy()
        ivar_interp = data_stack[:, 2, :].copy()

        # Create a boolean mask for bad pixels
        bad_mask = (data_stack[:, 3, :] != 0) | (data_stack[:, 2, :] == 0)

        # Set bad pixels to NaN to be ignored by np.interp
        flux_interp[bad_mask] = np.nan
        ivar_interp[bad_mask] = np.nan

        # Get the x-coordinates (wavelengths)
        x_coords = data_stack[:, 0, :]

        # Iterate through each spectrum to perform interpolation
        # This is necessary because np.interp doesn't handle NaNs in y-values
        # in a way that allows for a fully vectorized solution for this problem.
        for i in range(n_spectra):
            y_flux = flux_interp[i]
            y_ivar = ivar_interp[i]
            x = x_coords[i]
            
            # Create a mask for the valid (non-NaN) points for this spectrum
            valid_mask = ~np.isnan(y_flux)
            
            # If there are no valid points, fill with zeros. Otherwise, interpolate.
            if np.any(valid_mask):
                # Interpolate flux
                data_stack[i, 1, :] = np.interp(x, x[valid_mask], y_flux[valid_mask])
                # Interpolate ivar
                data_stack[i, 2, :] = np.interp(x, x[valid_mask], y_ivar[valid_mask])
            else:
                data_stack[i, 1, :] = 0.0
                data_stack[i, 2, :] = 0.0

        # Reset the mask array to all zeros as all bad pixels have been handled
        data_stack[:, 3, :] = 0
        
        # Ensure the wavelength grid is uniform
        data_stack[:, 0, :] = new_grid[np.newaxis, :]
        self.data_stack = data_stack

class FitSpectrum:
    def __init__(self,):
        pass

    def shift_to_rest_frame(self, data_class:Spectrum,
                            i=None, id=None, dz=None):
        
        n_spectra = data_class.n_spectra
        data_stack = data_class.data_stack
        id2index = data_class.id2index
        z = data_class.z
        
        if hasattr(self, 'shifted') is False:
            self.shifted = np.full(n_spectra, False)
        
        if (i is None) or (id is None):
            if self.shifted.all() is True: # all lam are shifted
                pass
            else:
                data_stack[:, 0, :] = desi_wavelength.copy() / (1 + z[:, np.newaxis])
                self.shifted[:] = True
        else:
            if id is not None:
                i = id2index(id)
            
            if dz is None:
                data_stack[i, 0, :] = desi_wavelength.copy() / (1 + z[i])
            else:
                data_stack[i, 0, :] = desi_wavelength.copy() / (1 + z[i] + dz)
            self.shifted[i] = True
        
        data_class.data_stack = data_stack
        return data_class

    #
    # Fit spectrum
    #

    
    def fit_onhs_dz(self, data_class:Spectrum, 
                    id=None, two_component=False, w_dz=False):
        
        data_stack = data_class.data_stack
        idx = data_class.id2index(id)
        
        Halpha_crop_region  = [Halpha_rest[0]-30, Halpha_rest[0]+200]
        OII_crop_region     = [   OII_rest[0]-30,    OII_rest[1]+ 30]

        lam, flux, ivar = data_stack[idx, 0, :], data_stack[idx, 1, :], data_stack[idx, 2, :]

        Halpha_slice    = (lam >= Halpha_crop_region[0]) & (lam <= Halpha_crop_region[1])
        Halpha_lam      = lam[Halpha_slice]
        Halpha_flux     = flux[Halpha_slice]
        Halpha_ivar     = ivar[Halpha_slice]
        Halpha_sigma    = np.sqrt(1/Halpha_ivar)
        
        OII_slice       = (lam >= OII_crop_region[0]) & (lam <= OII_crop_region[1])
        OII_lam         = lam[OII_slice]
        OII_flux        = flux[OII_slice]
        OII_ivar        = ivar[OII_slice]
        OII_sigma       = np.sqrt(1/OII_ivar)

        combine_lam     = np.concatenate([OII_lam, Halpha_lam])
        combine_flux    = np.concatenate([OII_flux, Halpha_flux])
        combine_sigma   = np.concatenate([OII_sigma, Halpha_sigma])

        lines_to_fit    = [*lines_vac['OII'], *lines_vac['Halpha'], *lines_vac['NII'], *lines_vac['SII']]

        if two_component:
            lam0 = lines_to_fit + lines_to_fit
        else:
            lam0 = lines_to_fit
        
        
        def fitting_func(lam_grid, *params):
            
            if two_component:
                if w_dz:
                    dz, sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2 = params[:9]
                    amp_start_index = 9
                else:
                    sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2 = params[:8]
                    amp_start_index = 8
            else:
                dz, sigma_1, conti_a_1, conti_b_1, conti_a_2, conti_b_2 = params[:6]
                amp_start_index = 6

            
            conti_parms_1           = (conti_a_1, conti_b_1)
            conti_parms_2           = (conti_a_2, conti_b_2)

            gaussian_parms_1 = []
            gaussian_parms_2 = []
            for j in range(len(lam0)):
                sigma_j = sigma_1
                amp_j   = params[j+amp_start_index]
                if two_component:
                    if w_dz:
                        lam0_j  = lam0[j] * (1 + dz + dz_r)
                    else:
                        lam0_j  = lam0[j] * (1 + dz_r)
                    if j >= len(lines_to_fit):
                        sigma_j = sigma_2
                        amp_j   = params[j+amp_start_index]
                        if w_dz:
                            lam0_j  = lam0[j] * (1 + dz + dz_l)
                        else:
                            lam0_j  = lam0[j] * (1 + dz_l)
                else:
                    lam0_j  = lam0[j] * (1 + dz)
                    
                
                if (j != 0) and (j != 1) and (j != (0+ len(lines_to_fit))) and (j != (1 + len(lines_to_fit))):
                    gaussian_parms_2.append((amp_j, lam0_j, sigma_j)) # nhs
                else:
                    gaussian_parms_1.append((amp_j, lam0_j, sigma_j)) # OII

            
            lam1 = lam_grid[:len(OII_lam)]
            lam2 = lam_grid[len(OII_lam):]

            combine_model = np.concatenate([
                model(lam1, gaussian_parms=gaussian_parms_1, conti_parms=conti_parms_1), # OII
                model(lam2, gaussian_parms=gaussian_parms_2, conti_parms=conti_parms_2) # nhs
            ])

            return combine_model

        dz_init, dz_upper, dz_lower                     = 0, 1e-3, -1e-3

        sigma_1_init, sigma_1_upper, sigma_1_lower      = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
        if two_component:
            sigma_2_init, sigma_2_upper, sigma_2_lower  = sigma_1_init, sigma_1_upper, sigma_1_lower

        if two_component:
            dz_r_init, dz_r_upper, dz_r_lower              =  1e-6, 1e-3,     0     # right component
            dz_l_init, dz_l_upper, dz_l_lower              = -1e-6,    0, -1e-3     # left component

        conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init = 0.0, 0.0, 0.0, 0.0
        conti_a_1_lower, conti_a_1_upper , conti_a_2_lower, conti_a_2_upper = -np.inf, np.inf, -np.inf, np.inf
        conti_b_1_lower, conti_b_1_upper, conti_b_2_lower, conti_b_2_upper = 0.0, np.inf, 0.0, np.inf

        amp_init, amp_upper, amp_lower              = [1]*len(lam0), [np.inf]*len(lam0), [0]*len(lam0)

        
        if two_component:
            if w_dz:
                p0 = [dz_init, sigma_1_init, sigma_2_init, dz_r_init, dz_l_init, conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init] + amp_init
                bounds_lower = [dz_lower, sigma_1_lower, sigma_2_lower, dz_r_lower, dz_l_lower, conti_a_1_lower, conti_b_1_lower, conti_a_2_lower, conti_b_2_lower] + amp_lower
                bounds_upper = [dz_upper, sigma_1_upper, sigma_2_upper, dz_r_upper, dz_l_upper, conti_a_1_upper, conti_b_1_upper, conti_a_2_upper, conti_b_2_upper] + amp_upper
            else:
                p0 = [sigma_1_init, sigma_2_init, dz_r_init, dz_l_init, conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init] + amp_init
                bounds_lower = [sigma_1_lower, sigma_2_lower, dz_r_lower, dz_l_lower, conti_a_1_lower, conti_b_1_lower, conti_a_2_lower, conti_b_2_lower] + amp_lower
                bounds_upper = [sigma_1_upper, sigma_2_upper, dz_r_upper, dz_l_upper, conti_a_1_upper, conti_b_1_upper, conti_a_2_upper, conti_b_2_upper] + amp_upper
              
        else:
            p0 = [dz_init, sigma_1_init, conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init] + amp_init
            bounds_lower = [dz_lower, sigma_1_lower, conti_a_1_lower, conti_b_1_lower, conti_a_2_lower, conti_b_2_lower] + amp_lower
            bounds_upper = [dz_upper, sigma_1_upper, conti_a_1_upper, conti_b_1_upper, conti_a_2_upper, conti_b_2_upper] + amp_upper
        
        popt, pcov = curve_fit(fitting_func, combine_lam, combine_flux, p0=p0, sigma=combine_sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
        
        params = {}
        if two_component:
            if w_dz:
                dz, sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2 = popt[:9]
                params['dz'] = dz
            else:
                sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2 = popt[:8]
                params['dz'] = (dz_r, dz_l)
            params['sigma'] = (sigma_1, sigma_2)
            params['dlam'] = (dz_r, dz_l)
            params['conti_params'] = ((conti_a_1, conti_b_1), (conti_a_2, conti_b_2))
            right_comp = []
            left_comp = []
            for k, l_0 in enumerate(lam0):
                if w_dz:
                    if k < len(lines_to_fit):
                        right_comp.append((popt[k+9], l_0*(1+dz+dz_r), sigma_1))
                    else:
                        left_comp.append((popt[k+9],  l_0*(1+dz+dz_l), sigma_2))
                else:
                    if k < len(lines_to_fit):
                        right_comp.append((popt[k+8], l_0*(1+dz_r), sigma_1))
                    else:
                        left_comp.append((popt[k+8],  l_0*(1+dz_l), sigma_2))
            params['right_comp'] = right_comp
            params['left_comp'] = left_comp
            params['gaussian_params'] = right_comp+left_comp
        else:
            dz, sigma_1, conti_a_1, conti_b_1, conti_a_2, conti_b_2 = popt[:6]
            params['dz'] = dz
            params['sigma'] = sigma_1
            params['dlam'] = None
            params['conti_params'] = ((conti_a_1, conti_b_1), (conti_a_2, conti_b_2))
            gaussian_params = []
            for k, l_0 in enumerate(lam0):
                gaussian_params.append((popt[k+6], l_0*(1+dz), sigma_1))
            params['gaussian_params'] = gaussian_params
        
        return params, (combine_lam, combine_flux, combine_sigma), len(OII_lam)
    
    
    
    def fit_onhs_dz_OII_version(self, data_class:Spectrum, 
                                id=None, two_component=False, w_dz=False):
        
        data_stack = data_class.data_stack
        idx = data_class.id2index(id)
        
        Halpha_crop_region  = [Halpha_rest[0]-30, Halpha_rest[0]+200]
        OII_crop_region     = [   OII_rest[0]-30,    OII_rest[1]+ 30]

        lam, flux, ivar = data_stack[idx, 0, :], data_stack[idx, 1, :], data_stack[idx, 2, :]

        Halpha_slice    = (lam >= Halpha_crop_region[0]) & (lam <= Halpha_crop_region[1])
        Halpha_lam      = lam[Halpha_slice]
        Halpha_flux     = flux[Halpha_slice]
        Halpha_ivar     = ivar[Halpha_slice]
        Halpha_sigma    = np.sqrt(1/Halpha_ivar)
        
        OII_slice       = (lam >= OII_crop_region[0]) & (lam <= OII_crop_region[1])
        OII_lam         = lam[OII_slice]
        OII_flux        = flux[OII_slice]
        OII_ivar        = ivar[OII_slice]
        OII_sigma       = np.sqrt(1/OII_ivar)

        combine_lam     = np.concatenate([OII_lam, Halpha_lam])
        combine_flux    = np.concatenate([OII_flux, Halpha_flux])
        combine_sigma   = np.concatenate([OII_sigma, Halpha_sigma])

        lines_to_fit    = [lines_vac['OII'][1], *lines_vac['Halpha'], *lines_vac['NII'], *lines_vac['SII']]

        if two_component:
            lam0 = lines_to_fit + lines_to_fit
        else:
            lam0 = lines_to_fit
        
        
        def fitting_func(lam_grid, *params):
            
            if two_component:
                if w_dz:
                    dz, sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio = params[:10]
                    amp_start_index = 10
                else:
                    sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio = params[:9]
                    amp_start_index = 9
            else:
                dz, sigma_1, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio = params[:7]
                amp_start_index = 7

            conti_parms_1           = (conti_a_1, conti_b_1)
            conti_parms_2           = (conti_a_2, conti_b_2)

            gaussian_parms_1 = []
            gaussian_parms_2 = []
            for j in range(len(lam0)):
                
                # ADD [OII] 3727
                if (j == 0) or (j == (0 + len(lines_to_fit))):
                    if two_component:
                        
                        amp_j = oii_ratio*params[j+amp_start_index]  
                        sigma_j = sigma_1
                        if w_dz:
                            lam0_j  = lines_vac['OII'][0] * (1 + dz + dz_r)
                        else:
                            lam0_j  = lines_vac['OII'][0] * (1 + dz_r)
                        gaussian_parms_1.append((amp_j, lam0_j, sigma_j)) # OII 3727
                    else:
                        amp_j = oii_ratio*params[j+amp_start_index]  
                        sigma_j = sigma_1
                        lam0_j  = lines_vac['OII'][0] * (1 + dz)
                        gaussian_parms_1.append((amp_j, lam0_j, sigma_j)) # OII 3727
                        


                amp_j   = params[j+amp_start_index]
                if two_component:
                    if w_dz:
                        lam0_j  = lam0[j] * (1 + dz + dz_r)
                    else:
                        lam0_j  = lam0[j] * (1 + dz_r)
                    if j >= len(lines_to_fit):
                        sigma_j = sigma_2
                        amp_j   = params[j+amp_start_index]
                        if w_dz:
                            lam0_j  = lam0[j] * (1 + dz + dz_l)
                        else:
                            lam0_j  = lam0[j] * (1 + dz_l)
                else:
                    lam0_j  = lam0[j] * (1 + dz)
                    
                
                if (j != 0) and (j != (0+ len(lines_to_fit))):
                    gaussian_parms_2.append((amp_j, lam0_j, sigma_j)) # nhs
                else:
                    gaussian_parms_1.append((amp_j, lam0_j, sigma_j)) # OII

            
            lam1 = lam_grid[:len(OII_lam)]
            lam2 = lam_grid[len(OII_lam):]

            combine_model = np.concatenate([
                model(lam1, gaussian_parms=gaussian_parms_1, conti_parms=conti_parms_1), # OII
                model(lam2, gaussian_parms=gaussian_parms_2, conti_parms=conti_parms_2) # nhs
            ])

            return combine_model

        dz_init, dz_upper, dz_lower                     = 0, 1e-3, -1e-3

        sigma_1_init, sigma_1_upper, sigma_1_lower      = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
        if two_component:
            sigma_2_init, sigma_2_upper, sigma_2_lower  = sigma_1_init, sigma_1_upper, sigma_1_lower

        if two_component:
            dz_r_init, dz_r_upper, dz_r_lower              =  1e-6, 5e-4,     0     # right component
            dz_l_init, dz_l_upper, dz_l_lower              = -1e-6,    0, -5e-4     # left component

        conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init = 0.0, 0.0, 0.0, 0.0
        conti_a_1_lower, conti_a_1_upper , conti_a_2_lower, conti_a_2_upper = -np.inf, np.inf, -np.inf, np.inf
        conti_b_1_lower, conti_b_1_upper, conti_b_2_lower, conti_b_2_upper = 0.0, np.inf, 0.0, np.inf

        amp_init, amp_upper, amp_lower              = [1]*len(lam0), [np.inf]*len(lam0), [0]*len(lam0)

        oii_ratio_init, oii_ratio_upper, oii_ratio_lower = 1.0, 5, 0.2
        
        if two_component:
            if w_dz:
                p0 = [dz_init, sigma_1_init, sigma_2_init, dz_r_init, dz_l_init, conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init, oii_ratio_init] + amp_init
                bounds_lower = [dz_lower, sigma_1_lower, sigma_2_lower, dz_r_lower, dz_l_lower, conti_a_1_lower, conti_b_1_lower, conti_a_2_lower, conti_b_2_lower, oii_ratio_lower] + amp_lower
                bounds_upper = [dz_upper, sigma_1_upper, sigma_2_upper, dz_r_upper, dz_l_upper, conti_a_1_upper, conti_b_1_upper, conti_a_2_upper, conti_b_2_upper, oii_ratio_upper] + amp_upper
            else:
                p0 = [sigma_1_init, sigma_2_init, dz_r_init, dz_l_init, conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init, oii_ratio_init] + amp_init
                bounds_lower = [sigma_1_lower, sigma_2_lower, dz_r_lower, dz_l_lower, conti_a_1_lower, conti_b_1_lower, conti_a_2_lower, conti_b_2_lower, oii_ratio_lower] + amp_lower
                bounds_upper = [sigma_1_upper, sigma_2_upper, dz_r_upper, dz_l_upper, conti_a_1_upper, conti_b_1_upper, conti_a_2_upper, conti_b_2_upper, oii_ratio_upper] + amp_upper

        else:
            p0 = [dz_init, sigma_1_init, conti_a_1_init, conti_b_1_init, conti_a_2_init, conti_b_2_init, oii_ratio_init] + amp_init
            bounds_lower = [dz_lower, sigma_1_lower, conti_a_1_lower, conti_b_1_lower, conti_a_2_lower, conti_b_2_lower, oii_ratio_lower] + amp_lower
            bounds_upper = [dz_upper, sigma_1_upper, conti_a_1_upper, conti_b_1_upper, conti_a_2_upper, conti_b_2_upper, oii_ratio_upper] + amp_upper

        popt, pcov = curve_fit(fitting_func, combine_lam, combine_flux, p0=p0, sigma=combine_sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
        
        params = {}
        if two_component:
            if w_dz:
                dz, sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio = popt[:10]
                params['dz'] = dz
            else:
                sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio = popt[:9]
                params['dz'] = (dz_r, dz_l)
            params['sigma'] = (sigma_1, sigma_2)
            params['dlam'] = (dz_r, dz_l)
            params['conti_params'] = ((conti_a_1, conti_b_1), (conti_a_2, conti_b_2))
            right_comp = []
            left_comp = []
            for k, l_0 in enumerate(lam0):
                # ADD [OII] 3727
                if (k == 0) or (k == (0 + len(lines_to_fit))):
                    if w_dz:
                        if k < len(lines_to_fit):
                            right_comp.append((oii_ratio*popt[k+10], lines_vac['OII'][0]*(1+dz+dz_r), sigma_1))
                        else:
                            left_comp.append((oii_ratio*popt[k+10],  lines_vac['OII'][0]*(1+dz+dz_l), sigma_2))
                    else:
                        if k < len(lines_to_fit):
                            right_comp.append((oii_ratio*popt[k+9], lines_vac['OII'][0]*(1+dz_r), sigma_1))
                        else:
                            left_comp.append((oii_ratio*popt[k+9],  lines_vac['OII'][0]*(1+dz_l), sigma_2))
        
                if w_dz:
                    if k < len(lines_to_fit):
                        right_comp.append((popt[k+10], l_0*(1+dz+dz_r), sigma_1))
                    else:
                        left_comp.append((popt[k+10],  l_0*(1+dz+dz_l), sigma_2))
                else:
                    if k < len(lines_to_fit):
                        right_comp.append((popt[k+9], l_0*(1+dz_r), sigma_1))
                    else:
                        left_comp.append((popt[k+9],  l_0*(1+dz_l), sigma_2))
            params['right_comp'] = right_comp
            params['left_comp'] = left_comp
            params['gaussian_params'] = right_comp+left_comp
        else:
            dz, sigma_1, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio = popt[:7]
            params['dz'] = dz
            params['sigma'] = sigma_1
            params['dlam'] = None
            params['conti_params'] = ((conti_a_1, conti_b_1), (conti_a_2, conti_b_2))
            gaussian_params = []
            for k, l_0 in enumerate(lam0):
                # ADD [OII] 3727
                if k == 0:
                    gaussian_params.append((oii_ratio*popt[k+7], lines_vac['OII'][0]*(1+dz), sigma_1))
                gaussian_params.append((popt[k+7], l_0*(1+dz), sigma_1))
            params['gaussian_params'] = gaussian_params
        
        return params, (combine_lam, combine_flux, combine_sigma), len(OII_lam)
    
    
    def fit_z(self, data_class:Spectrum,
               id=None,):

        try:
            params_2comp, (combine_lam, combine_flux, combine_sigma), separation = self.fit_onhs_dlam(data_class, id=id, two_component=True)
            dz_l = params_2comp['dz']
            conti_1, conti_2 = params_2comp['conti_params']
            conti           = np.concatenate([model(combine_lam[:separation], conti_parms=conti_1), 
                                            model(combine_lam[separation:], conti_parms=conti_2)])
            left_comp       = np.concatenate([model(combine_lam[:separation], gaussian_parms=params_2comp['left_comp'][:2]), 
                                            model(combine_lam[separation:], gaussian_parms=params_2comp['left_comp'][2:])])
            right_comp      = np.concatenate([model(combine_lam[:separation], gaussian_parms=params_2comp['right_comp'][:2]), 
                                            model(combine_lam[separation:], gaussian_parms=params_2comp['right_comp'][2:])])
            combine_2comp   = conti + left_comp + right_comp
            chisq_2comp     = np.sum(((combine_flux - combine_2comp)/combine_sigma)**2)
            dof_2comp       = len(combine_flux) - 19   # 9 + 7*2 - 4(conti; not relevant to F-test)
                
            params_1comp, (combine_lam, combine_flux, combine_sigma), separation = self.fit_onhs_dlam(data_class, id=id, two_component=False)
            dz_r = params_1comp['dz']
            conti_1, conti_2 = params_1comp['conti_params']
            conti           = np.concatenate([model(combine_lam[:separation], conti_parms=conti_1), 
                                            model(combine_lam[separation:], conti_parms=conti_2)])
            gaussian_comp   = np.concatenate([model(combine_lam[:separation], gaussian_parms=params_1comp['gaussian_params']), 
                                            model(combine_lam[separation:], gaussian_parms=params_1comp['gaussian_params'])])
            combine_1comp   = conti + gaussian_comp
            chisq_1comp     = np.sum(((combine_flux - combine_1comp)/combine_sigma)**2)
            dof_1comp       = len(combine_flux) - 9   # 6 + 7 - 4(conti; not relevant to F-test)

            F_stat = ((chisq_1comp - chisq_2comp) / (dof_1comp - dof_2comp)) / (chisq_2comp / dof_2comp)
            p_value = 1 - f.cdf(F_stat, dof_1comp - dof_2comp, dof_2comp)
            
            if p_value < 0.1: # try 2 component
                delta_lam = np.abs(params_2comp['dlam'][0] - params_2comp['dlam'][1])
                if delta_lam >= 0.8:
                    mode = '2_component'
                else:
                    mode = '1_component'
            else:
                mode = '1_component'
        except:
            try:
                params_1comp, (combine_lam, combine_flux, combine_sigma), separation = self.fit_onhs(data_class, id=id, two_component=False)
                dz_r = params_1comp['dz']
                conti_1, conti_2 = params_1comp['conti_params']
                conti           = np.concatenate([model(combine_lam[:separation], conti_parms=conti_1), 
                                                model(combine_lam[separation:], conti_parms=conti_2)])
                gaussian_comp   = np.concatenate([model(combine_lam[:separation], gaussian_parms=params_1comp['gaussian_params']), 
                                                model(combine_lam[separation:], gaussian_parms=params_1comp['gaussian_params'])])
                combine_1comp   = conti + gaussian_comp
            except:
                mode = 'fit_failed'
        
    def fit_flux(self,
                 id=None,
                 fitting_model=model, 
                 lam0=None, 
                 region=None, 
                 fit_z=True,
                 two_component=False, 
                 e_or_a='e',):

        ##############################
        i = self.id2index(id)
        
        """
        Data to fit
        """
        # lam = desi_wavelength.copy()
        
        """
        Fitting parameters
        ------------------------------
        If fit_z is True
        """
        fit_line = lam0.copy()
        if two_component:
            lam0 = lam0 + lam0 # duplicate to generate free components
        
        ##############################
        
        def fitting_func(lam, *params):
            gaussian_parms = []
            if fit_z:
                dz, sigma_1 = params[0], params[1]
                if two_component:
                    sigma_2 = params[2]
                    
                for j in range(len(lam0)):
                    if two_component:
                        amp_j, sigma_j = params[j+3], sigma_1
                        lam0_j = lam0[j] * (1 + dz) # fixed components

                        if j >= len(fit_line):
                            dlam = params[-3]
                            amp_j, sigma_j = params[j+3], sigma_2
                            lam0_j = (lam0[j]+dlam) * (1 + dz) # free components
                    else:
                        amp_j, sigma_j = params[j+2], sigma_1
                        lam0_j = lam0[j] * (1 + dz)
                    gaussian_parms.append((amp_j, lam0_j, sigma_j))
            else:
                sigma_1 = params[0]
                if two_component:
                    sigma_2 = params[1]
                for j in range(len(lam0)):
                    if two_component:
                        amp_j, sigma_j = params[j+2], sigma_1
                        lam0_j = lam0[j] # fixed components
                        if j >= len(fit_line):
                            dlam = params[-3]
                            amp_j, sigma_j = params[j+2], sigma_2
                            lam0_j = (lam0[j]+dlam) # free components
                    else:
                        offset = params[-3] # system offset
                        amp_j, sigma_j = params[j+1], sigma_1
                        lam0_j = (lam0[j] + offset)

                    gaussian_parms.append((amp_j, lam0_j, sigma_j))
            
            conti_a = params[-2]
            conti_b = params[-1]
            conti_parms = (conti_a, conti_b)
            
            return fitting_model(lam, gaussian_parms=gaussian_parms, conti_parms=conti_parms)
        
        # self.shift_to_rest_frame()
        lam, flux, ivar = self.data_stack[i, 0, :], self.data_stack[i, 1, :], self.data_stack[i, 2, :]

        if region is None: region = (np.min(lam0)-50, np.max(lam0)+50)

        crop_region = (lam >= region[0]) & (lam <= region[1])
        lam = lam[crop_region]
        flux = flux[crop_region]
        ivar = ivar[crop_region]
        sigma = np.sqrt(1/ivar)

        # Initial guess for parameters

        conti_a_init, conti_b_init = 0.0, 0.0
        conti_a_lower, conti_a_upper = -np.inf, np.inf
        conti_b_lower, conti_b_upper = 0.0, np.inf

        if (fit_z is True) and (two_component is True):
            """
            base_2 or OII_2 model
            len(p0) = 3 + n_fit_line * 2 + 3
            """
            
            dz_init, dz_upper, dz_lower                 = 0, 0.01, -0.01
            sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
            sigma_2_init, sigma_2_upper, sigma_2_lower  = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
            dlam_init, dlam_upper, dlam_lower           = 0, 10, -10
            amp_init, amp_upper, amp_lower              = [1]*len(lam0), [np.inf]*len(lam0), [0]*len(lam0)
            
            p0 = [dz_init, sigma_1_init, sigma_2_init] + amp_init + [dlam_init, conti_a_init, conti_b_init]
            bounds_lower = [dz_lower, sigma_1_lower, sigma_2_lower] + amp_lower + [dlam_lower, conti_a_lower, conti_b_lower]
            bounds_upper = [dz_upper, sigma_1_upper, sigma_2_upper] + amp_upper + [dlam_upper, conti_a_upper, conti_b_upper]
        
        elif (fit_z is True) and (two_component is False):
            """
            base or OII model
            len(p0) = 2 + n_fit_line + 2
            """
            dz_init, dz_upper, dz_lower                 = 0, 0.01, -0.01
            sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
            amp_init, amp_upper, amp_lower              = [1]*len(lam0), [np.inf]*len(lam0), [0]*len(lam0)
            
            p0 = [dz_init, sigma_1_init] + amp_init + [conti_a_init, conti_b_init]
            bounds_lower = [dz_lower, sigma_1_lower] + amp_lower + [conti_a_lower, conti_b_lower]
            bounds_upper = [dz_upper, sigma_1_upper] + amp_upper + [conti_a_upper, conti_b_upper]
            
        elif (fit_z is False) and (two_component is True):
            """
            Fit two components without redshift adjustment
            len(p0) = 2 + n_fit_line * 2 + 3
            """
            sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
            sigma_2_init, sigma_2_upper, sigma_2_lower  = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
            dlam_init, dlam_upper, dlam_lower           = 0, 6, -6
            if e_or_a == 'a':
                amp_init, amp_upper, amp_lower          = [-1]*len(lam0), [-(1e-2)]*len(lam0), [-np.inf]*len(lam0)
            else:
                amp_init, amp_upper, amp_lower          = [1]*len(lam0), [np.inf]*len(lam0), [1e-2]*len(lam0)
            
            p0 = [sigma_1_init, sigma_2_init] + amp_init + [dlam_init, conti_a_init, conti_b_init]
            bounds_lower = [sigma_1_lower, sigma_2_lower] + amp_lower + [dlam_lower, conti_a_lower, conti_b_lower]
            bounds_upper = [sigma_1_upper, sigma_2_upper] + amp_upper + [dlam_upper, conti_a_upper, conti_b_upper]
        
        elif (fit_z is False) and (two_component is False):
            """
            Fit one component without redshift adjustment
            len(p0) = 1 + n_fit_line + 3
            """
            sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
            offset_init, offset_upper, offset_lower     = 0, 5, -5
            if e_or_a == 'a':
                amp_init, amp_upper, amp_lower          = [-1]*len(lam0), [-(1e-2)]*len(lam0), [-np.inf]*len(lam0)
            else:
                amp_init, amp_upper, amp_lower          = [1]*len(lam0), [np.inf]*len(lam0), [1e-2]*len(lam0)

            p0 = [sigma_1_init] + amp_init + [offset_init, conti_a_init, conti_b_init]
            bounds_lower = [sigma_1_lower] + amp_lower + [offset_lower, conti_a_lower, conti_b_lower]
            bounds_upper = [sigma_1_upper] + amp_upper + [offset_upper, conti_a_upper, conti_b_upper]

        try:
            popt, pcov = curve_fit(fitting_func, lam, flux, p0=p0, sigma=sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
            if popt is None:
                status = False
            else:
                status = True
        except:
            popt = None
            status = False

        params = {}
        if status is True:
            if (fit_z is True) and (two_component is True):
                params['dz'] = popt[0]
                params['sigma'] = (popt[1], popt[2])
                params['dlam'] = popt[-3]
                params['conti_params'] = (popt[-2], popt[-1])
                
                fixed_comps = []
                free_comps = []
                for k, l_0 in enumerate(lam0):
                    if k < len(fit_line):
                        fixed_comps.append((popt[k+3], l_0*(1+popt[0]), popt[1]))
                    else:
                        free_comps.append((popt[k+3], (l_0+popt[-3])*(1+popt[0]), popt[2]))
                params['fixed_comps'] = fixed_comps
                params['free_comps'] = free_comps
                params['gaussian_params'] = free_comps + fixed_comps
                
            elif (fit_z is True) and (two_component is False):
                params['dz'] = popt[0]
                params['sigma'] = popt[1]
                params['conti_params'] = (popt[-2], popt[-1])
                free_comps = []
                for k, l_0 in enumerate(lam0):
                    free_comps.append((popt[k+2], l_0*(1+popt[0]), popt[1]))
                params['free_comps'] = free_comps
                params['fixed_comps'] = None
                params['gaussian_params'] = free_comps

            elif (fit_z is False) and (two_component is True):
                params['sigma'] = (popt[0], popt[1])
                params['dlam'] = popt[-3]
                params['conti_params'] = (popt[-2], popt[-1])
                fixed_comps = []
                free_comps = []
                for k, l_0 in enumerate(lam0):
                    if k < len(fit_line):
                        fixed_comps.append((popt[k+2], l_0, popt[0]))
                    else:
                        free_comps.append((popt[k+2], l_0 + popt[-3], popt[1]))
                params['fixed_comps'] = fixed_comps
                params['free_comps'] = free_comps
                params['gaussian_params'] = fixed_comps + free_comps
                
            elif (fit_z is False) and (two_component is False):
                params['sigma'] = popt[0]
                params['dlam'] = popt[-3]
                params['conti_params'] = (popt[-2], popt[-1])
                free_comps = []
                for k, l_0 in enumerate(lam0):
                    free_comps.append((popt[k+1], l_0 + popt[-3], popt[0]))
                params['free_comps'] = free_comps
                params['fixed_comps'] = None
                params['gaussian_params'] = free_comps
        else:
            params['dz'] = None
            params['sigma'] = None
            params['dlam'] = None
            params['conti_params'] = None
            params['fixed_comps'] = None
            params['free_comps'] = None
            params['gaussian_params'] = None

        return params, status
    
    def calculate_chisq(self, id, params, region=None):
        if region is None:
            region = [lines_vac['Halpha'][0]-40, lines_vac['Halpha'][0]+40]
        
        i = self.id2index(id)
        lam = self.data_stack[i, 0, :]
        flux = self.data_stack[i, 1, :]
        ivar = self.data_stack[i, 2, :]
        
        crop = (lam > region[0]) & (lam < region[1])
        lam = lam[crop]
        flux = flux[crop]
        ivar = ivar[crop]
        
        sigma = np.sqrt(1/ivar)
        fitted_model = model(lam, gaussian_parms=params['gaussian_params'], conti_parms=params['conti_params'])

        chisq = np.sum(((flux - fitted_model) / sigma) ** 2)
        return chisq

    def calculate_f_test(self, id, params_s, params_d, region=None):
        if region is None:
            region = [lines_vac['Halpha'][0]-40, lines_vac['Halpha'][0]+40]
        
        i = self.id2index(id)
        lam = self.data_stack[i, 0, :]
        flux = self.data_stack[i, 1, :]
        crop = (lam > region[0]) & (lam < region[1])
        lam = lam[crop]
        flux = flux[crop]
        
        chisq_s = self.calculate_chisq(id, params_s, region=region)
        chisq_d = self.calculate_chisq(id, params_d, region=region)

        n_params_s = 2 + len(params_s['gaussian_params']) -2 + 2  # +2 for continuum parameters
        n_params_d = 3 + len(params_d['gaussian_params']) -4 + 2  # +2 for continuum parameters


        dof_s = len(flux) - n_params_s  # subtract number of parameters
        dof_d = len(flux) - n_params_d  # subtract number of parameters

        F_stat = ((chisq_s - chisq_d) / (dof_s - dof_d)) / (chisq_d / dof_d)
        p_value = 1 - f.cdf(F_stat, dof_s - dof_d, dof_d)

        return F_stat, p_value
    
    def adjust_z(self, id, mode='base_2'):
        
        """
        Adjust redshift based on fitted dz with forbidden lines.
        
        Mode:
        - 'auto'  : Choose the best mode automatically
        - 'base_2': Using Halpha, NII, SII (2 components)
        - 'base'  : Using Halpha, NII, SII (1 component)
        - 'OII_2' : Using OII (2 components)
        - 'OII'   : Using OII (1 component)
        - 'no'  : No adjustment
        
        """
        
        i = self.id2index(id)
        
        if hasattr(self, 'adjust_z_mode') is False or self.adjust_z_mode is None:
            self.adjust_z_mode = ['no'] * self.n_spectra
        
        Halpha_rest = lines_vac['Halpha']
        NII_rest    = lines_vac['NII']
        SII_rest    = lines_vac['SII']
        OII_rest    = lines_vac['OII']

        modes = {
            'base_2': ([*Halpha_rest, *NII_rest, *SII_rest], True),
            'base': ([*Halpha_rest, *NII_rest, *SII_rest], False),
            'OII_2': ([*OII_rest], True),
            'OII': ([*OII_rest], False),
        }

        def get_z(fit_line, two_comp):
            region = (np.min(fit_line)-50, np.max(fit_line)+50)
            try:
                params, status = self.fit_flux(id=id, fitting_model=model, lam0=fit_line, region=region, fit_z=True, two_component=two_comp)

            except Exception as e:
                print(f"Error fitting flux for {self.targetID[i]}: {e}")
                return 0

            lam, flux, ivar, mask = self.data_stack[i, 0, :], self.data_stack[i, 1, :], self.data_stack[i, 2, :], self.data_stack[i, 3, :]
            
            crop_region = (lam >= region[0]) & (lam <= region[1])
            lam = lam[crop_region]
            flux = flux[crop_region]

            
            
            if status is True:
                if two_comp:
                    dz, dlam = params['dz'], params['dlam']
                    line_free = params['free_comps']
                    line_fixed = params['fixed_comps']
                    if np.abs(dlam) < 0.8:
                        return 0
                    fitted_model = model(lam, gaussian_parms=params['gaussian_params'], conti_parms=params['conti_params'])

                else:
                    dz = params['dz']
                    line_free = params['free_comps']
                    fitted_model = model(lam, gaussian_parms=params['gaussian_params'], conti_parms=params['conti_params'])

                noise = np.std(flux - fitted_model)
                count = 0
                for j in range(len(fit_line)):
                    if two_comp:
                        if min(line_free[j][0], line_fixed[j][0]) > 3*noise:
                            count += 1
                    else:
                        if line_free[j][0] > 3*noise:
                            count += 1
                if count >= (len(fit_line)//3):
                    return dz
                else:
                    return 0
            else:
                return 0


        if mode == 'auto':
            for using_mode in list(modes.keys()):
                fit_line, two_comp = modes[using_mode]
                dz = get_z(fit_line, two_comp)
                mode = using_mode
                if np.abs(dz) >= 1e-4:
                    break
            if (np.abs(dz) < 1e-5):
                mode = 'no'
                dz = 0
        elif mode == 'base_2' or mode == 'OII_2':
            fit_line, two_comp = modes[mode]
            dz = get_z(fit_line, two_comp)
            if (np.abs(dz) < 1e-4):
                mode = mode.replace('_2', '')
                fit_line, two_comp = modes[mode]
                dz = get_z(fit_line, two_comp)
                if (np.abs(dz) < 1e-5):
                    mode = 'no'
                    dz = 0
        elif mode == 'base' or mode == 'OII':
            fit_line, two_comp = modes[mode]
            dz = get_z(fit_line, two_comp)
            if (np.abs(dz) < 1e-5):
                mode = 'no'
                dz = 0
        elif mode == 'no':
            dz = 0

        self.adjust_z_mode[i] = mode
        self.z[i] = self.z_pipe[i] + dz
        return dz
    
    def search_NaD(self, id, two_component=True):
        
        i = self.id2index(id)
        
        if hasattr(self, 'searched_NaD') is False or self.searched_NaD is None:
            self.searched_NaD = np.full(self.n_spectra, 'no', dtype=object)
        else:
            self.searched_NaD = np.asarray(self.searched_NaD, dtype=object)

        NaD_rest = lines_vac['NaD']

        region = (np.min(NaD_rest) - 50, np.max(NaD_rest) + 50)
        
        
        z = self.z[i]
        lam, flux, ivar, mask = self.data_stack[i, 0, :], self.data_stack[i, 1, :], self.data_stack[i, 2, :], self.data_stack[i, 3, :]
        crop_region = (lam >= region[0]) & (lam <= region[1])
        lam = lam[crop_region]
        flux = flux[crop_region]
        
        if two_component:
            try:
                params, status = self.fit_flux(id=id, fitting_model=model, lam0=[*NaD_rest], region=region, fit_z=False, two_component=True, e_or_a='a')
                if status is False:
                    return self.search_NaD(id=id, two_component=False)
                
                line_D2_fixed, line_D1_fixed = params['fixed_comps']
                line_D2_free, line_D1_free   = params['free_comps']
                amp_D2, _, _      = line_D2_fixed
                amp_D1, _, _      = line_D1_fixed
                amp_D2_free, _, _ = line_D2_free
                amp_D1_free, _, _ = line_D1_free
                
                dlam = params['dlam']
                if np.abs(dlam) < 0.8:
                    return self.search_NaD(id=id, two_component=False)

                fitted_model = model(lam, gaussian_parms=params['gaussian_params'], conti_parms=params['conti_params'])
                noise = np.std(flux - fitted_model)
                
                if min(min(np.abs(amp_D1), np.abs(amp_D1_free)), min(np.abs(amp_D2), np.abs(amp_D2_free))) > 3*noise:
                    if dlam > 0.8:
                        self.searched_NaD[i] = 'inflow_2'
                    elif dlam < -0.8:
                        self.searched_NaD[i] = 'outflow_2'
                    else:
                        self.searched_NaD[i] = 'systematic_2'
                    return params, status
                else:
                    return self.search_NaD(id=id, two_component=False)
            except:
                return self.search_NaD(id=id, two_component=False)
        else:
            try:
                params, status = self.fit_flux(id=id, fitting_model=model, lam0=[*NaD_rest], region=region, fit_z=False, two_component=False, e_or_a='a')
                if status is False:
                    self.searched_NaD[i] = 'no_detection'
                    return params, False
                
                line_D2, line_D1 = params['free_comps']
                amp_D2, _, _ = line_D2
                amp_D1, _, _ = line_D1
                
                offset = params['dlam']
                fitted_model = model(lam, gaussian_parms=params['gaussian_params'], conti_parms=params['conti_params'])
                noise = np.std(flux - fitted_model)
                
                if min(np.abs(amp_D1), np.abs(amp_D2)) > 3*noise:
                    if offset > 0.8:
                        self.searched_NaD[i] = 'inflow'
                    elif offset < -0.8:
                        self.searched_NaD[i] = 'outflow'
                    else:
                        self.searched_NaD[i] = 'systematic'
                    return params, True
                else:
                    self.searched_NaD[i] = 'no_detection'
                    return params, False
            except Exception as e:
                print(f"Error occurred while searching NaD for {self.targetID[i]}: {e}")
                self.searched_NaD[i] = 'no_detection'
                return params, False


class PlotSpectrum:
    def __init__(self, data_class:Spectrum):
        self.n_spectra     = data_class.n_spectra
        self.targetID      = data_class.targetID
        self.z_pipe        = data_class.z_pipe
        self.z             = data_class.z
        self.RA            = data_class.RA
        self.DEC           = data_class.DEC
        self.coadd_data    = data_class.coadd_data
        self.ivar          = data_class.ivar
        self.mask          = data_class.mask
        self.target_label  = data_class.target_label
        self.spectype      = data_class.spectype
        self.color_mag     = data_class.color_mag
        self.id2index      = data_class.id2index

    def hist_color(self, x='g-z', show=True, save=True, fname=None, **kwargs):
        if '-' in x:
            left_color, right_color = x.split('-')
            left_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[left_color]]
            right_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[right_color]]
            x_data = left_mag - right_mag
        else:
            x_data = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[x]]
        
        plt.figure(figsize=kwargs.get('figsize', (8,6)))
        
        plt.hist(x_data, bins=kwargs.get('bins', 30), color=kwargs.get('color', 'blue'), alpha=0.7)
        plt.xlabel(f'{x}')
        plt.ylabel('Count')
        plt.title(f'Histogram of {x}')
        if save:
            plt.savefig(fname if fname else f'hist_{x}.png', dpi=300)
            print(f"Saved histogram as {fname if fname else f'hist_{x}.png'}")
        if show:
            plt.show()
        

    def scat_colors(self, x='g-r', y='g-z', show=True, save=True, fname=None, **kwargs):
        if '-' in x:
            left_color, right_color = x.split('-')
            left_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[left_color]]
            right_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[right_color]]
            x_data = left_mag - right_mag
        else:
            x_data = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[x]]
            
        if '-' in y:
            left_color, right_color = y.split('-')
            left_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[left_color]]
            right_mag = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[right_color]]
            y_data = left_mag - right_mag
        else:
            y_data = self.color_mag[:, {'g': 0, 'r': 1, 'z': 2, 'w1': 3, 'w2': 4}[y]]

        plt.figure(figsize=kwargs.get('figsize', (8,6)))
        plt.scatter(x_data, y_data, s=kwargs.get('s', 10), c=kwargs.get('c', 'blue'), alpha=kwargs.get('alpha', 0.7))
        plt.xlabel(f'{x}')
        plt.ylabel(f'{y}')
        plt.title(f'Scatter plot of {y} vs {x}')
        
        if save:
            plt.savefig(fname if fname else f'scat_{y}_vs_{x}.png', dpi=300)
            print(f"Saved scatter plot as {fname if fname else f'scat_{y}_vs_{x}.png'}")
        if show:
            plt.show()

    def step_spectrum(self, id, range=None, show=True, save=True, fname=None, **kwargs):
        i = self.id2index(id)

        lam = desi_wavelength.copy() /  (1 + self.z[i])
        mask = self.mask[i]
        flux = self.coadd_data[i]
        ivar = self.ivar[i]
        sigma = np.sqrt(1 / ivar)

        
        if range:
            crop = (lam > range[0]) & (lam < range[1])
            lam = lam[crop]
            flux = flux[crop]
            sigma = sigma[crop]
            mask = mask[crop]

        plt.figure(figsize=kwargs.get('figsize', (12,6)))
        plt.step(lam, flux, where='mid', color=kwargs.get('color', 'black'), label='Flux')
        plt.plot(lam, sigma, color=kwargs.get('sigma_color', 'red'), linestyle='--', label='sigma')
        plt.plot(lam, mask*10, color=kwargs.get('mask_color', 'red'), linestyle=':', label='mask')
        for line_name, line_wavelength in lines_vac.items():
            for lw in line_wavelength:
                if lw > np.max(lam) or lw < np.min(lam):
                    continue
                else:
                    plt.axvline(x=lw, color=kwargs.get('line_color', 'darkgreen'), linestyle=':', alpha=0.7)
                    plt.text(lw+3, np.max(flux)*0.95, line_name, rotation=90, color=kwargs.get('line_color', 'darkgreen'), fontsize=8)
        
        plt.xlabel('Rest Wavelength (Å)')
        plt.ylabel('Flux')
        plt.title(f'Spectrum of TargetID: {self.targetID[i]} at z={self.z[i]:.4f}')
        plt.legend()
        plt.xlim(kwargs.get('xlim', (np.min(lam), np.max(lam))))
        
        if save:
            plt.savefig(fname if fname else f'spectrum_{self.targetID[i]}.png')
            print(f"Saved spectrum plot as {fname if fname else f'spectrum_{self.targetID[i]}.png'}")
        if show:
            plt.show()
