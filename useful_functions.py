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


OIII_rest   = lines_vac['OIII']
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

def image_link(RA, DEC, save_image=False, fname=None, plot=False):
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
    if plot and save_image:
        plt.figure(figsize=(5,5))
        plt.imshow(plt.imread(fname if fname else "cutout.jpg"))
        plt.axis('off')
        plt.show()
    return f'https://www.legacysurvey.org/viewer?ra={RA}&dec={DEC}&layer=ls-dr10-grz&zoom=14'

def spectrum_link(targetID): 
    return f'https://www.legacysurvey.org/viewer/desi-spectrum/dr1/targetid{targetID}'


################################################################################################################
class Spectrum:

    def __init__(self, spectra_data, color_data, cigale_data, fastspecfit_data):

        # Extract data tables
        sd1_all         = spectra_data[1].data
        cd1_all         = color_data[1].data
        cigale_all      = cigale_data[1].data
        fastspecfit_all = fastspecfit_data[1].data


        # Find common TARGETIDs
        common_ids  = np.intersect1d(sd1_all['TARGETID'], cd1_all['TARGETID'])
        common_ids  = np.intersect1d(common_ids,  fastspecfit_all['TARGETID'])
        final_ids   = np.intersect1d(common_ids,       cigale_all['TARGETID'])

        # Create boolean masks for filtering
        spectra_mask        = np.isin(sd1_all['TARGETID'], final_ids)
        color_mask          = np.isin(cd1_all['TARGETID'], final_ids)
        cigale_mask         = np.isin(cigale_all['TARGETID'], final_ids)
        fastspecfit_mask    = np.isin(fastspecfit_all['TARGETID'], final_ids)

        # Filter the data
        sd1_filtered            = sd1_all[spectra_mask]
        cd1_filtered            = cd1_all[color_mask]
        cigale_filtered         = cigale_all[cigale_mask]
        fastspecfit_filtered    = fastspecfit_all[fastspecfit_mask]

        # Get the sorting order from one of the filtered arrays
        sort_order = np.argsort(sd1_filtered['TARGETID'])

        # Apply the same sorting order to all datasets
        sd1         = sd1_filtered[sort_order]
        sd2         = spectra_data[2].data[spectra_mask][sort_order]
        sd3         = spectra_data[3].data[spectra_mask][sort_order]
        sd4         = spectra_data[4].data[spectra_mask][sort_order]
        cd1         = cd1_filtered[np.argsort(cd1_filtered['TARGETID'])] # Re-sort to ensure alignment
        cigale      = cigale_filtered[np.argsort(cigale_filtered['TARGETID'])] # Re-sort to ensure alignment
        fd2         = fastspecfit_data[2].data[fastspecfit_mask][np.argsort(fastspecfit_filtered['TARGETID'])] # CONTINUUM
        fd4         = fastspecfit_data[4].data[fastspecfit_mask][np.argsort(fastspecfit_filtered['TARGETID'])] # EMISSION


        self.targetID   = np.asarray(sd1['TARGETID'])
        self.n_spectra  = len(self.targetID)
        self.z_pipe     = np.asarray(sd1['Z'])
        self.z          = self.z_pipe.copy()
        self.RA         = np.asarray(sd1['RA'])
        self.DEC        = np.asarray(sd1['DEC'])
        self.coadd_data = np.asarray(sd2[:, 0, :], dtype=np.float32)
        self.ivar       = np.asarray(sd3[:, 0, :], dtype=np.float32)
        self.mask       = np.asarray(sd4[:, 0, :], dtype=np.float32)
        self.spectype   = cd1['SPECTYPE']
        self.logM       = cigale['LOGM']
        self.logSFR     = cigale['LOGSFR']
        self.continuum  = np.asarray(fd2, dtype=np.float32)
        self.emission   = np.asarray(fd4, dtype=np.float32)
        
        
        
        g = cd1['FLUX_G']; r = cd1['FLUX_R']; z = cd1['FLUX_Z']
        w1 = cd1['FLUX_W1']; w2 = cd1['FLUX_W2']
        color_flux = np.column_stack((g, r, z, w1, w2))
        with np.errstate(divide='ignore', invalid='ignore'):
            color_mag_all = 22.5 - 2.5 * np.log10(color_flux)
        self.color_mag = color_mag_all
        
        
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
        self.n_spectra  = len(self.targetID)
        self.z_pipe     = self.z_pipe[idx]
        self.z          = self.z[idx]
        self.RA         = self.RA[idx]
        self.DEC        = self.DEC[idx]
        self.coadd_data = self.coadd_data[idx]
        self.ivar       = self.ivar[idx]
        self.mask       = self.mask[idx]
        self.spectype   = self.spectype[idx]
        self.logM       = self.logM[idx]
        self.logSFR     = self.logSFR[idx]
        self.continuum  = self.continuum[idx]
        self.emission   = self.emission[idx]
        if hasattr(self, 'data_stack'):     self.data_stack = self.data_stack[idx, :, :]
        if hasattr(self, "color_mag"):      self.color_mag = self.color_mag[idx]
        if hasattr(self, "smoothed_flux"):  self.smoothed_flux = self.smoothed_flux[idx]
        if hasattr(self, "target_label") and self.target_label is not None:
            if isinstance(idx, np.ndarray) and idx.dtype == bool:
                self.target_label = [lbl for lbl, keep in zip(self.target_label, idx) if keep]
            else:
                self.target_label = [self.target_label[i] for i in np.atleast_1d(idx)]

        
        self._id_to_idx = {int(tid): i for i, tid in enumerate(self.targetID)}
    
    def shrink_dataset(self, step: int):
        self._apply_index(slice(None, None, step))

    def subset(self, criteria):
        self._apply_index(criteria)

    #
    # Translate targetID to index
    #
    def id2index(self, targetID):
        """
        Convert targetID or a list/array of targetIDs to internal array index/indices.
        """
        if np.isscalar(targetID):
            idx = self._id_to_idx.get(int(targetID))
            if idx is None:
                raise ValueError(f"targetID {targetID} not found in the dataset.")
            return idx
        else:
            # Handle list or numpy array of targetIDs
            try:
                # Use a list comprehension for efficiency
                indices = [self._id_to_idx[int(tid)] for tid in targetID]
                # Return a numpy array if the input was one
                if isinstance(targetID, np.ndarray):
                    return np.array(indices)
                return indices
            except KeyError as e:
                # If any ID is not found, raise an error
                raise ValueError(f"targetID {e.args[0]} not found in the dataset.") from e

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

    def SFG_criteria(self, exclude=False):
        logM = self.logM
        logSFR = self.logSFR
        criteria = logSFR > (1 * (logM - 10) - 3)

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
    # Stack data and mask bad pixels for convenience
    #    
    def stack_data(self):
        n_spectra = self.n_spectra
        coadd_data = self.coadd_data
        ivar = self.ivar
        mask = self.mask
        continuum = self.continuum
        emission = self.emission
        
        
        lam = np.tile(desi_wavelength, (n_spectra, 1))
        data_stack = np.column_stack((lam, coadd_data-continuum, ivar, mask, continuum, emission))
        data_stack = data_stack.reshape(n_spectra, 6, -1)
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
    
    #
    # Shift to rest frame
    #
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

    def significant_emission_filter(self, data_class:Spectrum):
        n_spectra = data_class.n_spectra
        data_stack = data_class.data_stack

        filter_mask = np.full(n_spectra, False)
        for idx in range(n_spectra):
            count = 0
            for l0 in [lines_vac['OIII'][1], lines_vac['Halpha'][0]]:
                l0_idx = np.searchsorted(data_stack[idx,0,:], l0)  - 1
                if data_stack[idx,5,l0_idx] > 8/data_stack[idx,2,l0_idx]:
                    count += 1
            if count > 0:
                filter_mask[idx] = True

        data_class.subset(filter_mask)
        return data_class

    #
    # Fit spectrum
    #
    def fit_onhs_dz_modified(self, data_class:Spectrum, 
                                 id=None, two_component=False, w_dz=False):
        
        data_stack = data_class.data_stack
        idx = data_class.id2index(id)

        Halpha_crop_region  = [Halpha_rest[0]-30,    SII_rest[1]+ 30]
        OII_crop_region     = [   OII_rest[0]-30,    OII_rest[1]+ 30]
        OIII_crop_region    = [ Hbeta_rest[0]-30,   OIII_rest[1]+ 30]

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
        
        OIII_slice       = (lam >= OIII_crop_region[0]) & (lam <= OIII_crop_region[1])
        OIII_lam         = lam[OIII_slice]
        OIII_flux        = flux[OIII_slice]
        OIII_ivar        = ivar[OIII_slice]
        OIII_sigma       = np.sqrt(1/OIII_ivar)

        combine_lam     = np.concatenate([OII_lam, OIII_lam, Halpha_lam])
        combine_flux    = np.concatenate([OII_flux, OIII_flux, Halpha_flux])
        combine_sigma   = np.concatenate([OII_sigma, OIII_sigma, Halpha_sigma])

        lines_to_fit    = [OII_rest[1], OIII_rest[1], *Hbeta_rest, *Halpha_rest, *NII_rest, *SII_rest]

        if two_component:
            lam0 = lines_to_fit + lines_to_fit
        else:
            lam0 = lines_to_fit
        
        
        oii_ratio   = 1/1.33  # fixed ratio for [OII]3729/3727
        oiii_ratio  = 1/3.0  # fixed ratio for [OIII]5007/4959
        
        def fitting_func(lam_grid, *params):
            
            if two_component:
                if w_dz:
                    dz, sigma_1, sigma_2, dz_r, dz_l = params[:5]
                    amp_start_index = 5
                else:
                    sigma_1, sigma_2, dz_r, dz_l = params[:4]
                    amp_start_index = 4
            else:
                if w_dz:
                    dz, sigma_1 = params[:2]
                    amp_start_index = 2
                else:
                    sigma_1 = params[0]
                    amp_start_index = 1

            gaussian_parms_OII      = []
            gaussian_parms_OIII     = []
            gaussian_parms_Halpha   = []
            for j in range(len(lam0)):
                
                # ADD [OII] 3727
                if (j == 0) or (j == (0 + len(lines_to_fit))):
                    if two_component:
                        amp_j = oii_ratio * params[j+amp_start_index]
                        if j==0:
                            sigma_j = sigma_1
                            if w_dz:
                                lam0_j  = OII_rest[0] * (1 + dz + dz_r)
                            else:
                                lam0_j  = OII_rest[0] * (1 + dz_r)
                        else:
                            sigma_j = sigma_2
                            if w_dz:
                                lam0_j  = OII_rest[0] * (1 + dz + dz_l)
                            else:
                                lam0_j  = OII_rest[0] * (1 + dz_l)
                            
                        gaussian_parms_OII.append((amp_j, lam0_j, sigma_j)) # OII 3727
                    else:
                        amp_j = oii_ratio * params[j+amp_start_index]
                        sigma_j = sigma_1
                        if w_dz:
                            lam0_j  = OII_rest[0] * (1 + dz)
                        else:
                            lam0_j  = OII_rest[0]
                        gaussian_parms_OII.append((amp_j, lam0_j, sigma_j)) # OII 3727
                        
                        
                # ADD [OIII] 4959        
                if (j == 1) or (j == (1 + len(lines_to_fit))):
                    if two_component:
                        amp_j = oiii_ratio * params[j+amp_start_index]
                        if j==0:
                            sigma_j = sigma_1
                            if w_dz:
                                lam0_j  = OII_rest[0] * (1 + dz + dz_r)
                            else:
                                lam0_j  = OII_rest[0] * (1 + dz_r)
                        else:
                            sigma_j = sigma_2
                            if w_dz:
                                lam0_j  = OII_rest[0] * (1 + dz + dz_l)
                            else:
                                lam0_j  = OII_rest[0] * (1 + dz_l)
                            
                        gaussian_parms_OIII.append((amp_j, lam0_j, sigma_j)) # OIII 4959
                    else:
                        amp_j = oiii_ratio * params[j+amp_start_index]
                        sigma_j = sigma_1
                        if w_dz:
                            lam0_j  = OIII_rest[0] * (1 + dz)
                        else:
                            lam0_j  = OIII_rest[0]
                        gaussian_parms_OIII.append((amp_j, lam0_j, sigma_j)) # OIII 4959
                        

                if two_component:
                    amp_j   = params[j+amp_start_index]
                    if j < len(lines_to_fit):
                        sigma_j = sigma_1
                        if w_dz:
                            lam0_j  = lam0[j] * (1 + dz + dz_r)
                        else:
                            lam0_j  = lam0[j] * (1 + dz_r)
                    else:
                        sigma_j = sigma_2
                        if w_dz:
                            lam0_j  = lam0[j] * (1 + dz + dz_l)
                        else:
                            lam0_j  = lam0[j] * (1 + dz_l)
                    
                else:
                    amp_j   = params[j+amp_start_index]
                    sigma_j = sigma_1
                    if w_dz:
                        lam0_j  = lam0[j] * (1 + dz)
                    else:
                        lam0_j  = lam0[j]

                if (j == 0) or (j == (0+ len(lines_to_fit))):
                    gaussian_parms_OII.append((amp_j, lam0_j, sigma_j)) # OII
                elif (j == 1) or (j == (1+ len(lines_to_fit))) or (j == 2) or (j == (2+ len(lines_to_fit))):
                    gaussian_parms_OIII.append((amp_j, lam0_j, sigma_j)) # OIII
                else:
                    gaussian_parms_Halpha.append((amp_j, lam0_j, sigma_j)) # Halpha

            
            lam_OII = lam_grid[:len(OII_lam)]
            lam_OIII = lam_grid[len(OII_lam):len(OII_lam)+len(OIII_lam)]
            lam_Halpha = lam_grid[len(OII_lam)+len(OIII_lam):]

            combine_model = np.concatenate([
                model(   lam_OII,    gaussian_parms=gaussian_parms_OII,), # OII
                model(  lam_OIII,   gaussian_parms=gaussian_parms_OIII,), # OIII
                model(lam_Halpha, gaussian_parms=gaussian_parms_Halpha,) # Halpha
            ])
            return combine_model

        dz_init, dz_upper, dz_lower                     = 0, 1e-3, -1e-3

        sigma_1_init, sigma_1_upper, sigma_1_lower      = 1, 7/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
        if two_component:
            sigma_2_init, sigma_2_upper, sigma_2_lower  = sigma_1_init, sigma_1_upper, sigma_1_lower

        if two_component:
            dz_r_init, dz_r_upper, dz_r_lower              =  1e-6, 1e-3,     0     # right component
            dz_l_init, dz_l_upper, dz_l_lower              = -1e-6,    0, -1e-3     # left component


        amp_init, amp_upper, amp_lower              = [1]*len(lam0), [np.max(combine_flux)]*len(lam0), [0]*len(lam0)

        # oii_ratio_init, oii_ratio_upper, oii_ratio_lower = 0.7518, 0.76, 0.74
        
        if two_component:
            if w_dz:
                p0 = [dz_init, sigma_1_init, sigma_2_init, dz_r_init, dz_l_init] + amp_init
                bounds_lower = [dz_lower, sigma_1_lower, sigma_2_lower, dz_r_lower, dz_l_lower] + amp_lower
                bounds_upper = [dz_upper, sigma_1_upper, sigma_2_upper, dz_r_upper, dz_l_upper] + amp_upper
            else:
                p0 = [sigma_1_init, sigma_2_init, dz_r_init, dz_l_init] + amp_init
                bounds_lower = [sigma_1_lower, sigma_2_lower, dz_r_lower, dz_l_lower] + amp_lower
                bounds_upper = [sigma_1_upper, sigma_2_upper, dz_r_upper, dz_l_upper] + amp_upper

        else:
            if w_dz:
                p0 = [dz_init, sigma_1_init] + amp_init
                bounds_lower = [dz_lower, sigma_1_lower] + amp_lower
                bounds_upper = [dz_upper, sigma_1_upper] + amp_upper
            else:
                p0 = [sigma_1_init] + amp_init
                bounds_lower = [sigma_1_lower] + amp_lower
                bounds_upper = [sigma_1_upper] + amp_upper

        popt, pcov = curve_fit(fitting_func, combine_lam, combine_flux, p0=p0, sigma=combine_sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
        
        params = {}
        if two_component:
            if w_dz:
                dz, sigma_1, sigma_2, dz_r, dz_l = popt[:5]
                amp_start_index = 5
                params['dz'] = dz
            else:
                sigma_1, sigma_2, dz_r, dz_l = popt[:4]
                amp_start_index = 4
                params['dz'] = (dz_r, dz_l)
            params['dz_centroid'] = None
            params['sigma'] = (sigma_1, sigma_2)
            params['dlam'] = (dz_r, dz_l)
            params['conti_params'] = ((0, 0), (0, 0), (0, 0))
            right_comp = []
            left_comp = []
            for k, l_0 in enumerate(lam0):
                # ADD [OII] 3727
                if (k == 0) or (k == (0 + len(lines_to_fit))):
                    if w_dz:
                        if k == 0:
                            right_comp.append((oii_ratio*popt[k+amp_start_index], OII_rest[0]*(1+dz+dz_r), sigma_1))
                        else:
                            left_comp.append((oii_ratio*popt[k+amp_start_index],  OII_rest[0]*(1+dz+dz_l), sigma_2))
                    else:
                        if k == 0:
                            right_comp.append((oii_ratio*popt[k+amp_start_index], OII_rest[0]*(1+dz_r), sigma_1))
                        else:
                            left_comp.append((oii_ratio*popt[k+amp_start_index],  OII_rest[0]*(1+dz_l), sigma_2))

                # ADD [OIII] 4959
                if (k == 1) or (k == (1 + len(lines_to_fit))):
                    if w_dz:
                        if k == 1:
                            right_comp.append((oiii_ratio*popt[k+amp_start_index], OIII_rest[0]*(1+dz+dz_r), sigma_1))
                        else:
                            left_comp.append((oiii_ratio*popt[k+amp_start_index],  OIII_rest[0]*(1+dz+dz_l), sigma_2))
                    else:
                        if k == 1:
                            right_comp.append((oiii_ratio*popt[k+amp_start_index], OIII_rest[0]*(1+dz_r), sigma_1))
                        else:
                            left_comp.append((oiii_ratio*popt[k+amp_start_index],  OIII_rest[0]*(1+dz_l), sigma_2))

                if w_dz:
                    if k < len(lines_to_fit):
                        right_comp.append((popt[k+amp_start_index], l_0*(1+dz+dz_r), sigma_1))
                    else:
                        left_comp.append((popt[k+amp_start_index],  l_0*(1+dz+dz_l), sigma_2))
                else:
                    if k < len(lines_to_fit):
                        right_comp.append((popt[k+amp_start_index], l_0*(1+dz_r), sigma_1))
                    else:
                        left_comp.append((popt[k+amp_start_index],  l_0*(1+dz_l), sigma_2))
            params['right_comp'] = right_comp
            params['left_comp'] = left_comp
            params['gaussian_params'] = right_comp+left_comp
        else:
            if w_dz:
                dz, sigma_1 = popt[:2]
                amp_start_index = 2
                params['dz'] = dz
                params['sigma'] = sigma_1
                params['dlam'] = None
                params['conti_params'] = ((0, 0), (0, 0), (0, 0))
                gaussian_params = []
                for k, l_0 in enumerate(lam0):
                    # ADD [OII] 3727
                    if k == 0:
                        gaussian_params.append((oii_ratio*popt[k+amp_start_index], lines_vac['OII'][0]*(1+dz), sigma_1))
                    # ADD [OIII] 4959
                    if k == 1:
                        gaussian_params.append((oiii_ratio*popt[k+amp_start_index], lines_vac['OIII'][0]*(1+dz), sigma_1))
                    gaussian_params.append((popt[k+amp_start_index], l_0*(1+dz), sigma_1))
                params['gaussian_params'] = gaussian_params
            else:
                sigma_1 = popt[0]
                amp_start_index = 1
                params['dz'] = 0
                params['sigma'] = sigma_1
                params['dlam'] = None
                params['conti_params'] = ((0, 0), (0, 0), (0, 0))
                gaussian_params = []
                for k, l_0 in enumerate(lam0):
                    # ADD [OII] 3727
                    if k == 0:
                        gaussian_params.append((oii_ratio*popt[k+amp_start_index], lines_vac['OII'][0], sigma_1))
                    # ADD [OIII] 4959
                    if k == 1:
                        gaussian_params.append((oiii_ratio*popt[k+amp_start_index], lines_vac['OIII'][0], sigma_1))
                    gaussian_params.append((popt[k+amp_start_index], l_0, sigma_1))
                params['gaussian_params'] = gaussian_params
        
        return params, (combine_lam, combine_flux, combine_sigma), (len(OII_lam), len(OIII_lam))
    
    
    def fit_z(self, data_class:Spectrum,
               id=None,):
        try:
            # params_1comp, (combine_lam, combine_flux, combine_sigma), seperation = self.fit_onhs_dz_OII_version(data_class, id=id, two_component=False)
            params_1comp, (combine_lam, combine_flux, combine_sigma), seperation = self.fit_onhs_dz_modified(data_class, id=id, two_component=False)

            conti_1, conti_2    = params_1comp['conti_params']
            conti_1comp         = np.concatenate([model(combine_lam[:seperation], conti_parms=conti_1), 
                                                model(combine_lam[seperation:], conti_parms=conti_2)])
            combine_flux_1comp  = combine_flux - conti_1comp
            model_1comp         = np.concatenate([model(combine_lam[:seperation], gaussian_parms=params_1comp['gaussian_params']), 
                                                model(combine_lam[seperation:], gaussian_parms=params_1comp['gaussian_params'])])
            chisq_1comp         = np.sum(((combine_flux_1comp - model_1comp)/combine_sigma)**2)
            dof_1comp           = len(combine_lam) - (7+6-4)
            # 7: dz, sigma_1, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio
            # 6: amp_OII, amp_Halpha, amp_NII*2, amp_SII*2
            # -4: conti_a_1, conti_b_1, conti_a_2, conti_b_2
            
            criteria = 3 * combine_sigma
            count = 0
            for comp in params_1comp['gaussian_params']:
                amp = comp[0]
                l0 = comp[1]
                l0_idx = np.searchsorted(combine_lam, l0)
                if amp > criteria[l0_idx]:
                    count += 1
                else:
                    continue
            if count >= 2: # significant detection
                pass
            else: # noise
                return f'noise_{count}' 
        except: # one comp fitting failed
            return '1_comp_fit_failed' 
        
        
        try:    
            # params_wdz, (combine_lam, combine_flux, combine_sigma), seperation = self.fit_onhs_dz_OII_version(data_class, id=id, two_component=True, w_dz=True)
            params_wdz, (combine_lam, combine_flux, combine_sigma), seperation = self.fit_onhs_dz_OII_modified(data_class, id=id, two_component=True, w_dz=True)

            conti_1, conti_2    = params_wdz['conti_params']
            conti               = np.concatenate([model(combine_lam[:seperation], conti_parms=conti_1), 
                                                model(combine_lam[seperation:], conti_parms=conti_2)])
            combine_flux_wdz    = combine_flux - conti
            left_wdz            = np.concatenate([model(combine_lam[:seperation], gaussian_parms=params_wdz['left_comp'][:2]), 
                                                model(combine_lam[seperation:], gaussian_parms=params_wdz['left_comp'][2:])])
            right_wdz           = np.concatenate([model(combine_lam[:seperation], gaussian_parms=params_wdz['right_comp'][:2]), 
                                                model(combine_lam[seperation:], gaussian_parms=params_wdz['right_comp'][2:])])
            model_wdz           = left_wdz + right_wdz
            chisq_wdz           = np.sum(((combine_flux_wdz - model_wdz)/combine_sigma)**2)
            dof_2comp           = len(combine_lam) - (10+6*2-4)
            # 10: dz, sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio
            # 6*2: (amp_OII, amp_Halpha, amp_NII*2, amp_SII*2)*2 two components
            # -4: conti_a_1, conti_b_1, conti_a_2, conti_b_2

            F_stat = ((chisq_1comp - chisq_wdz) / (dof_1comp - dof_2comp)) / (chisq_wdz / dof_2comp)
            p_value = 1 - f.cdf(F_stat, dof_1comp - dof_2comp, dof_2comp)

            if p_value < 0.05: # two comp with w_dz is better
                return params_wdz
            else: # one comp is enough
                return params_1comp
            
        except: # two comp with w_dz fitting failed
            try:
                # params_free, (combine_lam, combine_flux, combine_sigma), seperation = self.fit_onhs_dz_OII_version(data_class, id=id, two_component=True, w_dz=False)
                params_free, (combine_lam, combine_flux, combine_sigma), seperation = self.fit_onhs_dz_OII_modified(data_class, id=id, two_component=True, w_dz=False)

                conti_1, conti_2    = params_free['conti_params']
                conti               = np.concatenate([model(combine_lam[:seperation], conti_parms=conti_1), 
                                                    model(combine_lam[seperation:], conti_parms=conti_2)])
                combine_flux_free   = combine_flux - conti
                left_free           = np.concatenate([model(combine_lam[:seperation], gaussian_parms=params_free['left_comp'][:2]), 
                                                    model(combine_lam[seperation:], gaussian_parms=params_free['left_comp'][2:])])
                right_free          = np.concatenate([model(combine_lam[:seperation], gaussian_parms=params_free['right_comp'][:2]), 
                                                    model(combine_lam[seperation:], gaussian_parms=params_free['right_comp'][2:])])
                model_free          = left_free + right_free
                chisq_free          = np.sum(((combine_flux_free - model_free)/combine_sigma)**2)
                dof_free            = len(combine_lam) - (9+6*2-4)
                # 9: sigma_1, sigma_2, dz_r, dz_l, conti_a_1, conti_b_1, conti_a_2, conti_b_2, oii_ratio
                # 6*2: (amp_OII, amp_Halpha, amp_NII*2, amp_SII*2)*2 two components
                # -4: conti_a_1, conti_b_1, conti_a_2, conti_b_2

                F_stat = ((chisq_1comp - chisq_free) / (dof_1comp - dof_free)) / (chisq_free / dof_free)
                p_value = 1 - f.cdf(F_stat, dof_1comp - dof_free, dof_free)
                if p_value < 0.05: # two comp with free dlam is better
                    left_area = np.sum(left_free)
                    right_area = np.sum(right_free)
                    dz1, dz2 = params_free['dz']
                    dz_centroid = (dz2*left_area + dz1*right_area)/(left_area+right_area)
                    params_free['dz_centroid'] = dz_centroid
                    return params_free
                else: # one comp is enough
                    return params_1comp
            except: # two comp with free dlam fitting failed
                return params_1comp



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
