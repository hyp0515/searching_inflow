import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import requests
from pathlib import Path
import warnings
from multiprocessing import Pool, cpu_count
import functools
import tqdm

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
        
        
        
        # Match target IDs between spectra and color catalogs
        targetID_spectra = spectra_data[1].data[:]['TARGETID'] # spectra catalog has 100638 spectra
        targetID_color    = color_data[1].data[:]['TARGETID'] # color catalog has 100642 spectra
        rearranged_indices = np.searchsorted(targetID_color, targetID_spectra)
        targetID_color = targetID_color[rearranged_indices]

        self.targetID       = np.array(targetID_spectra)
        self.z_pipe         = np.array(spectra_data[1].data['Z'])
        self.z              = self.z_pipe.copy()
        self.RA             = spectra_data[1].data['RA']
        self.DEC            = spectra_data[1].data['DEC']

        self.coadd_data     = np.take(spectra_data[2].data, 0, axis=1)
        self.ivar           = np.take(spectra_data[3].data, 0, axis=1)
        self.mask           = np.take(spectra_data[4].data, 0, axis=1)

        self.spectype      = color_data[1].data['SPECTYPE'][rearranged_indices]
        
        g_flux = color_data[1].data['FLUX_G']
        r_flux = color_data[1].data['FLUX_R']
        z_flux = color_data[1].data['FLUX_Z']
        w1_flux = color_data[1].data['FLUX_W1']
        w2_flux = color_data[1].data['FLUX_W2']
        color_flux = np.vstack([g_flux, r_flux, z_flux, w1_flux, w2_flux]).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.color_mag = (22.5 - 2.5 * np.log10(color_flux))[rearranged_indices, :]
        
        self.n_spectra      = len(spectra_data[1].data)
        self.adjust_z_mode   = [[] for _ in range(self.n_spectra)]
        self.searched_NaD    = [[] for _ in range(self.n_spectra)]
        self.target_label    = [[] for _ in range(self.n_spectra)]
        self.smoothed_flux = smooth_spectrum(self.coadd_data, sigma=1)
    
    def shrink_dataset(self, indices):
        self.n_spectra      = len(self.targetID[::indices])
        self.targetID       = self.targetID[::indices]
        self.z_pipe         = self.z_pipe[::indices]
        self.z              = self.z[::indices]
        self.RA             = self.RA[::indices]
        self.DEC            = self.DEC[::indices]
        self.coadd_data     = self.coadd_data[::indices]
        self.ivar           = self.ivar[::indices]
        self.mask           = self.mask[::indices]
        self.color_mag      = self.color_mag[::indices]
        self.smoothed_flux  = self.smoothed_flux[::indices]

    #
    # Translate targetID to index
    #
    def id2index(self, targetID):
        """
        Convert targetID to index in the dataset.
        """
        try:
            index = np.where(self.targetID == targetID)[0][0]
            return index
        except IndexError:
            print(f"targetID {targetID} not found in the dataset.")
            return None
    
    #
    # Color criteria
    #
    def color_criteria(self, criterion='g-r>=0.85', exclude=True):
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
        if exclude:
            criteria = ~criteria
        return criteria

    def subtype_criteria(self, subtype='QSO', exclude=True):
        
        if subtype.upper() not in ['QSO', 'LRG', 'ELG', 'BGS', 'MWS']:
            print(f"Subtype '{subtype}' not recognized. Available subtypes: ['QSO', 'LRG', 'ELG', 'BGS', 'MWS']")
            return np.array([False] * self.n_spectra)
        
        # simple vectorized comparison
        is_subtype = (self.spectype == subtype.upper())

        # # or explicitly with numpy
        # is_subtype = np.equal(self.spectype, subtype.upper())

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
        if i is not None:
            self.target_label[i].append(label_type)
        elif criteria is not None:
            self.target_label[criteria].append(label_type)

    def clean_label(self, label=None):
        if (label is None):
            self.target_label = [[] for _ in range(self.n_spectra)]
        elif (label is not None):
            for j in range(self.n_spectra):
                if label in self.target_label[j]:
                    self.target_label[j].remove(label)

    def label_filter(self, label=['QSO', 'NaD'], exclude=True):
        
        if label is None:
            print("Please specify the label to filter.")
            print("Available labels: ", ['QSO', 'LRG', 'ELG', 'BGS', 'MWS'] + list(lines_air.keys()))
            print("Dataset remains unfiltered.")
            return
        elif any(lbl not in ['QSO', 'LRG', 'ELG', 'BGS', 'MWS'] + list(lines_air.keys()) for lbl in label):
            print(f"Label '{label}' not recognized. Available labels: {['QSO', 'LRG', 'ELG', 'BGS', 'MWS'] + list(lines_air.keys())}")
            print("Dataset remains unfiltered.")
            return

        is_label = []
        for i in range(self.n_spectra):
            is_label.append(all(lbl in self.target_label[i] for lbl in label))

        is_label = np.array(is_label)
        if exclude:
            is_label = ~is_label
            
        self.n_spectra      = len(self.targetID[is_label])
        self.targetID       = self.targetID[is_label]
        self.z_pipe         = self.z_pipe[is_label]
        self.z              = self.z[is_label]
        self.RA             = self.RA[is_label]
        self.DEC            = self.DEC[is_label]
        self.coadd_data     = self.coadd_data[is_label][:]
        self.ivar           = self.ivar[is_label][:]
        self.mask           = self.mask[is_label][:]
        self.target_label   = [lbls for j, lbls in enumerate(self.target_label) if is_label[j]]
    
    def subset(self, criteria):
        self.n_spectra      = len(self.targetID[criteria])
        self.targetID       = self.targetID[criteria]
        self.z_pipe         = self.z_pipe[criteria]
        self.z              = self.z[criteria]
        self.RA             = self.RA[criteria]
        self.DEC            = self.DEC[criteria]
        self.coadd_data     = self.coadd_data[criteria][:]
        self.ivar           = self.ivar[criteria][:]
        self.mask           = self.mask[criteria][:]
        

class FitSpectrum:
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
        self.adjust_z_mode = data_class.adjust_z_mode
        self.searched_NaD  = data_class.searched_NaD
        self.spectype      = data_class.spectype
        self.color_mag     = data_class.color_mag
        self.id2index      = data_class.id2index
    #
    # Fit spectrum
    #    
    def fit_flux(self, 
                 i=None, 
                 fitting_model=model, 
                 lam0=None, 
                 region=None, 
                 fit_z=True,
                 two_component=False, 
                 e_or_a='e',):

        ##############################
        """
        Data to fit
        """
        lam = desi_wavelength.copy()
        
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
                        lam0_j = lam0[j] * (1 + z + dz) # fixed components

                        if j >= len(fit_line):
                            dlam = params[-3]
                            amp_j, sigma_j = params[j+3], sigma_2
                            lam0_j = (lam0[j]+dlam) * (1 + z + dz) # free components
                    else:
                        amp_j, sigma_j = params[j+2], sigma_1
                        lam0_j = lam0[j] * (1 + z + dz)
                    gaussian_parms.append((amp_j, lam0_j, sigma_j))
            else:
                sigma_1 = params[0]
                if two_component:
                    sigma_2 = params[1]
                for j in range(len(lam0)):
                    if two_component:
                        amp_j, sigma_j = params[j+2], sigma_1
                        lam0_j = lam0[j] * (1 + z) # fixed components
                        if j >= len(fit_line):
                            dlam = params[-3]
                            amp_j, sigma_j = params[j+2], sigma_2
                            lam0_j = (lam0[j]+dlam) * (1 + z) # free components
                    else:
                        offset = params[-3] # system offset
                        amp_j, sigma_j = params[j+1], sigma_1
                        lam0_j = (lam0[j] + offset) * (1 + z)

                    gaussian_parms.append((amp_j, lam0_j, sigma_j))
            
            conti_a = params[-2]
            conti_b = params[-1]
            conti_parms = (conti_a, conti_b)
            
            return fitting_model(lam, gaussian_parms=gaussian_parms, conti_parms=conti_parms)
        
        
        if i is not None:
            flux = self.coadd_data[i]
            ivar = self.ivar[i]
            z = self.z[i]
            
            if region is None:
                region = (np.min(lam0)*(1+z)- 50, np.max(lam0)*(1+z) + 50)

            crop_region = (lam >= region[0]) & (lam <= region[1])
            lam = lam[crop_region]
            flux = flux[crop_region]
            ivar = ivar[crop_region]
            mask = self.mask[i][crop_region]
            
            good = (mask == 0)
            lam = lam[good]
            flux = flux[good]
            ivar = ivar[good]
            sigma = np.sqrt(1 / ivar)

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
                sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 10/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
                sigma_2_init, sigma_2_upper, sigma_2_lower  = 1, 10/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
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
                sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 10/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
                amp_init, amp_upper, amp_lower              = [1]*len(lam0), [np.inf]*len(lam0), [0]*len(lam0)
                
                p0 = [dz_init, sigma_1_init] + amp_init + [conti_a_init, conti_b_init]
                bounds_lower = [dz_lower, sigma_1_lower] + amp_lower + [conti_a_lower, conti_b_lower]
                bounds_upper = [dz_upper, sigma_1_upper] + amp_upper + [conti_a_upper, conti_b_upper]
                
            elif (fit_z is False) and (two_component is True):
                """
                Fit two components without redshift adjustment
                len(p0) = 2 + n_fit_line * 2 + 3
                """
                sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 10/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
                sigma_2_init, sigma_2_upper, sigma_2_lower  = 1, 10/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
                dlam_init, dlam_upper, dlam_lower           = 0, 6, -6
                if e_or_a == 'a':
                    amp_init, amp_upper, amp_lower          = [-1]*len(lam0), [0]*len(lam0), [-np.inf]*len(lam0)
                else:
                    amp_init, amp_upper, amp_lower          = [1]*len(lam0), [np.inf]*len(lam0), [0]*len(lam0)
                
                p0 = [sigma_1_init, sigma_2_init] + amp_init + [dlam_init, conti_a_init, conti_b_init]
                bounds_lower = [sigma_1_lower, sigma_2_lower] + amp_lower + [dlam_lower, conti_a_lower, conti_b_lower]
                bounds_upper = [sigma_1_upper, sigma_2_upper] + amp_upper + [dlam_upper, conti_a_upper, conti_b_upper]
            
            elif (fit_z is False) and (two_component is False):
                """
                Fit one component without redshift adjustment
                len(p0) = 1 + n_fit_line + 3
                """
                sigma_1_init, sigma_1_upper, sigma_1_lower  = 1, 10/(2*np.sqrt(2*np.log(2))), 2/(2*np.sqrt(2*np.log(2)))
                offset_init, offset_upper, offset_lower     = 0, 6, -6
                if e_or_a == 'a':
                    amp_init, amp_upper, amp_lower          = [-1]*len(lam0), [0]*len(lam0), [-np.inf]*len(lam0)
                else:
                    amp_init, amp_upper, amp_lower          = [1]*len(lam0), [np.inf]*len(lam0), [0]*len(lam0)

                p0 = [sigma_1_init] + amp_init + [offset_init, conti_a_init, conti_b_init]
                bounds_lower = [sigma_1_lower] + amp_lower + [offset_lower, conti_a_lower, conti_b_lower]
                bounds_upper = [sigma_1_upper] + amp_upper + [offset_upper, conti_a_upper, conti_b_upper]

            try:
                popt, pcov = curve_fit(fitting_func, lam, flux, p0=p0, sigma=sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
            except:
                popt = None

            return popt
            
        elif i is None:
            fit_params = [[] for _ in range(self.n_spectra)]
            for k in range(self.n_spectra):
                popt = self.fit_flux(i=k, fitting_model=fitting_model, lam0=lam0, region=region, fit_z=fit_z)
                fit_params[k] = popt
            return fit_params

    def adjust_z(self, i, mode='base_2'):
        
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
            region = (np.min(fit_line)*(1+self.z_pipe[i])- 50, np.max(fit_line)*(1+self.z_pipe[i]) + 50)
            try:
                popt = self.fit_flux(i=i, fitting_model=model, lam0=fit_line, region=region, fit_z=True, two_component=two_comp)

            except Exception as e:
                print(f"Error fitting flux for {self.targetID[i]}: {e}")
                return 0

            lam = desi_wavelength.copy()
            flux = self.coadd_data[i]
            mask = self.mask[i]
            
            crop_region = (lam >= region[0]) & (lam <= region[1])
            lam = lam[crop_region]
            flux = flux[crop_region]
            mask = mask[crop_region]
            good = (mask == 0)
            lam = lam[good]
            flux = flux[good]
            
            if popt is not None:
                if two_comp:
                    dz, sigma_1, sigma_2, dlam, conti_a, conti_b = popt[0], popt[1], popt[2], popt[-3], popt[-2], popt[-1]
                    if (np.abs(dlam) < 0.8) and (np.abs(dz) < 1e-4):
                        return 0
                    z = self.z_pipe[i]
                    fixed_comp = []
                    free_comp  = []
                    for k in range(len(fit_line)):
                        fixed_comp.append((popt[k+3], fit_line[k]*(1+z+dz), sigma_1))
                        free_comp.append((popt[(k+len(fit_line))+3], (fit_line[k]+dlam)*(1+z+dz), sigma_2))
                    fitted_model = model(lam, gaussian_parms=(fixed_comp+free_comp), conti_parms=(conti_a, conti_b))
                    
                else:
                    dz, sigma_1, conti_a, conti_b = popt[0], popt[1], popt[-2], popt[-1]
                    if (np.abs(dz) < 1e-4):
                        return 0
                    z = self.z_pipe[i]
                    offset_comp = []
                    for k in range(len(fit_line)):
                        offset_comp.append((popt[k+2], fit_line[k]*(1+z+dz), sigma_1))
                    fitted_model = model(lam, gaussian_parms=offset_comp, conti_parms=(conti_a, conti_b))
                    
                noise = np.std(flux - fitted_model)
                count = 0
                for j in range(len(fit_line)):
                    if two_comp:
                        if min(popt[(j+len(fit_line))+3], popt[j+3]) > 3*noise:
                            count += 1
                        
                    else:
                        if popt[j+2] > 3*noise:
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
            if (dz is None) or (np.abs(dz) < 1e-4):
                mode = 'no'
                dz = 0
        elif mode == 'base_2' or mode == 'OII_2':
            fit_line, two_comp = modes[mode]
            dz = get_z(fit_line, two_comp)
            if np.abs(dz) < 1e-4:
                mode = mode.replace('_2', '')
                fit_line, two_comp = modes[mode]
                dz = get_z(fit_line, two_comp)
                if np.abs(dz) < 1e-4:
                    mode = 'no'
                    dz = 0
        elif mode == 'base' or mode == 'OII':
            fit_line, two_comp = modes[mode]
            dz = get_z(fit_line, two_comp)
            if np.abs(dz) < 1e-4:
                mode = 'no'
                dz = 0
                
        elif mode == 'no':
            dz = 0

        self.adjust_z_mode[i] = mode
        self.z[i] = self.z_pipe[i] + dz
        return dz
    
    def search_NaD(self, i, two_component=True):

        NaD_rest = lines_vac['NaD']

        region = (np.min(NaD_rest)*(1+self.z_pipe[i])- 50, np.max(NaD_rest)*(1+self.z_pipe[i]) + 50)
        
        lam = desi_wavelength.copy()
        flux = self.coadd_data[i]
        mask = self.mask[i]
        crop_region = (lam >= region[0]) & (lam <= region[1])
        lam = lam[crop_region]
        flux = flux[crop_region]
        mask = mask[crop_region]
        good = (mask == 0)
        lam = lam[good]
        flux = flux[good]
        
        z = self.z[i]
        
        if two_component:
            try:
                popt = self.fit_flux(i=i, fitting_model=model, lam0=[*NaD_rest], region=region, fit_z=False, two_component=True, e_or_a='a')
                sigma_1, sigma_2, dlam, conti_a, conti_b = popt[0], popt[1], popt[-3], popt[-2], popt[-1]
                amp_D1, amp_D2 = popt[2], popt[3]
                amp_D1_free, amp_D2_free = popt[4], popt[5]
                if np.abs(dlam) < 0.8:
                    # print(f"Small dlam in two-component fit for NaD in {self.targetID[i]}, trying one-component fit.")
                    return self.search_NaD(i=i, two_component=False)
                
                fixed_comp = [(amp_D1, NaD_rest[0]*(1+z), sigma_1), (amp_D2, NaD_rest[1]*(1+z), sigma_1)]
                free_comp  = [(amp_D1_free, (NaD_rest[0]+dlam)*(1+z), sigma_2), (amp_D2_free, (NaD_rest[1]+dlam)*(1+z), sigma_2)]
                
                fitted_model = model(lam, gaussian_parms=(fixed_comp+free_comp), conti_parms=(conti_a, conti_b))
                noise = np.std(flux - fitted_model)
                
                # count = 0
                # for j in range(len(NaD_rest)):
                #     if min(max(popt[1], popt[3]), max(popt[2], popt[4])) > 3*noise:
                #         count += 1
                if min(min(np.abs(amp_D1), np.abs(amp_D1_free)), min(np.abs(amp_D2), np.abs(amp_D2_free))) > 3*noise:
                    if dlam > 0.8:
                        self.searched_NaD[i] = 'inflow_2'
                    elif dlam < -0.8:
                        self.searched_NaD[i] = 'outflow_2'
                    else:
                        self.searched_NaD[i] = 'systemic_2'
                    return popt
                else:
                    return self.search_NaD(i=i, two_component=False)
            except:
                # print(f"Two-component fit failed for NaD in {self.targetID[i]}, trying one-component fit.")
                return self.search_NaD(i=i, two_component=False)
        else:
            try:
                popt = self.fit_flux(i=i, fitting_model=model, lam0=[*NaD_rest], region=region, fit_z=False, two_component=False, e_or_a='a')
                sigma_1, offset, conti_a, conti_b = popt[0], popt[-3], popt[-2], popt[-1]
                amp_D1, amp_D2 = popt[1], popt[2]
                
                offset_comp = [(amp_D1, (NaD_rest[0]+offset)*(1+z), sigma_1), (amp_D2, (NaD_rest[1]+offset)*(1+z), sigma_1)]
                fitted_model = model(lam, gaussian_parms=offset_comp, conti_parms=(conti_a, conti_b))
                noise = np.std(flux - fitted_model)
                
                # count = 0
                # for j in range(len(NaD_rest)):
                #     if popt[j+1] > 3*noise:
                #         count += 1
                if min(np.abs(amp_D1), np.abs(amp_D2)) > 3*noise:
                    if offset > 0.8:
                        self.searched_NaD[i] = 'inflow'
                    elif offset < -0.8:
                        self.searched_NaD[i] = 'outflow'
                    else:
                        self.searched_NaD[i] = 'systemic'
                    return popt
                else:
                    self.searched_NaD[i] = 'no_detection'
                    return None
            except Exception as e:
                print(f"Error occurred while searching NaD for {self.targetID[i]}: {e}")
                self.searched_NaD[i] = 'no_detection'
                return None

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
    
    def step_spectrum(self, i, range=None, show=True, save=True, fname=None, **kwargs):
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
    
    def plot_model(self):
        
        return