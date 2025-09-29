import numpy as np
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import requests
import json
from pathlib import Path

c = const.c.cgs.value * 1e-5  # speed of light in km/s
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

def check_bits(ID, bit):
    """
    Target bits from DESI:
    1. BGS: bit 60
    2. LRG: bit 0
    3. ELG: bit 1
    4. QSO: bit 2
    5. MWS: bit 61
    6. Secondary Targets: bit 62
    """
    
    val = (2**bit)
    res = ID & val != 0
    return (res)

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

# def fit_flux(lam, flux, ivar, z, fitting_model=model, lam0=None, region=None):
    

    
#     if region is not None:
#         crop_region = (lam >= region[0]) & (lam <= region[1])
#         lam = lam[crop_region]
#         flux = flux[crop_region]
#         ivar = ivar[crop_region]
#     sigma = np.sqrt(1 / ivar)
    
    
    
    

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



class Spectrum:
    
    def __init__(self, fits_data):
        self.fits_data = fits_data
        
        self.n_spectra      = len(fits_data[1].data)
        self.targetID       = np.array(fits_data[1].data[:]['TARGETID'])
        self.z_pipe         = np.array(fits_data[1].data[:]['Z'])
        self.RA             = np.array(fits_data[1].data[:]['RA'])
        self.DEC            = np.array(fits_data[1].data[:]['DEC'])
        self.coadd_data     = np.array(fits_data[2].data)[:, 0, :]
        self.ivar           = np.array(fits_data[3].data)[:, 0, :]
        self.mask           = np.array(fits_data[4].data)[:, 0, :]
        self.subtype        = 'ALL' 
    
    def add_attributes(self, attr_names, attr_values):
        for name, value in zip(attr_names, attr_values):
            setattr(self, name, value)

    def del_attributes(self, attr_name):
        if hasattr(self, attr_name):
            delattr(self, attr_name)
        else:
            print(f"Attribute '{attr_name}' not found.")
    
    def subtype_filter(self, subtype='QSO', exclude=True):

        subtype_dict = {
            'QSO': 2,
            'LRG': 0,
            'ELG': 1,
            'BGS': 60,
            'MWS': 61,
        }
        
        if subtype is None:
            print("Please specify the subtype to filter.")
            print("Available subtypes: ", list(subtype_dict.keys()))
            print("Dataset remains unfiltered.")
        if subtype.upper() not in subtype_dict.keys():
            print(f"Subtype '{subtype}' not recognized. Available subtypes: {list(subtype_dict.keys())}")
            print("Dataset remains unfiltered.")
        
        is_subtype = check_bits(self.targetID, subtype_dict[subtype.upper()])
        
        if exclude:
            self.n_spectra      = len(self.targetID[~is_subtype])
            self.targetID       = self.targetID[~is_subtype]
            self.z_pipe         = self.z_pipe[~is_subtype]
            self.RA             = self.RA[~is_subtype]
            self.DEC            = self.DEC[~is_subtype]
            self.coadd_data     = self.coadd_data[~is_subtype][:]
            self.ivar           = self.ivar[~is_subtype][:]
            self.mask           = self.mask[~is_subtype][:]
            self.subtype        += f" w/o {subtype.upper()}"

    def mask_spectrum(self, i=None):
        if i is not None:
            good = (self.mask[i] == 0)
            self.coadd_data[i][~good] = np.nan
            self.ivar[i][~good] = np.nan
            if hasattr(self, 'smoothed_flux'):
                self.smoothed_flux[i][~good] = np.nan
        else:
            good = (self.mask == 0)
            self.coadd_data[~good] = np.nan
            self.ivar[~good] = np.nan
            if hasattr(self, 'smoothed_flux'):
                self.smoothed_flux[~good] = np.nan

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
        
    def fit_flux(self, i=None, fitting_model=model, lam0=None, region=None, fit_z=True):
        
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
        if fit_z:
            # If fit_z is True, dz + Gaussian (amp, sigma) + continuum (a, b) 
            n_params = 1 + 2 * len(lam0) + 2  
        else:
            # If fit_z is False, Gaussian (amp, lam0, sigma) + continuum (a, b)
            n_params = 3 * len(lam0) + 2
            
        self.fit_params = np.zeros((self.n_spectra, n_params))
        ##############################
        
        def fitting_func(lam, *params):
            gaussian_parms = []
            if fit_z:
                for j in range(len(lam0)):
                    amp = params[2*j+1]
                    lam0_j = lam0[j] * (1 + z_pipe + params[0])
                    sigma_j = params[2*j + 2]
                    gaussian_parms.append((amp, lam0_j, sigma_j))
            else:
                for j in range(len(lam0)):
                    amp = params[3*j]
                    lam0_j = params[3*j + 1]
                    sigma_j = params[3*j + 2]
                    gaussian_parms.append((amp, lam0_j, sigma_j))
            conti_a = params[-2]
            conti_b = params[-1]
            conti_parms = (conti_a, conti_b)
            
            return fitting_model(lam, gaussian_parms=gaussian_parms, conti_parms=conti_parms)
        
        
        if i is not None:
            flux = self.coadd_data[i]
            ivar = self.ivar[i]
            z_pipe = self.z_pipe[i]
            
            
            if region is not None:
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
            p0 = []
            if fit_z:
                p0.append(0) # dz
                for j in range(len(lam0)):
                    p0.append(1) # Initial guess for amplitude
                    p0.append(1)  # Initial guess for sigma
            else:
                for j in range(len(lam0)):
                    p0.append(1)  # Initial guess for amplitude
                    p0.append(lam0[j] * (1 + z_pipe))  # Initial guess for lam0
                    p0.append(1)  # Initial guess for sigma
                # Initial guess for continuum parameters
            p0.extend([0.0,5.0])  # conti_a, conti_b
            
            # Bound
            bounds_lower = []
            bounds_upper = []
            if fit_z:
                bounds_lower.append(-0.01)  # dz lower bound
                bounds_upper.append(+0.01)  # dz upper bound
                for j in range(len(lam0)):
                    bounds_lower.append(0) # Amp lower bound
                    bounds_lower.append(2/(2*np.sqrt(2*np.log(2))))  # fwhm > 2

                    bounds_upper.append(+100) # Amp lower bound
                    bounds_upper.append(10/(2*np.sqrt(2*np.log(2)))) # fwhm < 10
            else:
                for j in range(len(lam0)):
                    bounds_lower.append(-np.inf) # Amp lower bound
                    bounds_lower.append(lam0[j] * (1 + z_pipe) - 5)  # lam0 lower bound
                    bounds_lower.append(2/(2*np.sqrt(2*np.log(2))))  # fwhm > 2
                    
                    bounds_upper.append(+np.inf) # Amp upper bound
                    bounds_upper.append(lam0[j] * (1 + z_pipe) + 5)  # lam0 upper bound
                    bounds_upper.append(10/(2*np.sqrt(2*np.log(2)))) # fwhm < 10
            bounds_lower.extend([-np.inf, 0])  # conti_a, conti_b lower bounds
            bounds_upper.extend([np.inf, np.inf])    # conti_a, conti_b upper bounds

            # Fit the model to the data
            try:
                popt, pcov = curve_fit(fitting_func, lam, flux, p0=p0, sigma=sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
                self.fit_params[i, :] = popt
                return popt
            except Exception as e:
                print(f"Error fitting flux for spectrum {self.targetID[i]}: {e}")
                # return np.nan

        elif i is None:
            for k in range(self.n_spectra):
                popt = self.fit_flux(i=k, fitting_model=fitting_model, lam0=lam0, region=region, fit_z=fit_z)
                self.fit_params[k, :] = popt