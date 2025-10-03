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


def chisq(observed, model, ivar):
    sigma = np.sqrt(1 / ivar)
    residuals = observed - model
    chi_squared = np.sum((residuals ** 2) / (sigma ** 2))
    return chi_squared

################################################################################################################
class Spectrum:
    
    def __init__(self, fits_data):
        self.fits_data = fits_data
        
        self.n_spectra      = len(fits_data[1].data)
        self.targetID       = np.array(fits_data[1].data[:]['TARGETID'])
        self.z_pipe         = np.array(fits_data[1].data[:]['Z'])
        self.z              = self.z_pipe.copy()
        self.RA             = np.array(fits_data[1].data[:]['RA'])
        self.DEC            = np.array(fits_data[1].data[:]['DEC'])
        self.coadd_data     = np.array(fits_data[2].data)[:, 0, :]
        self.ivar           = np.array(fits_data[3].data)[:, 0, :]
        self.mask           = np.array(fits_data[4].data)[:, 0, :]
        self.data_type      = 'ALL' 
        self.adjust_z_mode   = [[] for _ in range(self.n_spectra)]
        self.line_detections = [[] for _ in range(self.n_spectra)]
        self.target_label    = [[] for _ in range(self.n_spectra)]
        self.subtype        = 'Original'
        self.smooth_spectrum()
    
    def smooth_spectrum(self):
        self.smoothed_flux = smooth_spectrum(self.coadd_data, sigma=1)
        
    def add_attributes(self, attr_names, attr_values):
        for name, value in zip(attr_names, attr_values):
            setattr(self, name, value)

    def del_attributes(self, attr_name):
        if hasattr(self, attr_name):
            delattr(self, attr_name)
        else:
            print(f"Attribute '{attr_name}' not found.")
    
    def shrink_dataset(self, indices):
        self.n_spectra      = len(self.targetID[::indices])
        self.targetID       = self.targetID[::indices]
        self.z_pipe         = self.z_pipe[::indices]
        self.z              = self.z[::indices]
        self.RA             = self.RA[::indices]
        self.DEC            = self.DEC[::indices]
        self.coadd_data     = self.coadd_data[::indices][:]
        self.ivar           = self.ivar[::indices][:]
        self.mask           = self.mask[::indices][:]
        self.smoothed_flux  = self.smoothed_flux[::indices][:]
        self.target_label   = self.target_label[::indices][:]
        self.line_detections= self.line_detections[::indices][:]
    
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
    # Add label to the spectrum
    #
    def add_label(self, i=None, label_type='QSO', label_at=None):
        """Add a label to the spectrum.

        Args:
            i (int, optional): Index of the spectrum. Defaults to None.
            label_type (str, optional): Type of label to add. Defaults to 'QSO'.
            label_at (int, optional): Index to add the label at. Defaults to None. This is only used when label_type is uncategorized.
        """
        if label_type in ['QSO', 'LRG', 'ELG', 'BGS', 'MWS']:
            subtype_dict = {
                'QSO': 2,
                'LRG': 0,
                'ELG': 1,
                'BGS': 60,
                'MWS': 61,
            }
            if i is not None:
                is_subtype = check_bits(self.targetID[i], subtype_dict[label_type.upper()])
                if is_subtype:
                    self.target_label[i].append(label_type)
            elif i is None:
                is_subtype = check_bits(self.targetID, subtype_dict[label_type.upper()])
                for j in range(self.n_spectra):
                    if is_subtype[j]:
                        self.target_label[j].append(label_type)
        elif label_type in list(lines_air.keys()):
            is_line_detected = label_type in self.line_detections[i]
            if is_line_detected:
                self.target_label[i].append(label_type)
        else:
            self.target_label[label_at].append(label_type)

    def clean_label(self, i=None, label=None):
        if (i is None) and (label is None):
            self.target_label = [[] for _ in range(self.n_spectra)]
        elif (i is not None) and (label is None):
            self.target_label[i] = []
        elif (i is None) and (label is not None):
            for j in range(self.n_spectra):
                if label in self.target_label[j]:
                    self.target_label[j].remove(label)
        elif (i is not None) and (label is not None):
            if label in self.target_label[i]:
                self.target_label[i].remove(label)
    
    def label_filter(self, label=['QSO', 'NaD'], exclude=True):
        
        if label is None:
            print("Please specify the label to filter.")
            print("Available labels: ", ['QSO', 'LRG', 'ELG', 'BGS', 'MWS'] + list(lines_air.keys()))
            print("Dataset remains unfiltered.")
            return
        elif any(lbl not in ['QSO', 'LRG', 'ELG', 'BGS', 'MWS', 'test'] + list(lines_air.keys()) for lbl in label):
            print(f"Label '{label}' not recognized. Available labels: {['QSO', 'LRG', 'ELG', 'BGS', 'MWS'] + list(lines_air.keys())}")
            print("Dataset remains unfiltered.")
            return

        is_label = []
        for i in range(self.n_spectra):
            is_label.append(all(lbl in self.target_label[i] for lbl in label))
            
        if exclude:
            self.n_spectra      = len(self.targetID[~is_label])
            self.targetID       = self.targetID[~is_label]
            self.z_pipe         = self.z_pipe[~is_label]
            self.RA             = self.RA[~is_label]
            self.DEC            = self.DEC[~is_label]
            self.coadd_data     = self.coadd_data[~is_label][:]
            self.ivar           = self.ivar[~is_label][:]
            self.mask           = self.mask[~is_label][:]
            self.target_label   = [lbls for j, lbls in enumerate(self.target_label) if not is_label[j]]
            self.line_detections= [lbls for j, lbls in enumerate(self.line_detections) if not is_label[j]]
            self.subtype        += f" w/o {label}"
        else:
            self.n_spectra      = len(self.targetID[is_label])
            self.targetID       = self.targetID[is_label]
            self.z_pipe         = self.z_pipe[is_label]
            self.RA             = self.RA[is_label]
            self.DEC            = self.DEC[is_label]
            self.coadd_data     = self.coadd_data[is_label][:]
            self.ivar           = self.ivar[is_label][:]
            self.mask           = self.mask[is_label][:]
            self.target_label   = [lbls for j, lbls in enumerate(self.target_label) if is_label[j]]
            self.line_detections= [lbls for j, lbls in enumerate(self.line_detections) if is_label[j]]
    
    #
    # Mask the bad pixels in the spectrum
    #
    def mask_spectrum(self, i=None):
        if i is not None:
            good = (self.mask[i] == 0)
            self.coadd_data[i][~good] = np.nan
            self.ivar[i][~good] = np.nan
            self.smoothed_flux[i][~good] = np.nan
        else:
            good = (self.mask == 0)
            self.coadd_data[~good] = np.nan
            self.ivar[~good] = np.nan
            self.smoothed_flux[~good] = np.nan

    

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
                 record_detection=None,
                 e_or_a='e'):

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
        if two_component:
            lam0 = lam0 + lam0 # duplicate to generate free components
        
        self.fit_params = [[] for _ in range(self.n_spectra)]
        ##############################
        
        def fitting_func(lam, *params):
            gaussian_parms = []
            if fit_z:
                dz = params[0]
                for j in range(len(lam0)):
                    if two_component:
                        dlam = params[-3]
                        amp_j, sigma_j = params[2*j+1], params[2*j+2]
                        lam0_j = lam0[j] * (1 + z + dz) # fixed components
                        if j >= len(lam0)//2:
                            lam0_j = (lam0[j]+dlam) * (1 + z + dz) # free components
                    else:
                        amp_j, sigma_j = params[2*j+1], params[2*j + 2]
                        lam0_j = lam0[j] * (1 + z + dz)
                    gaussian_parms.append((amp_j, lam0_j, sigma_j))
            else:
                for j in range(len(lam0)):
                    if two_component:
                        dlam = params[-3]
                        amp_j, sigma_j = params[2*j], params[2*j + 1]
                        lam0_j = lam0[j] * (1 + z) # fixed components
                        if j >= len(lam0)//2:
                            lam0_j = (lam0[j]+dlam) * (1 + z) # free components
                    else:
                        offset = params[-3] # system offset
                        amp_j, sigma_j = params[3*j], params[3*j + 2]
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
                region = (np.min(lam0)*(1+z)- 200, np.max(lam0)*(1+z) + 200)

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
                    p0.append(1) # amp
                    p0.append(1)  # sigma
                if two_component:
                    p0.append(5) #dlam
            else:
                for j in range(len(lam0)):
                    if e_or_a == 'e':
                        p0.append(1)  # amp (positive for emission)
                    else:
                        p0.append(-1)  # amp
                    p0.append(1)  # sigma
                p0.append(0)  # dlam/offset
            p0.extend([0.0,5.0])  # conti_a, conti_b
            
            # Bound
            bounds_lower = []
            bounds_upper = []
            if fit_z:
                bounds_lower.append(-0.01)  # dz lower bound
                bounds_upper.append(+0.01)  # dz upper bound
                for j in range(len(lam0)):
                    bounds_lower.append(0) # Amp lower bound
                    bounds_upper.append(+100) # Amp upper bound
                    
                    
                    bounds_lower.append(2/(2*np.sqrt(2*np.log(2))))  # fwhm > 2
                    bounds_upper.append(10/(2*np.sqrt(2*np.log(2)))) # fwhm < 10
                if two_component:
                    bounds_lower.append(-10)  # dlam lower bound
                    bounds_upper.append(+10)  # dlam upper bound
            else:
                for j in range(len(lam0)):
                    if e_or_a == 'e':
                        bounds_lower.append(0) # Amp lower bound
                        bounds_upper.append(np.inf)    # Amp upper bound
                    elif e_or_a == 'a':
                        bounds_lower.append(-np.inf) # Amp lower bound
                        bounds_upper.append(0)    # Amp upper bound
                    bounds_lower.append(2/(2*np.sqrt(2*np.log(2))))  # fwhm > 2
                    bounds_upper.append(10/(2*np.sqrt(2*np.log(2)))) # fwhm < 10
                bounds_lower.append(-6)  # dlam/offset lower bound
                bounds_upper.append(+6)  # dlam/offset upper bound
                
            bounds_lower.extend([-np.inf, 0])  # conti_a, conti_b lower bounds
            bounds_upper.extend([np.inf, np.inf])    # conti_a, conti_b upper bounds
            

            try:
                popt, pcov = curve_fit(fitting_func, lam, flux, p0=p0, sigma=sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
                self.fit_params[i] = popt
                if (record_detection is not None) and (fit_z is False):
                    if two_component:
                        fitted_model = model(lam, 
                                    gaussian_parms=[(popt[2*k], lam0[k]*(1+z), popt[2*k+1]) for k in range(len(lam0)//2)]+\
                                        [(popt[2*k+(len(lam0)//2)], (lam0[k]+popt[-3])*(1+z), popt[2*k+1+(len(lam0)//2)]) for k in range(len(lam0)//2)],
                                    conti_parms=(popt[-2], popt[-1]))
                    else:
                        fitted_model = model(lam, 
                                    gaussian_parms=[(popt[2*k], (lam0[k]+popt[-3])*(1+z), popt[2*k+1]) for k in range(len(lam0))],
                                    conti_parms=(popt[-2], popt[-1]))
                    noise = np.std(flux - fitted_model)
                    
                    if (record_detection == 'NaD') and (two_component is False):
                        
                        if (min(-popt[0], -popt[2]) > 3*noise) \
                        and ((min(popt[1], popt[3]) > 2/(2*np.sqrt(2*np.log(2)))) and (max(popt[1], popt[3]) < 10/(2*np.sqrt(2*np.log(2))))):
                            self.line_detections[i].append(record_detection)
                    elif (record_detection == 'NaD') and (two_component is True):
                        if (min(max(-popt[0], -popt[4]), max(-popt[2], -popt[6])) > 3*noise) \
                        and ((min(popt[1], popt[5], popt[3], popt[7]) > 2/(2*np.sqrt(2*np.log(2)))) and (max(popt[1], popt[5], popt[3], popt[7]) < 10/(2*np.sqrt(2*np.log(2))))):
                            self.line_detections[i].append(record_detection)
                return popt
            except Exception as e:
                print(f"Error fitting flux for spectrum {self.targetID[i]}: {e}")
                self.fit_params[i] = None
                return None
        elif i is None:
            for k in range(self.n_spectra):
                popt = self.fit_flux(i=k, fitting_model=fitting_model, lam0=lam0, region=region, fit_z=fit_z)
                self.fit_params[k] = popt

                if (record_detection is not None) and (self.fit_params[i] is not None) and (fit_z is False):
                    self.fit_params[i] = popt
                    fitted_model = model(lam, 
                                gaussian_parms=[(popt[3*k], popt[3*k+1], popt[3*k+2]) for k in range(len(lam0))],
                                conti_parms=(popt[-2], popt[-1]))
                    noise = np.std(flux - fitted_model)
                    count = 0
                    for j in range(len(lam0)):
                        if np.abs(popt[3*j]) > 3*noise:
                            count += 1
                    if count >= len(lam0)//2:
                        self.line_detections[i].append(record_detection)
        
    def adjust_z(self, i, mode='base_2'):
        
        # self.adjust_z_mode[i] = mode
        
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
        
        Halpha_rest = air2vac(lines_air['Halpha'])
        NII_rest    = air2vac(lines_air['NII'])
        SII_rest    = air2vac(lines_air['SII'])
        OII_rest    = air2vac(lines_air['OII'])
        
        modes = {
            'auto': ([*Halpha_rest, *NII_rest,*SII_rest], True),
            'base_2': ([*Halpha_rest, *NII_rest,*SII_rest], True),
            'OII_2': ([*OII_rest], True),
            'base': ([*Halpha_rest, *NII_rest,*SII_rest], False),
            'OII': ([*OII_rest], False),
            'no': (0, 0),
        }
        
        
        use_mode = mode
        
        def get_dz_new(fit_z_mode):
            
            current_mode = list(modes.keys()).index(fit_z_mode)
            self.adjust_z_mode[i] = fit_z_mode
            
            if fit_z_mode == 'no':
                self.adjust_z_mode[i] = 'no'
                return 0
            
            fit_line, two_comp = modes[fit_z_mode]
            
            region = (np.min(fit_line)*(1+self.z_pipe[i])- 200, np.max(fit_line)*(1+self.z_pipe[i]) + 200)

            
            
            popt = self.fit_flux(i=i, fitting_model=model, lam0=fit_line, region=region, fit_z=True, two_component=two_comp)
            if popt is None:
                if use_mode != 'auto':
                    return get_dz_new(fit_z_mode='no')
                else:
                    current_mode += 1
                    self.adjust_z_mode[i] = list(modes.keys())[current_mode]
                    return get_dz_new(fit_z_mode=list(modes.keys())[current_mode])
            
            
            lam = desi_wavelength.copy()
            flux = self.coadd_data[i]
            ivar = self.ivar[i]
            mask = self.mask[i]
            
            crop_region = (lam >= region[0]) & (lam <= region[1])
            lam = lam[crop_region]
            flux = flux[crop_region]
            ivar = ivar[crop_region]
            mask = mask[crop_region]
            
            good = (mask == 0)
            lam = lam[good]
            flux = flux[good]
            ivar = ivar[good]
            sigma = np.sqrt(1 / ivar)
            
            if two_comp:
                fitted_model = model(lam, 
                   gaussian_parms=\
                        [(popt[2*k+1], fit_line[k]*(1+self.z_pipe[i]+(popt[0])), popt[2*k+2]) for k in range(len(fit_line))] +\
                        [(popt[2*k+len(fit_line)+1], (fit_line[k]+popt[-3])*(1+self.z_pipe[i]+(popt[0])), popt[2*k+len(fit_line)+2]) for k in range(len(fit_line))],
                    conti_parms=(popt[-2], popt[-1]))
                noise = np.std(flux - fitted_model)
                count = 0
                for j in range(len(fit_line)):
                    if max(popt[2*(j+len(fit_line))+1], popt[2*j+1]) > 3*noise:
                        count += 1
                if count >= (len(fit_line)//2):
                    dz = popt[0]
                    return dz
                else:
                    if use_mode != 'auto':
                        return get_dz_new(fit_z_mode='no')
                    else:
                        current_mode +=1
                        self.adjust_z_mode[i] = list(modes.keys())[current_mode]
                        return get_dz_new(fit_z_mode=list(modes.keys())[current_mode])
            else:
                fitted_model = model(lam, 
                        gaussian_parms=[(popt[2*k+1], fit_line[k]*(1+self.z_pipe[i]+(popt[0])), popt[2*k+2]) for k in range(len(fit_line))],
                        conti_parms=(popt[-2], popt[-1]))
                noise = np.std(flux - fitted_model)
                count = 0
                for j in range(len(fit_line)):
                    if popt[2*j+1] > 3*noise:
                        count += 1
                if count >= len(fit_line)//2:
                    dz = popt[0]
                    return dz
                else:
                    if use_mode != 'auto':
                        return get_dz_new(fit_z_mode='no')
                    else:
                        current_mode += 1
                        self.adjust_z_mode[i] = list(modes.keys())[current_mode]
                        return get_dz_new(fit_z_mode=list(modes.keys())[current_mode])

        dz = get_dz_new(mode)
        self.z[i] = self.z_pipe[i] + dz if dz is not None else self.z_pipe[i]