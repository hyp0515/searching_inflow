import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d
import requests
from astropy.io import fits


c = const.c.cgs.value * 1e-5  # speed of light in km/s
desi_wavelength = np.arange(3600, 9824 + .8, .8) # DESI's observe wavelength

# https://astronomy.nmsu.edu/drewski/tableofemissionlines.html
lines_air = {
    'Halpha'    : [6562.819],
    'Hbeta'     : [4861.333],
    'OIII'      : [4958.911, 5006.843],
    'OII'       : [3726.032, 3728.815],
    'SII'       : [6716.440, 6730.810],
    'NII'       : [6548.050, 6583.460],
    'CaII'      : [8498.020, 8542.090, 8662.140],
    'NaD'       : [5890.004, 5895.985],
    'HeI'       : [5875.624]
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

def lam2vel(lam, lam0, z=0):
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

def gaussian_lam(lam, amp, lam0, sigma):
    """
    Generate a Gaussian profile.
    """
    return amp * np.exp(-0.5 * ((lam-lam0)**2) / (sigma**2))

def model_lam(lam, gaussian_parms=None, conti_parms=(0, 0)):
    
    flux = 0

    if gaussian_parms is not None:
        for p in gaussian_parms:
            amp, lam0, sigma = p
            gauss = gaussian_lam(lam, amp, lam0, sigma)
            flux += gauss

    conti_a, conti_b = conti_parms
    conti = conti_a * lam + conti_b

    return flux + conti

def gaussian_vel(lam, amp, lam0, dv, sigma):
    """
    Generate a Gaussian profile.
    """
    return amp * np.exp(-0.5 * ((c*(lam-lam0)/lam0 - dv)**2) / (sigma**2))

def model_vel(lam, gaussian_parms=None, conti_parms=(0, 0)):
    
    flux = 0

    if gaussian_parms is not None:
        for p in gaussian_parms:
            amp, lam0, dv, sigma = p
            gauss = gaussian_vel(lam, amp, lam0, dv, sigma)
            flux += gauss

    conti_a, conti_b = conti_parms
    conti = conti_a * lam + conti_b

    return flux + conti

def image_link(RA, DEC, save_image=False, fname=None, plot=False, side_arcmin=0.5):
    if save_image:
        scale = np.round(side_arcmin / 3, 6)
        url = f'https://www.legacysurvey.org/viewer/cutout.jpg?ra={RA}&dec={DEC}&pixscale={scale}&layer=hsc-dr3&size=200'
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
        pixscale = side_arcmin * 60 / 200  # arcsec/pixel
        radius_arcsec = 1.5 / 2
        radius_pixels = radius_arcsec / pixscale
        circle = plt.Circle((100, 100), radius_pixels, color='magenta', fill=False, lw=1.5, linestyle='--')
        plt.gca().add_patch(circle)
        plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(fname if fname else "cutout.jpg")
        plt.show()
    
    return f'https://www.legacysurvey.org/viewer?ra={RA}&dec={DEC}&layer=hsc-dr3&zoom=14'

def spectrum_link(targetID): 
    return f'https://www.legacysurvey.org/viewer/desi-spectrum/dr1/targetid{targetID}'


def new_fits(fits_data, targetIDs, fname=None):
    """
    Create a new FITS file containing only the entries for the specified target IDs.

    Parameters:
    fits_data : astropy.io.fits.HDUList
        The original FITS data.
    targetIDs : list or array-like
        List of target IDs to include in the new FITS file.

    Returns:
    astropy.io.fits.HDUList
        A new FITS HDUList containing only the specified target IDs.
    """

    # Create a mask for the rows to keep
    original_targetIDs = fits_data[1].data['TARGETID']
    mask = np.isin(original_targetIDs, targetIDs)

    # Create a new HDUList
    new_hdul = fits.HDUList()
    
    # Copy the primary HDU
    new_hdul.append(fits.PrimaryHDU(header=fits_data[0].header))
    
    # Copy and filter each extension
    for i in range(1, len(fits_data)):
        hdu = fits_data[i]
        header = hdu.header
        if isinstance(hdu, fits.BinTableHDU):
            # This is a table, filter rows based on TARGETID
            # This assumes the table has the same number of rows as the one used to create the mask
            if hdu.data is not None and len(hdu.data) == len(mask):
                data = hdu.data[mask]
                new_hdul.append(fits.BinTableHDU(data=data, header=header, name=hdu.name))
            else:
                # If row count doesn't match, copy as is with a warning or handle as needed
                new_hdul.append(hdu)
        else:
            # For other HDU types, just copy them
            new_hdul.append(hdu)
    
    if fname is not None:
        new_hdul.writeto(fname, overwrite=True)
        print(f"New FITS file saved as {fname}")
    
    return new_hdul


def read_ids(fname):
    """
    Read target IDs from a text file.

    Parameters:
    fname : str
        Path to the text file containing target IDs.

    Returns:
    list
        List of target IDs read from the file.
    """
    with open(fname, 'r') as f:
        targetIDs = [int(line.strip()) for line in f if line.strip().isdigit()]
    return targetIDs