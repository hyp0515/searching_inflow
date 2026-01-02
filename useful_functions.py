from matplotlib.table import Table
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import f
import requests
from pathlib import Path
import warnings
from multiprocessing import Pool, cpu_count
import pandas as pd
from astropy.table import Table
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

################################################################################################################
class Spectrum:

    def __init__(self, spectra_data, color_data, cigale_data, fastspecfit_data, load_targetID=None):

        color_extract_columns = ['TARGETID', 'SPECTYPE', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']
        cigale_extract_columns = ['TARGETID', 'LOGM', 'LOGSFR']
        fastspecfit_extract_columns = ['TARGETID']

        # Convert the data from each FITS file into a pandas DataFrame
        df_spectra = Table(spectra_data[1].data).to_pandas()
        df_color = Table(color_data[1].data).to_pandas()
        df_cigale = Table(cigale_data[1].data).to_pandas()
        df_fastspecfit = Table(fastspecfit_data[1].data).to_pandas()

        # Merge the DataFrames on the 'TARGETID' column
        # Start with the first DataFrame
        merged_df = df_spectra
        
        # Sequentially merge the other DataFrames
        # Using an inner join to keep only the target IDs present in all files
        merged_df = pd.merge(merged_df, df_color[color_extract_columns], on='TARGETID', how='inner')
        merged_df = pd.merge(merged_df, df_cigale[cigale_extract_columns], on='TARGETID', how='inner')
        merged_df = pd.merge(merged_df, df_fastspecfit[fastspecfit_extract_columns], on='TARGETID', how='inner')

        # Sort by TARGETID to ensure consistent order with array data
        merged_df = merged_df.sort_values('TARGETID').reset_index(drop=True)

        if load_targetID is not None:
            merged_df = merged_df[merged_df['TARGETID'].isin(load_targetID)].reset_index(drop=True)
        
        # Store the metadata in the dataframe
        self.df = merged_df
        self.targetID = self.df['TARGETID'].to_numpy()
        self.n_spectra = len(self.targetID)

        # Get the sorting indices to match the dataframe's order
        # This is necessary because the dataframe is sorted by TARGETID
        # Create a mapping from the original TARGETID to its index for spectra and fastspecfit data
        spectra_id_map = {tid: i for i, tid in enumerate(spectra_data[1].data['TARGETID'])}
        fastspecfit_id_map = {tid: i for i, tid in enumerate(fastspecfit_data[1].data['TARGETID'])}

        # Get the desired order of indices based on the sorted TARGETIDs in the merged dataframe
        spectra_indices = [spectra_id_map[tid] for tid in self.df['TARGETID']]
        fastspecfit_indices = [fastspecfit_id_map[tid] for tid in self.df['TARGETID']]

        # Use advanced integer indexing to select and reorder the data in a single step
        self.coadd_data = np.asarray(spectra_data[2].data[spectra_indices, 0, :], dtype=np.float32)
        self.ivar       = np.asarray(spectra_data[3].data[spectra_indices, 0, :], dtype=np.float32)
        self.mask       = np.asarray(spectra_data[4].data[spectra_indices, 0, :], dtype=np.float32)
        self.continuum  = np.asarray(fastspecfit_data[2].data[fastspecfit_indices], dtype=np.float32)
        self.emission   = np.asarray(fastspecfit_data[4].data[fastspecfit_indices], dtype=np.float32)

        # Add columns to the dataframe that were previously separate attributes
        self.df['z_pipe'] = self.df['Z']
        self.df['z'] = self.df['z_pipe'].copy()
        self.df.drop(columns=['Z'], inplace=True)

        # Calculate and store color magnitudes
        color_flux = self.df[['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']].to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            color_mag_all = 22.5 - 2.5 * np.log10(color_flux)
            color_mag_all[np.isinf(color_mag_all)] = np.nan # Handle -inf from log10(0)
        # self.df['color_mag'] = color_mag_all
        # self.df.drop(columns=['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'], inplace=True)
        self.df.drop(columns=['FLUX_W1', 'FLUX_W2'], inplace=True)
        self.color_mag = color_mag_all

        # For compatibility with existing methods, create view attributes
        # These will be updated if the dataframe is modified by other methods
        # self.z_pipe     = self.df['z_pipe'].to_numpy()
        # self.z          = self.df['z'].to_numpy()
        # self.RA         = self.df['RA'].to_numpy()
        # self.DEC        = self.df['DEC'].to_numpy()
        # self.spectype   = self.df['SPECTYPE'].to_numpy()
        # self.logM       = self.df['LOGM'].to_numpy()
        # self.logSFR     = self.df['LOGSFR'].to_numpy()
        
        
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

        self.df        = self.df.iloc[idx].reset_index(drop=True)
        
        self.coadd_data = self.coadd_data[idx]
        self.ivar       = self.ivar[idx]
        self.mask       = self.mask[idx]
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
        
        with open('filtered_ids.txt', 'w+') as f:
            for tid in self.targetID:
                f.write(f"{tid}\n")
    
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

    def SFG_filter(self, exclude=False):
        logM = self.df['LOGM'].to_numpy()
        logSFR = self.df['LOGSFR'].to_numpy()
        criteria = logSFR > (1 * (logM - 10) - 3)

        if exclude:
            criteria = ~criteria
            
        self.subset(criteria)
        return self

    def subtype_filter(self, subtype='QSO', exclude=True):
        
        if subtype.upper() not in ['QSO', 'LRG', 'ELG', 'BGS', 'MWS']:
            print(f"Subtype '{subtype}' not recognized. Available subtypes: ['QSO', 'LRG', 'ELG', 'BGS', 'MWS']")
            return np.array([False] * self.n_spectra)
        
        # simple vectorized comparison
        is_subtype = (self.df['SPECTYPE'].to_numpy() == subtype.upper())

        if exclude:
            is_subtype = ~is_subtype
        
        self.subset(is_subtype)
        return self

    def z_filter(self, max_z=0.4):
        z = self.df['z'].to_numpy()
        criteria = z < max_z
        self.subset(criteria)
        return self

    #
    # Stack data and mask bad pixels for convenience
    #    
    def stack_data(self):
        n_spectra   = self.n_spectra
        coadd_data  = self.coadd_data
        ivar        = self.ivar
        mask        = self.mask
        continuum   = self.continuum
        emission    = self.emission

        
        lam = np.tile(desi_wavelength, (n_spectra, 1))
        data_stack = np.column_stack((lam, coadd_data-continuum, ivar, mask, continuum, emission))
        data_stack = data_stack.reshape(n_spectra, 6, -1)
        self.add_attribute('data_stack', data_stack)

    def mask_bad(self):
        if not hasattr(self, 'data_stack'):
            self.stack_data()

        data_stack = self.data_stack

        # Create a boolean mask for bad pixels
        bad_mask = (data_stack[:, 3, :] != 0) | (data_stack[:, 2, :] <= 0)

        # Use a copy of the flux and ivar to modify
        flux = data_stack[:, 1, :].copy()
        ivar = data_stack[:, 2, :].copy()

        # Set bad pixels to NaN to be handled by pandas' interpolate
        flux[bad_mask] = np.nan
        ivar[bad_mask] = np.nan

        # Convert to pandas DataFrame for fast, vectorized interpolation
        df_flux = pd.DataFrame(flux)
        df_ivar = pd.DataFrame(ivar)

        # Interpolate along rows (axis=1). 'linear' is equivalent to np.interp.
        # limit_direction='both' fills NaNs at the start and end of the series.
        df_flux.interpolate(method='quadratic', axis=1, limit_direction='both', inplace=True)
        df_ivar.interpolate(method='quadratic', axis=1, limit_direction='both', inplace=True)

        # Convert back to numpy arrays and update the data_stack
        # Fill any remaining NaNs (e.g., if a whole spectrum was bad) with 0
        data_stack[:, 1, :] = df_flux.to_numpy(na_value=0.0)
        data_stack[:, 2, :] = df_ivar.to_numpy(na_value=0.0)

        # Reset the mask array to all zeros as bad pixels have been handled
        data_stack[:, 3, :] = 0

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
        z = data_class.df['z'].to_numpy()
        
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
                data_class.df.at[i, 'z'] = z[i] + dz
            self.shifted[i] = True
        
        data_class.data_stack = data_stack
        return data_class

    def label_emission_lines(self, data_class:Spectrum, s_2_n=3):
        n_spectra = data_class.n_spectra
        data_stack = data_class.data_stack

        OII_labels = []
        OIII_labels = []
        Halpha_labels = []
        NII_labels = []
        SII_labels = []
        Hbeta_labels = []
        for idx in range(n_spectra):
            label = []
            for l0, label in zip([OII_rest[1], OIII_rest[1], Halpha_rest[0], NII_rest[1], Hbeta_rest[0], SII_rest[0]],
                                [OII_labels, OIII_labels, Halpha_labels, NII_labels, Hbeta_labels, SII_labels]):
                l0_idx = np.searchsorted(data_stack[idx,0,:], l0)  - 1
                if data_stack[idx,5,l0_idx] > s_2_n/data_stack[idx,2,l0_idx]:
                    label.append(True)
                else:
                    label.append(False)

        data_class.df['OII']     = OII_labels
        data_class.df['Hbeta']   = Hbeta_labels
        data_class.df['OIII']    = OIII_labels
        data_class.df['Halpha']  = Halpha_labels
        data_class.df['NII']     = NII_labels
        data_class.df['SII']     = SII_labels

        return data_class
        
    
    
    def significant_emission_filter(self, data_class:Spectrum):
        df = data_class.df

        line_columns = ['OII', 'OIII', 'Hbeta', 'Halpha', 'NII', 'SII']
        detected_line_count = df[line_columns].sum(axis=1)
        filter_mask = (detected_line_count >= 1).values
        data_class.subset(filter_mask)
        return data_class

    def line_ratios(self, data_class:Spectrum):
        n_spectra = data_class.n_spectra
        data_stack = data_class.data_stack
        df = data_class.df
        
        oiii_hbeta = []
        nii_halpha = []
        for idx in range(n_spectra):
            lam = data_stack[idx, 0, :]
            flux = data_stack[idx, 1, :]
            ivar = data_stack[idx, 2, :]

            def get_line_flux(line_rest):
                line_idx = np.searchsorted(lam, line_rest) - 1
                line_flux = flux[line_idx]
                line_ivar = ivar[line_idx]
                line_sigma = np.sqrt(1/line_ivar)
                return line_flux, line_sigma

            OIII_5007_flux, OIII_5007_sigma = get_line_flux(OIII_rest[1])
            OIII_4959_flux, OIII_4959_sigma = get_line_flux(OIII_rest[0])
            Hbeta_flux, Hbeta_sigma         = get_line_flux(Hbeta_rest[0])
            NII_6583_flux, NII_6583_sigma   = get_line_flux(NII_rest[1])
            Halpha_flux, Halpha_sigma       = get_line_flux(Halpha_rest[0])

            OIII_Hbeta_ratio = OIII_5007_flux / Hbeta_flux
            NII_Halpha_ratio = NII_6583_flux / Halpha_flux


            oiii_hbeta.append(OIII_Hbeta_ratio)
            nii_halpha.append(NII_Halpha_ratio)
        
        OIII_Hbeta_ratio = np.array(oiii_hbeta)
        NII_Halpha_ratio = np.array(nii_halpha)
        
        df['OIII_Hbeta_ratio'] = OIII_Hbeta_ratio
        df['NII_Halpha_ratio'] = NII_Halpha_ratio
        

        return data_class
    

    def fit_multi_emission_vel(self, data_class:Spectrum, 
                               id=None, two_component=False, w_dz=False):
        
        data_stack = data_class.data_stack
        idx = data_class.id2index(id)
        df = data_class.df.iloc[idx]
        
        line_choices = {
            'OII'       : ([OII_rest[0]-20, OII_rest[1]+20], [(OII_rest[0], OII_rest[1])], 1/1.33),  # fixed ratio for [OII]3727/3729
            'Hbeta'     : ([Hbeta_rest[0]-20, Hbeta_rest[0]+20], [Hbeta_rest[0]], 0),
            'OIII'      : ([OIII_rest[0]-20, OIII_rest[1]+20], [(OIII_rest[0], OIII_rest[1])], 1/3.00),  # fixed ratio for [OIII]4959/5007
            'Halpha'    : ([NII_rest[0]-20, NII_rest[1]+20], [NII_rest[0], Halpha_rest[0], NII_rest[1]], 0),
            'SII'       : ([SII_rest[0]-20, SII_rest[1]+20], [SII_rest[0], SII_rest[1]], 0)
        }
        
        crop_region = []
        lines_to_fit = []
        line_ratios = []
        processed_halpha_nii = False
        for detected_line in ['OII', 'Hbeta', 'OIII', 'Halpha', 'NII', 'SII']:
            if detected_line in ['Halpha', 'NII']:
                if not processed_halpha_nii and (bool(df['Halpha']) or bool(df['NII'])):
                    crop_region.append(line_choices['Halpha'][0])
                    lines_to_fit.append(line_choices['Halpha'][1])
                    line_ratios.append(line_choices['Halpha'][2])
                    processed_halpha_nii = True
            elif bool(df[detected_line]):
                crop_region.append(line_choices[detected_line][0])
                lines_to_fit.append(line_choices[detected_line][1])
                line_ratios.append(line_choices[detected_line][2])

        
        def count_lines(region):
            count = 0
            for item in region:
                if isinstance(item, tuple):
                    count += len(item)
                else:
                    count += 1
            return count

        n_lines_total_region = [count_lines(region) for region in lines_to_fit]
        
        n_lines_fit_regions = [len(lines_to_fit[i]) for i in range(len(lines_to_fit))]
        n_lines_fit         = int(np.sum(n_lines_fit_regions))
        nline_start_indices = np.concatenate(([0], np.cumsum(n_lines_fit_regions)[:-1]))
        
        lam, flux, ivar = data_stack[idx, 0, :], data_stack[idx, 1, :], data_stack[idx, 2, :]

        slice_indices = []
        lams = []
        fluxes = []
        sigmas = []
        for i in range(len(crop_region)):
            slice_mask = (lam >= crop_region[i][0]) & (lam <= crop_region[i][1])
            lams.append(lam[slice_mask])
            fluxes.append(flux[slice_mask])
            sigmas.append(np.sqrt(1/np.abs(ivar[slice_mask])))
            slice_indices.append(np.sum(slice_mask))

        if len(slice_indices) > 1:
            slice_indices = np.cumsum(slice_indices)[:-1]
        else:
            slice_indices = []

        combine_lam     = np.concatenate(lams)
        combine_flux    = np.concatenate(fluxes)
        combine_sigma   = np.concatenate(sigmas)

        def unpack_params(params):
            gaussian_parms = [[] for _ in range(len(crop_region))] # OII, OIII, Halpha, SII
            lam0_adj = 1.0
            if two_component:
                if w_dz:
                    dz, sigma_1, sigma_2, dv_r, dv_l = params[:5]
                    amp_start_index = 5
                    lam0_adj += dz
                else:
                    sigma_1, sigma_2, dv_r, dv_l = params[:4]
                    amp_start_index = 4
            else:
                if w_dz:
                    dz, sigma_1 = params[:2]
                    amp_start_index = 2
                    lam0_adj += dz
                else:
                    sigma_1 = params[0]
                    amp_start_index = 1
                    
            gaussian_parms = [[] for _ in range(len(crop_region))] # OII, OIII, Halpha, SII
            amps = [[] for _ in range(len(crop_region))]
            lam0s = [[] for _ in range(len(crop_region))]
            for idx_lines, lines in enumerate(lines_to_fit): # 0:OII, 1:OIII, 2:Halpha, 3:SII
                for idx_line, line in enumerate(lines):
                    if not isinstance(line, tuple): # not doublet
                        if two_component:
                            # right component
                            sigma_r = sigma_1
                            amp_r   = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_r  = line * lam0_adj
                            gaussian_parms[idx_lines].insert(0, (amp_r, lam0_r, dv_r, sigma_r))
                            amps[idx_lines].insert(0, amp_r)
                            lam0s[idx_lines].insert(0, lam0_r)

                            # left component
                            sigma_l = sigma_2
                            amp_l   = params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                            lam0_l  = line * lam0_adj
                            gaussian_parms[idx_lines].append((amp_l, lam0_l, dv_l, sigma_l))
                            amps[idx_lines].append(amp_l)
                            lam0s[idx_lines].append(lam0_l)
                        else:
                            amp   = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0  = line * lam0_adj
                            gaussian_parms[idx_lines].insert(0, (amp, lam0, 0, sigma_1))
                            amps[idx_lines].insert(0, amp)
                            lam0s[idx_lines].insert(0, lam0)
                    else: # doublet
                        line1, line2 = line
                        line_ratio = line_ratios[idx_lines]
                            
                        if two_component:
                            sigma_r     = sigma_1
                            sigma_l     = sigma_2
                            
                            amp_1_r     = line_ratio * params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_1_r    = line1 * lam0_adj
                            gaussian_parms[idx_lines].insert(0, (amp_1_r, lam0_1_r, dv_r, sigma_r))
                            amps[idx_lines].insert(0, amp_1_r)
                            lam0s[idx_lines].insert(0, lam0_1_r)

                            amp_2_r     = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_2_r    = line2 * lam0_adj
                            gaussian_parms[idx_lines].insert(0, (amp_2_r, lam0_2_r, dv_r, sigma_r))
                            amps[idx_lines].insert(0, amp_2_r)
                            lam0s[idx_lines].insert(0, lam0_2_r)

                            amp_1_l     = line_ratio * params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                            lam0_1_l    = line1 * lam0_adj
                            gaussian_parms[idx_lines].append((amp_1_l, lam0_1_l, dv_l, sigma_l))
                            amps[idx_lines].append(amp_1_l)
                            lam0s[idx_lines].append(lam0_1_l)

                            amp_2_l     = params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                            lam0_2_l    = line2 * lam0_adj
                            gaussian_parms[idx_lines].append((amp_2_l, lam0_2_l, dv_l, sigma_l))
                            amps[idx_lines].append(amp_2_l)
                            lam0s[idx_lines].append(lam0_2_l)
                        else:
                            amp_1       = line_ratio * params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_1      = line1 * lam0_adj
                            gaussian_parms[idx_lines].insert(0, (amp_1, lam0_1, 0, sigma_1))
                            amps[idx_lines].insert(0, amp_1)
                            lam0s[idx_lines].insert(0, lam0_1)

                            amp_2       = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_2      = line2 * lam0_adj
                            gaussian_parms[idx_lines].insert(0, (amp_2, lam0_2, 0, sigma_1))
                            amps[idx_lines].insert(0, amp_2)
                            lam0s[idx_lines].insert(0, lam0_2)
            return gaussian_parms, amps, lam0s

        def fitting_func(lam_grid, *params):
            lams = np.split(lam_grid, slice_indices)
            gaussian_parms, _, _ = unpack_params(params)
            combine_model = np.concatenate([
                model_vel(lams[i], gaussian_parms=gaussian_parms[i]) 
                for i in range(len(crop_region))
            ])
            return combine_model
        

        dz_init, dz_upper, dz_lower                     = 0, 1e-3, -1e-3
        sigma_1_init, sigma_1_upper, sigma_1_lower      = 70, 300, 30
        amp_init, amp_upper, amp_lower                  = [np.max(combine_flux)/2]*n_lines_fit, [np.max(combine_flux)]*n_lines_fit, [0]*n_lines_fit
        if two_component:
            sigma_2_init, sigma_2_upper, sigma_2_lower  = sigma_1_init, sigma_1_upper, sigma_1_lower
            dv_r_init, dv_r_upper, dv_r_lower           =  0.5, 500,    0     # right component
            dv_l_init, dv_l_upper, dv_l_lower           = -0.5,   0, -500     # left component
            amp_init, amp_upper, amp_lower              = [np.max(combine_flux)/2]*int(n_lines_fit*2), [np.max(combine_flux)]*int(n_lines_fit*2), [0]*int(n_lines_fit*2)

        if two_component:
            if w_dz:
                p0 = [dz_init, sigma_1_init, sigma_2_init, dv_r_init, dv_l_init] + amp_init
                bounds_lower = [dz_lower, sigma_1_lower, sigma_2_lower, dv_r_lower, dv_l_lower] + amp_lower
                bounds_upper = [dz_upper, sigma_1_upper, sigma_2_upper, dv_r_upper, dv_l_upper] + amp_upper
            else:
                p0 = [sigma_1_init, sigma_2_init, dv_r_init, dv_l_init] + amp_init
                bounds_lower = [sigma_1_lower, sigma_2_lower, dv_r_lower, dv_l_lower] + amp_lower
                bounds_upper = [sigma_1_upper, sigma_2_upper, dv_r_upper, dv_l_upper] + amp_upper

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
        # print(popt)
        params = {}
        if two_component:
            if w_dz:
                dz, sigma_1, sigma_2, dv_r, dv_l = popt[:5]
                params['dz'] = dz
            else:
                sigma_1, sigma_2, dv_r, dv_l = popt[:4]
                params['dz'] = 0
            params['sigma'] = sigma_1, sigma_2
            params['dv'] = (dv_r, dv_l)
        else:
            if w_dz:
                dz, sigma_1 = popt[:2]
                params['dz'] = dz
                params['sigma'] = sigma_1
                params['dv'] = None
            else:
                sigma_1 = popt[0]
                params['dz'] = 0
                params['sigma'] = sigma_1
                params['dv'] = None

        gaussian_parms, amps, lam0s = unpack_params(popt)
        params['gaussian_params'] = gaussian_parms
        params['amps'] = amps
        params['lam0s'] = lam0s
        
        if two_component:
            def split_components(data_list):
                right = [region[:n_lines_total_region[i]][::-1] for i, region in enumerate(data_list)]
                left = [region[n_lines_total_region[i]:] for i, region in enumerate(data_list)]
                return left, right

            params['left_comp'],   params['right_comp'] = split_components(gaussian_parms)
            params['left_amps'],   params['right_amps'] = split_components(amps)
            params['left_lam0s'], params['right_lam0s'] = split_components(lam0s)
        return params, (combine_lam, combine_flux, combine_sigma), slice_indices, n_lines_fit

    
    def find_dp(self, data_class:Spectrum, id=None):
        data_stack = data_class.data_stack
        if id is not None:
            idx = data_class.id2index(id)
            df = data_class.df.iloc[idx]
            
            params_1comp, (combine_lam, combine_flux, combine_sigma), slice_indices, n_lines_fit = self.fit_multi_emission_vel(data_class, id=id, two_component=False, w_dz=False)
            lams = np.split(combine_lam, slice_indices)
            model_1comp = np.concatenate([
                        model_vel(lams[i], gaussian_parms=params_1comp['gaussian_params'][i]) 
                        for i in range(len(lams))
                    ])
            residual_1comp = combine_flux - model_1comp
            
            
            params_2comp, (combine_lam, combine_flux, combine_sigma), slice_indices, n_lines_fit = self.fit_multi_emission_vel(data_class, id=id, two_component=True, w_dz=False)
            model_2comp = np.concatenate([
                model_vel(lams[i], gaussian_parms=params_2comp['gaussian_params'][i]) for i in range(len(lams))
            ])
            residual_2comp = combine_flux - model_2comp

            # Criteria 1: F-test
            chisq_1comp = np.sum((residual_1comp/combine_sigma)**2)
            dof_1comp = len(combine_lam) - (1+n_lines_fit)
            
            chisq_2comp = np.sum((residual_2comp/combine_sigma)**2)
            dof_2comp = len(combine_lam) - (4+n_lines_fit*2)
            
            F_stat = ((chisq_1comp - chisq_2comp) / (dof_1comp - dof_2comp)) / (chisq_2comp / dof_2comp)
            p_value = 1 - f.cdf(F_stat, dof_1comp - dof_2comp, dof_2comp)
            
            
            # Criteria 2: |dv_r-dv_l| > 3 vel_resolution or 200 km/s
            dv_r, dv_l = params_2comp['dv']
            delta_dv = np.abs(dv_r - dv_l)
            
            # Criteria 3: 1/3 < Amp1/Amp2 < 3
            line_choices = {
                'OII'       : ([OII_rest[0]-20, OII_rest[1]+20], [(OII_rest[0], OII_rest[1])], 1/1.33),  # fixed ratio for [OII]3727/3729
                'Hbeta'     : ([Hbeta_rest[0]-20, Hbeta_rest[0]+20], [Hbeta_rest[0]], 0),
                'OIII'      : ([OIII_rest[0]-20, OIII_rest[1]+20], [(OIII_rest[0], OIII_rest[1])], 1/3.00),  # fixed ratio for [OIII]4959/5007
                'Halpha'    : ([NII_rest[0]-20, NII_rest[1]+20], [NII_rest[0], Halpha_rest[0], NII_rest[1]], 0),
                'SII'       : ([SII_rest[0]-20, SII_rest[1]+20], [SII_rest[0], SII_rest[1]], 0)
            }

            
            left_amps  = params_2comp['left_amps']
            right_amps = params_2comp['right_amps']
            
            residual_region = np.split(residual_2comp, slice_indices)
            sigmab_region = [np.std(residual_region[i]) for i in range(len(residual_region))]
            
            
            # Determine which regions were actually fitted
            detected_line_names = []
            processed_halpha_nii = False
            for line in ['OII', 'Hbeta', 'OIII', 'Halpha', 'NII', 'SII']:
                if line in ['Halpha', 'NII']:
                    if not processed_halpha_nii and (df['Halpha'] or df['NII']):
                        detected_line_names.append('Halpha') # Represents the combined region
                        processed_halpha_nii = True
                elif df[line]:
                    detected_line_names.append(line)

            # Now iterate through the detected lines and their corresponding fit results
            dp_lines = []
            line_fluxes = []
            region_idx = 0
            n_lines_region = [2, 1, 2, 3, 2]
            for k, line_name in enumerate(['OII', 'Hbeta', 'OIII', 'Halpha', 'SII']):
                if line_name in detected_line_names:
                    if line_name == 'Halpha' and 'NII' in detected_line_names:
                        # This is the combined Halpha/NII region
                        # We process it once under 'Halpha' and skip for 'NII'
                        pass
                    
                    left_amp_region = left_amps[region_idx]
                    right_amp_region = right_amps[region_idx]
                    sigma_b = sigmab_region[region_idx]
                    
                    dps = [
                        (1/3 * la < ra < 3 * la) and (ra > 3 * sigma_b) and (la > 3 * sigma_b)
                        for la, ra in zip(left_amp_region, right_amp_region)
                    ]
                    dp_lines.append(dps)
                    
                    line_fluxes_region = []
                    for j in range(len(params_2comp['left_comp'][region_idx])):
                        left_comp_dz_free = np.concatenate([
                            model_vel(lams[region_idx], gaussian_parms=[params_2comp['left_comp'][region_idx][j]])])
                        right_comp_dz_free = np.concatenate([
                            model_vel(lams[region_idx], gaussian_parms=[params_2comp['right_comp'][region_idx][j]])])
                        model_lam_2comp_free = left_comp_dz_free + right_comp_dz_free
                        line_fluxes_region.append(np.max(model_lam_2comp_free))
                    line_fluxes.append(line_fluxes_region)
                    
                    region_idx += 1
                else:
                    dp_lines.append([False]*n_lines_region[k])
                    line_fluxes.append([0]*n_lines_region[k])
                    

            # Flatten the list of lists into a single list of booleans
            dp_detection = [
                item
                for sublist in dp_lines
                for item in sublist
            ]

            # Flatten the list of lists into a single list of fluxes
            line_fluxes_flat = [
                item
                for sublist in line_fluxes
                for item in sublist
            ]

            # detected_fluxes = np.array(line_fluxes_flat)[line_fluxes_flat!=0]
            # ranks = np.empty_like(detected_fluxes, dtype=int)
            # ranks[np.argsort(detected_fluxes)[::-1]] = np.arange(len(detected_fluxes))
            
            # line_fluxes_rank = np.full(len(line_fluxes_flat), -1, dtype=int)
            # line_fluxes_rank[line_fluxes_flat!=0] = ranks
            # Convert to numpy array for boolean indexing and calculations
            line_fluxes_flat = np.array(line_fluxes_flat)
            
            # Create an array to store ranks, initialized to -1
            line_fluxes_rank = np.full(line_fluxes_flat.shape, -1, dtype=int)
            
            # Get the indices of non-zero fluxes
            non_zero_indices = np.where(line_fluxes_flat > 0)[0]
            
            # If there are non-zero fluxes, rank them
            if non_zero_indices.size > 0:
                # Get the fluxes that are not zero
                detected_fluxes = line_fluxes_flat[non_zero_indices]
                
                # Get the indices that would sort the detected fluxes in descending order
                sorted_indices_of_detected = np.argsort(detected_fluxes)[::-1]
                
                # Create the ranks (0 for the highest flux, 1 for the second, etc.)
                ranks = np.arange(len(detected_fluxes))
                
                # Place the ranks back into the correct positions in the full rank array
                # The original indices of the detected fluxes are used to place the ranks.
                # The ranks are ordered according to the sorted detected fluxes.
                line_fluxes_rank[non_zero_indices[sorted_indices_of_detected]] = ranks
            return p_value, delta_dv, dp_detection, line_fluxes_rank, params_2comp
