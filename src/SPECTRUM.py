import numpy as np
import pandas as pd
from astropy.table import Table
from .misc import *


class Spectrum:

    def __init__(self, spectra_data, cigale_data, fastspecfit_data, load_targetID=None):

        cigale_extract_columns = ['TARGETID', 'SPECTYPE', 'LOGM', 'LOGSFR', 
                                  'FLUX_G', 'FLUX_R', 'FLUX_W1', 'FLUX_W2', 'FLUX_Z']
        fastspecfit_extract_columns = ['TARGETID']

        # Convert the data from each FITS file into a pandas DataFrame
        df_spectra = Table(spectra_data[1].data).to_pandas()
        df_cigale = Table(cigale_data[1].data).to_pandas()
        df_fastspecfit = Table(fastspecfit_data[1].data).to_pandas()

        # Merge the DataFrames on the 'TARGETID' column
        # Start with the first DataFrame
        merged_df = df_spectra
        
        # Sequentially merge the other DataFrames
        # Using an inner join to keep only the target IDs present in all files
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
        self.df.drop(columns=['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'], inplace=True)
        self.color_mag = color_mag_all
        
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

        self._id_to_idx = {int(tid): i for i, tid in enumerate(self.targetID)}
        
        with open('filtered_ids.txt', 'w+') as f:
            for tid in self.targetID:
                f.write(f"{tid}\n")
    
    def shrink_dataset(self, step: int):
        self._apply_index(slice(None, None, step))
        return self

    def subset(self, criteria):
        self._apply_index(criteria)
        return self

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
        mask        = mask.astype(int) | np.where(ivar <= 0, 1, 0)
        continuum   = self.continuum
        emission    = self.emission

        lam = np.tile(desi_wavelength, (n_spectra, 1))
        data_stack = np.column_stack((lam, coadd_data-continuum, ivar, mask, continuum, emission))
        data_stack = data_stack.reshape(n_spectra, 6, -1)
        self.add_attribute('data_stack', data_stack)
        return self
    #
    # Shift to rest frame
    #
    def shift_to_rest_frame(self, i=None, id=None, dz=None):

        data_stack = self.data_stack
        id2index = self.id2index
        z = self.df['z'].to_numpy()

        
        if (i is None) or (id is None):
            data_stack[:, 0, :] = desi_wavelength.copy() / (1 + z[:, np.newaxis])
        else:
            if id is not None:
                i = id2index(id)
            
            if dz is None:
                data_stack[i, 0, :] = desi_wavelength.copy() / (1 + z[i])
            else:
                data_stack[i, 0, :] = desi_wavelength.copy() / (1 + z[i] + dz)
                self.df.at[i, 'z'] = z[i] + dz

        self.data_stack = data_stack
        return self
