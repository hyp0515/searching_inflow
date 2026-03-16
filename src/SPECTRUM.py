import numpy as np
import pandas as pd
from astropy.table import Table
from .misc import *
import tqdm
import time

class Spectrum:

    def __init__(self, 
                 spectra_data, 
                 cigale_data, 
                 fastspecfit_data, 
                 load_targetID=None,
                 subtype_filter=None):
        start_time = time.time()
        cigale_extract_columns = ['TARGETID', 'SPECTYPE', 'LOGM', 'LOGSFR', 
                                  'FLUX_G', 'FLUX_R', 'FLUX_W1', 'FLUX_W2', 'FLUX_Z']
        fastspecfit_extract_columns = ['TARGETID']

        # Convert the data from each FITS file into a pandas DataFrame
        df_spectra      = Table(spectra_data[1].data).to_pandas()
        df_cigale       = Table(cigale_data[1].data).to_pandas()
        df_fastspecfit  = Table(fastspecfit_data[1].data).to_pandas()

        merged_df = df_spectra
        merged_df = pd.merge(merged_df, df_cigale[cigale_extract_columns], on='TARGETID', how='inner')
        merged_df = pd.merge(merged_df, df_fastspecfit[fastspecfit_extract_columns], on='TARGETID', how='inner')
        merged_df = merged_df.sort_values('TARGETID').reset_index(drop=True)

        if load_targetID is not None:
            merged_df = merged_df[merged_df['TARGETID'].isin(load_targetID)].reset_index(drop=True)

        if subtype_filter is not None:
            merged_df = merged_df[merged_df['SPECTYPE'] != subtype_filter].reset_index(drop=True)

        self.df = merged_df
        self.targetID = self.df['TARGETID'].to_numpy()
        self.n_spectra = len(self.targetID)
        self.df['Z'] = self.df['Z'].to_numpy()
        
        spectra_id_map = {tid: i for i, tid in enumerate(spectra_data[1].data['TARGETID'])}
        fastspecfit_id_map = {tid: i for i, tid in enumerate(fastspecfit_data[1].data['TARGETID'])}
        spectra_indices = [spectra_id_map[tid] for tid in self.df['TARGETID']]
        fastspecfit_indices = [fastspecfit_id_map[tid] for tid in self.df['TARGETID']]

        # Use advanced integer indexing to select and reorder the data in a single step
        self.coadd_data = np.asarray(spectra_data[2].data[spectra_indices, 0, :], dtype=np.float32)
        self.ivar       = np.asarray(spectra_data[3].data[spectra_indices, 0, :], dtype=np.float32)
        mask            = np.asarray(spectra_data[4].data[spectra_indices, 0, :], dtype=np.float32)
        self.mask       = mask.astype(int) | np.where(self.ivar <= 0, 1, 0)
        self.continuum  = np.asarray(fastspecfit_data[2].data[fastspecfit_indices], dtype=np.float32)
        self.flux = self.coadd_data - self.continuum
        # self.emission   = np.asarray(fastspecfit_data[4].data[fastspecfit_indices], dtype=np.float32)

        
        # Calculate and store color magnitudes
        color_flux = self.df[['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2']].to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            color_mag_all = 22.5 - 2.5 * np.log10(color_flux)
            color_mag_all[np.isinf(color_mag_all)] = np.nan # Handle -inf from log10(0)
        self.df.drop(columns=['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'], inplace=True)
        self.color_mag = color_mag_all
        
        self._id_to_idx = {int(tid): i for i, tid in enumerate(self.targetID)}
        end_time = time.time()
        print(f"Data loading and processing took {end_time - start_time:.2f} seconds.")
    
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
        # self.emission    = self.emission[idx]
    
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
        z = self.df['Z']
        criteria = z < max_z
        self.subset(criteria)
        return self

