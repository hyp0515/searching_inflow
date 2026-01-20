import numpy as np
from scipy.stats import f
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from .misc import *
from .SPECTRUM import Spectrum
from .FITSPECTRUM import FitSpectrum
from astropy.table import Table

FIT = FitSpectrum()

class DP:
    def __init__(self):
        pass

    def save_df(self, df: pd.DataFrame, fname: str):
        df.to_csv(fname, index=False)

    def record_ids(self, df: pd.DataFrame, fname: str):
        with open(fname, 'w+') as f:
            for tid in df['TARGETID']:
                f.write(f"{tid}\n")

    def reconstruct_fit(self, data_class:Spectrum, id=None):
        idx = data_class.id2index(id)
        lam = data_class.data_stack[idx, 0, :]


        params_1comp, _, slice_indices, _ = FIT.fit_multi_emission_vel(data_class=data_class, id=id, two_component=False, w_dz=False)
        model_1comp = np.sum([
                    model_vel(lam, gaussian_parms=params_1comp['gaussian_params'][i]) 
                    for i in range(len(slice_indices)+1)
                ], axis=0)


        params_2comp, _, slice_indices, _ = FIT.fit_multi_emission_vel(data_class=data_class, id=id, two_component=True, w_dz=False)
        left_2comp = np.sum([
            model_vel(lam, gaussian_parms=params_2comp['left_comp'][i]) for i in range(len(slice_indices)+1)
        ], axis=0)

        right_2comp = np.sum([
            model_vel(lam, gaussian_parms=params_2comp['right_comp'][i]) for i in range(len(slice_indices)+1)
        ], axis=0)
        return model_1comp, left_2comp, right_2comp
    
    def fit_dp(self, data_class:Spectrum, id=None):
        
        params_1comp, (combine_lam, combine_flux, combine_sigma), slice_indices, n_lines_fit = FIT.fit_multi_emission_vel(data_class=data_class, id=id, two_component=False, w_dz=False)
        lams = np.split(combine_lam, slice_indices)
        model_1comp = np.concatenate([
            model_vel(lams[i], gaussian_parms=params_1comp['gaussian_params'][i]) for i in range(len(lams))
        ])
        residual_1comp = combine_flux - model_1comp
        
        
        params_2comp, (combine_lam, combine_flux, combine_sigma), slice_indices, n_lines_fit = FIT.fit_multi_emission_vel(data_class=data_class, id=id, two_component=True, w_dz=False)
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


    
        # Criteria 2: |dv_r-dv_l| > 3 vel_resolution
        dv_r, dv_l = params_2comp['dv_r'], params_2comp['dv_l']
        delta_dv = np.abs(dv_r - dv_l)
        

        left_amps  = params_2comp['left_amps']
        right_amps = params_2comp['right_amps']
        
        residual_region = np.split(residual_2comp, slice_indices)
        sigmab_region = [np.std(residual_region[i]) if residual_region[i].size > 1 else 0 for i in range(len(residual_region))]
        # sigmab_region = [np.mean(combine_sigma[i]) for i in range(len(combine_sigma))]
        
        
        left_amps  = params_2comp['left_amps']
        right_amps = params_2comp['right_amps']

        residual_region = np.split(residual_2comp, slice_indices)
        sigmab_region = [np.std(residual_region[i]) if residual_region[i].size > 1 else 0 for i in range(len(residual_region))]

        dp_detections = []
        line_fluxes = []
        for k, _ in enumerate(['OII', 'Hbeta', 'OIII', 'Halpha', 'SII']):

            left_amp_region = left_amps[k]
            right_amp_region = right_amps[k]
            sigma_b = sigmab_region[k]

            for la, ra in zip(left_amp_region, right_amp_region):
                # Criteria 3: 1/3 < amp_r/amp_l < 3
                # Criteria 4: amp_r, amp_l > 3 sigma_background
                dp_detections.append((1/3 * la < ra < 3 * la)&(ra > 3 * sigma_b)&(la > 3 * sigma_b))

            for j in range(len(params_2comp['left_comp'][k])):
                left_comp_dz_free = np.concatenate([
                    model_vel(lams[k], gaussian_parms=[params_2comp['left_comp'][k][j]])])
                right_comp_dz_free = np.concatenate([
                    model_vel(lams[k], gaussian_parms=[params_2comp['right_comp'][k][j]])])
                model_lam_2comp_free = left_comp_dz_free + right_comp_dz_free
                try:
                    line_fluxes.append(np.max(model_lam_2comp_free))
                except:
                    line_fluxes.append(0)
        

        # Convert to numpy array for boolean indexing and calculations
        line_fluxes_flat = np.array(line_fluxes)
        
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
        return p_value, dp_detections, line_fluxes_rank, params_2comp, params_1comp

    
    def fit_all(self, data_class:Spectrum, n_jobs=5):
        dp_cols = ['OII3726_dp', 'OII3729_dp',
                'Hbeta_dp',
                'OIII4959_dp', 'OIII5007_dp',
                'NII6548_dp', 'Halpha_dp', 'NII6583_dp', 
                'SII6716_dp', 'SII6731_dp']
        dp_rank_cols = [f'{col[:-3]}_rank' for col in dp_cols]
        
        def process_target(target_id):
            """
            Processes a single target to find double-peaked features.
            """
            p_value, dp_detections, line_fluxes_rank, params_2comp, params_1comp = self.fit_dp(data_class=data_class, id=target_id)
            idx = data_class.id2index(target_id)
            Z, RA, DEC, LOGSFR, LOGM = data_class.df.iloc[idx][['z', 'RA', 'DEC', 'LOGSFR', 'LOGM']]
            
            model_1comp = np.sum([
                model_vel(data_class.data_stack[idx, 0, :], gaussian_parms=params_1comp['gaussian_params'][i]) for i in range(len(params_1comp['gaussian_params']))
            ], axis=0)
            
            left_2comp = np.sum([
                model_vel(data_class.data_stack[idx, 0, :], gaussian_parms=params_2comp['left_comp'][i]) for i in range(len(params_2comp['left_comp']))
            ], axis=0)

            right_2comp = np.sum([
                model_vel(data_class.data_stack[idx, 0, :], gaussian_parms=params_2comp['right_comp'][i]) for i in range(len(params_2comp['right_comp']))
            ], axis=0)
            
            data = {
                'TARGETID': target_id.astype(np.int64),
                'RA': RA.astype(np.float32),
                'DEC': DEC.astype(np.float32),
                'Z': Z.astype(np.float32),
                'LOGM': LOGM.astype(np.float32),
                'LOGSFR': LOGSFR.astype(np.float32),
                'dv_r': params_2comp['dv_r'].astype(np.float32),
                'dv_l': params_2comp['dv_l'].astype(np.float32),
                'sigma_r': params_2comp['sigma_r'].astype(np.float32),
                'sigma_l': params_2comp['sigma_l'].astype(np.float32),
                'sigma_1comp': params_1comp['sigma'].astype(np.float32),
                'p_value': p_value.astype(np.float32),
                'model_1comp': model_1comp.astype(np.float32),
                'left_2comp': left_2comp.astype(np.float32),
                'right_2comp': right_2comp.astype(np.float32)
            }
            data.update(dict(zip(dp_cols, dp_detections)))
            data.update(dict(zip(dp_rank_cols, line_fluxes_rank)))
            return data
        # Use joblib to parallelize the processing
        results = Parallel(n_jobs=n_jobs)(delayed(process_target)(target_id) for target_id in tqdm(data_class.targetID))
        
        dp_parent = pd.DataFrame(results)
        
        model_1comp = np.array(dp_parent['model_1comp'].to_list())
        left_2comp = np.array(dp_parent['left_2comp'].to_list())
        right_2comp = np.array(dp_parent['right_2comp'].to_list())
        dp_parent.drop(columns=['model_1comp', 'left_2comp', 'right_2comp'], inplace=True)
        return dp_parent, model_1comp, left_2comp, right_2comp
        
    
    def select_dp_sample(self, dp_parent: pd.DataFrame, model_1comp, left_2comp, right_2comp):
        # Criteria 1: p_value < 0.05
        # Criteria 2: |dv_r - dv_l| > 3 * vel_resolution
        criteria_1 = dp_parent['p_value'] < 0.05
        criteria_2 = (dp_parent['dv_r'] - dp_parent['dv_l']).abs() > 3 * c * 0.8 / (Halpha_rest[0] * (1 + dp_parent['Z']))
        dp_candidate = dp_parent[criteria_1 & criteria_2].copy()

        dp_cols = ['OII3726_dp', 'OII3729_dp',
                'Hbeta_dp',
                'OIII4959_dp', 'OIII5007_dp',
                'NII6548_dp', 'Halpha_dp', 'NII6583_dp', 
                'SII6716_dp', 'SII6731_dp']
        dp_rank_cols = [f'{col[:-3]}_rank' for col in dp_cols]
        

        def get_dp_info(row):
            """
            Identifies consecutive double peaks starting from the brightest emission line.
            - Filters out lines with a rank of -1 (not detected).
            - Sorts the remaining emission lines by rank (brightness).
            - If the brightest line does not have a double peak, returns 0 and an empty list.
            - Otherwise, counts consecutive lines with double peaks starting from the brightest
              and returns the count and a list of the corresponding line names.
            """
            # Create a list of (rank, dp_status, line_name) tuples for detected lines (rank != -1)
            line_info = []
            for rank_col, dp_col in zip(dp_rank_cols, dp_cols):
                rank = row[rank_col]
                if rank != -1:
                    line_name = dp_col[:-3]  # Remove '_dp' suffix
                    line_info.append((rank, row[dp_col], line_name))
                
            # If no lines were detected, there are no DPs to count.
            if not line_info:
                return 0
            
            # Sort by rank (the first element of the tuple)
            line_info.sort()

            # Check if the brightest line (first in the sorted list) has a double peak
            if not line_info[0][1]:  # line_info[0][1] is the dp_status
                return 0

            # Count consecutive double peaks from the brightest and collect their names
            actual_dp_count = 0
            actual_dp_lines = []
            for rank, has_dp, line_name in line_info:
                if has_dp:
                    actual_dp_count += 1
                    actual_dp_lines.append(line_name)
                else:
                    # Stop counting when a line without a double peak is found
                    break
            return actual_dp_count

        # Apply the function to each row to create the new columns
        dp_candidate['dp_count'] = dp_candidate.apply(get_dp_info, axis=1)
        dp_sample = dp_candidate[(dp_candidate['dp_count'] > 0)].copy()
        model_1comp = model_1comp[dp_sample.index]
        left_2comp = left_2comp[dp_sample.index]
        right_2comp = right_2comp[dp_sample.index]

        return dp_sample, model_1comp, left_2comp, right_2comp

    def select_nbcs(self, dp_parent: pd.DataFrame, dp_sample: pd.DataFrame):
        # control sample
        cs_df = dp_parent[~dp_parent.index.isin(dp_sample.index)].copy()

        # no-bias control sample
        z_bins  = np.linspace(dp_sample['Z'].min(), dp_sample['Z'].max(), 21)
        logm_bins = np.linspace(dp_sample['LOGM'].min(), dp_sample['LOGM'].max(), 21)

        # H_dp, _ = np.histogram(dp_sample['Z'], bins=z_bins)
        # H_cs, _ = np.histogram(cs_df['Z'], bins=z_bins)
        H_dp, _, _ = np.histogram2d(dp_sample['Z'], dp_sample['LOGM'], bins=[z_bins, logm_bins])
        H_cs, _, _ = np.histogram2d(cs_df['Z'], cs_df['LOGM'], bins=[z_bins, logm_bins])

        H_cs_safe       = np.where(H_cs == 0, np.inf, H_cs)
        sampling_ratio  = np.minimum(H_dp / H_cs_safe, 1.0)
        
        # z_bin_indices   = np.digitize(cs_df['Z'], bins=z_bins) - 1
        z_bin_indices, logm_bin_indices = np.digitize(cs_df['Z'], bins=z_bins) - 1, np.digitize(cs_df['LOGM'], bins=logm_bins) - 1
        z_bin_indices   = np.clip(z_bin_indices, 0, len(z_bins) - 2)
        logm_bin_indices = np.clip(logm_bin_indices, 0, len(logm_bins) - 2)

        p               = sampling_ratio[z_bin_indices, logm_bin_indices]

        keep_mask       = np.random.rand(len(cs_df)) < p
        matched_cs_indices      = cs_df.index[keep_mask]
        unmatched_cs_indices    = cs_df.index[~keep_mask]
        nbcs_df                 = cs_df.loc[matched_cs_indices].copy()
        cs_nbcs_df              = cs_df.loc[unmatched_cs_indices].copy()
        
        
        dp_cols = ['OII3726_dp', 'OII3729_dp',
                'Hbeta_dp',
                'OIII4959_dp', 'OIII5007_dp',
                'NII6548_dp', 'Halpha_dp', 'NII6583_dp', 
                'SII6716_dp', 'SII6731_dp']
        dp_rank_cols = [f'{col[:-3]}_rank' for col in dp_cols]
        cs_df.drop(columns=dp_cols+dp_rank_cols, inplace=True)
        nbcs_df.drop(columns=dp_cols+dp_rank_cols, inplace=True)
        cs_nbcs_df.drop(columns=dp_cols+dp_rank_cols, inplace=True)
        return cs_df, nbcs_df, cs_nbcs_df

    def get_catalog(self, df: pd.DataFrame, fname: str, model_1comp, left_2comp, right_2comp):
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.BinTableHDU(data=df.to_records(index=False), name='DATA'))
        hdul.append(fits.ImageHDU(data=model_1comp.astype(np.float32), name='1COMP'))
        hdul.append(fits.ImageHDU(data=left_2comp.astype(np.float32), name='2COMP_L'))
        hdul.append(fits.ImageHDU(data=right_2comp.astype(np.float32), name='2COMP_R'))
        hdul.writeto(fname, overwrite=True)

    def extract_fits_data(self, fname: str):
        fits_file = fits.open(fname)
        df = Table(fits_file['DATA'].data).to_pandas()
        model_1comp = fits_file['1COMP'].data
        left_2comp = fits_file['2COMP_L'].data
        right_2comp = fits_file['2COMP_R'].data
        fits_file.close()
        return df, model_1comp, left_2comp, right_2comp