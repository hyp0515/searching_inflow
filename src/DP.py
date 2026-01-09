import numpy as np
from scipy.stats import f
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from .misc import *
from .SPECTRUM import Spectrum
from .FITSPECTRUM import FitSpectrum

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

    def fit_dp(self, data_class:Spectrum, id=None):
        
        data_stack = data_class.data_stack
        idx = data_class.id2index(id)
        df = data_class.df.iloc[idx]
        
        params_1comp, (combine_lam, combine_flux, combine_sigma), slice_indices, n_lines_fit = FIT.fit_multi_emission_vel(data_class=data_class, id=id, two_component=False, w_dz=False)
        lams = np.split(combine_lam, slice_indices)
        model_1comp = np.concatenate([
                    model_vel(lams[i], gaussian_parms=params_1comp['gaussian_params'][i]) 
                    for i in range(len(lams))
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


    
        # Criteria 2: |dv_r-dv_l| > 3 vel_resolution or 200 km/s
        dv_r, dv_l = params_2comp['dv_r'], params_2comp['dv_l']
        delta_dv = np.abs(dv_r - dv_l)
        

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
                    # Criteria 3: 1/3 < Amp1/Amp2 < 3
                    # Criteria 4: amp > 3 * sigma_b
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
        return p_value, delta_dv, dp_detection, line_fluxes_rank, params_2comp, params_1comp

    def get_dp_candidate(self, data_class:Spectrum, n_jobs=5):
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
            p_value, delta_dv, dp_detection, line_fluxes_rank, params_2comp, params_1comp = self.fit_dp(data_class=data_class, id=target_id)
            idx = data_class.id2index(target_id)
            Z, RA, DEC, LOGSFR, LOGM = data_class.df.iloc[idx][['z', 'RA', 'DEC', 'LOGSFR', 'LOGM']]
            
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
            }
            data.update(dict(zip(dp_cols, dp_detection)))
            data.update(dict(zip(dp_rank_cols, line_fluxes_rank)))
            return data
        results = Parallel(n_jobs=n_jobs)(delayed(process_target)(target_id) for target_id in tqdm(data_class.targetID))
        # Use joblib to parallelize the processing
        dp_parent = pd.DataFrame(results)
        dp_candidate = dp_parent[(dp_parent['p_value'] < 0.05) & (dp_parent['dv_r']-dp_parent['dv_l'] > 75)].copy()
        
        
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
                return 0, []
            
            # Sort by rank (the first element of the tuple)
            line_info.sort()

            # Check if the brightest line (first in the sorted list) has a double peak
            if not line_info[0][1]:  # line_info[0][1] is the dp_status
                return 0, []

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
                
            return actual_dp_count, actual_dp_lines

        # Apply the function to each row to create the new columns
        dp_info = dp_candidate.apply(get_dp_info, axis=1)
        dp_candidate[['dp_count', 'dp_lines']] = pd.DataFrame(dp_info.tolist(), index=dp_candidate.index)
        dp_sample = dp_candidate[(dp_candidate['dp_count'] > 0)].copy()

        dp_parent.drop(columns=dp_cols+dp_rank_cols, inplace=True)
        dp_candidate.drop(columns=dp_cols+dp_rank_cols, inplace=True)
        dp_sample.drop(columns=dp_rank_cols, inplace=True)

        # Update dp_cols in dp_sample to reflect only the consecutive DPs
        for col in dp_cols:
            line_name = col[:-3]
            dp_sample[col] = dp_sample['dp_lines'].apply(lambda lines: line_name in lines)

        dp_sample.drop(['dp_count', 'dp_lines'], axis=1, inplace=True)

        return dp_parent, dp_candidate, dp_sample


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


    def get_catalog(self, data_class:Spectrum, df: pd.DataFrame, fname: str, n_jobs=5):
        
        def process_target_reconstruct(target_id):
            model_1comp, left_2comp, right_2comp = self.reconstruct_fit(data_class=data_class, id=target_id)
            return model_1comp, left_2comp, right_2comp

        results = Parallel(n_jobs=n_jobs)(delayed(process_target_reconstruct)(target_id) for target_id in tqdm(df['TARGETID']))
        
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.BinTableHDU(data=df.to_records(index=False), name='DATA'))
        hdul.append(fits.ImageHDU(data=np.array([res[0] for res in results]), name='1COMP'))
        hdul.append(fits.ImageHDU(data=np.array([res[1] for res in results]), name='2COMP_L'))
        hdul.append(fits.ImageHDU(data=np.array([res[2] for res in results]), name='2COMP_R'))
        hdul.writeto(fname, overwrite=True)