import numpy as np
from scipy.stats import f
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
from .misc import *
from .SPECTRUM import Spectrum
from .FITSPECTRUM import FitSpectrum
from astropy.table import Table
import warnings
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

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
                
    def record_ra_dec(self, df: pd.DataFrame, fname: str):
        with open(fname, 'w+') as f:
            f.write('#? ra dec\n')
            for ra, dec in zip(df['RA'], df['DEC']):
                f.write(f"{ra} {dec}\n")

    def reconstruct_fit(self, data_class:Spectrum, id=None):
        idx = data_class.id2index(id)
        lam = desi_wavelength / (1 + data_class.df.iloc[idx]['Z'])


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
        """
        The calculation chi-square here doesn't include the spectrum outside the fitted region.
        """
        chisq_1comp = np.sum((residual_1comp/combine_sigma)**2)
        dof_1comp = len(combine_lam) - (1+n_lines_fit)
        
        chisq_2comp = np.sum((residual_2comp/combine_sigma)**2)
        dof_2comp = len(combine_lam) - (4+n_lines_fit*2)
        
        F_stat = ((chisq_1comp - chisq_2comp) / (dof_1comp - dof_2comp)) / (chisq_2comp / dof_2comp)
        p_value = 1 - f.cdf(F_stat, dof_1comp - dof_2comp, dof_2comp)


        # Criteria 2: |dv_r-dv_l| > 3 vel_resolution
        dv_r, dv_l = params_2comp['dv_r'], params_2comp['dv_l']
        delta_dv = np.abs(dv_r - dv_l)
        

        # residual_region = np.split(residual_2comp, slice_indices)
        # sigmab_region = [np.std(residual_region[i]) if residual_region[i].size > 1 else 0 for i in range(len(residual_region))]
        noise_region = np.split(combine_sigma, slice_indices)
        sigmab_region = [np.sqrt(np.mean(noise_region[i]**2)) if noise_region[i].size > 1 else -1e10 for i in range(len(noise_region))]


        
        dp_detections = []
        line_snr = []
        
        lams_1comp = []
        flux_1comp = []

        lams_2compL = []
        flux_2compL = []

        lams_2compR = []
        flux_2compR = []
        
        noise_b = []
        for i, amp in enumerate(params_1comp['amps']):
            for j in range(len(amp)):
                
                lams_1comp.append(params_1comp['lam0s'][i][j])
                flux_1comp.append(params_1comp['amps'][i][j]*np.sqrt(2*np.pi)*params_1comp['sigma']*params_1comp['lam0s'][i][j]/c)


                # Criteria 3: 1/3 < amp_r/amp_l < 3
                # Criteria 4: amp_r, amp_l > 3 sigma_background
                amp_l, amp_r = params_2comp['left_amps'][i][j], params_2comp['right_amps'][i][j]
                dp_detections.append((1/3 * amp_l < amp_r < 3 * amp_l)&(amp_r > 3 * sigmab_region[i])&(amp_l > 3 * sigmab_region[i]))

                lams_2compL.append(params_2comp['left_lam0s'][i][j])
                flux_2compL.append(amp_l*np.sqrt(2*np.pi)*params_2comp['sigma_l']*params_2comp['left_lam0s'][i][j]*(1+params_2comp['dv_l']/c)/c)

                lams_2compR.append(params_2comp['right_lam0s'][i][j])
                flux_2compR.append(amp_r*np.sqrt(2*np.pi)*params_2comp['sigma_r']*params_2comp['right_lam0s'][i][j]*(1+params_2comp['dv_r']/c)/c)

                model_2comp_l = model_vel(lams[i], gaussian_parms=[(amp_l, 
                                                                    params_2comp['left_lam0s'][i][j], 
                                                                    params_2comp['dv_l'], 
                                                                    params_2comp['sigma_l'])])
                model_2comp_r = model_vel(lams[i], gaussian_parms=[(amp_r, 
                                                                    params_2comp['right_lam0s'][i][j], 
                                                                    params_2comp['dv_r'], 
                                                                    params_2comp['sigma_r'])])
                model_2comp = model_2comp_l + model_2comp_r
                noise_b.append(3*sigmab_region[i]*np.sqrt(2*np.pi)*params_1comp['sigma']*params_1comp['lam0s'][i][j]/c)
                try:
                    line_snr.append(np.max(model_2comp)/sigmab_region[i])
                except:
                    line_snr.append(0)
                    
        noise_b = np.array(noise_b)
        line_snr_flat = np.where(np.array(line_snr) > 3, np.array(line_snr), -1)
        # non_zero_indices = np.where(line_snr_flat > 0)[0]
        # line_snr_flat[~non_zero_indices] = -1
        
        line_cols = ['OII3729', 'OII3726',
                    'Hbeta',
                    'OIII5007', 'OIII4959',
                    'Halpha', 'NII6583', 'NII6548',
                    'SII6731', 'SII6716']
        df_1comp = pd.DataFrame({'line': line_cols, 
                                 'lam0': lams_1comp, 
                                 'flux_1comp': flux_1comp})
        df_2comp = pd.DataFrame({'lam0': lams_2compL, 
                                 'flux_2compL': flux_2compL, 
                                 'flux_2compR': flux_2compR,
                                 'dp': dp_detections,
                                 'noise_b': noise_b.astype(np.float32),
                                 'dp_snr': line_snr_flat.astype(np.float32)})
        df_line = pd.merge(df_1comp, df_2comp, on='lam0', how='outer').sort_values(by='lam0').reset_index(drop=True)
        return p_value, df_line, params_2comp, params_1comp

    
    def fit_all(self, data_class:Spectrum, n_jobs=5):
        line_cols = ['OII3726', 'OII3729',
                    'Hbeta',
                    'OIII4959', 'OIII5007',
                    'NII6548', 'Halpha', 'NII6583', 
                    'SII6716', 'SII6731']
        dp_cols = [f'{col}_DP' for col in line_cols]
        noise_b_cols = [f'{col}_3NOISE' for col in line_cols]
        dp_snr_cols = [f'{col}_SNR' for col in line_cols]
        flux_1comp_cols = [f'{col}_FLUX_1COMP' for col in line_cols]
        flux_2compL_cols = [f'{col}_FLUX_2COMP_L' for col in line_cols]
        flux_2compR_cols = [f'{col}_FLUX_2COMP_R' for col in line_cols]

        def process_target(target_id):
            """
            Processes a single target to find double-peaked features.
            """
            p_value, df_line, params_2comp, params_1comp = self.fit_dp(data_class=data_class, id=target_id)
            idx = data_class.id2index(target_id)
            Z, RA, DEC, LOGSFR, LOGM = data_class.df.iloc[idx][['Z', 'RA', 'DEC', 'LOGSFR', 'LOGM']]
            lam = desi_wavelength / (1 + Z)
            model_1comp = np.sum([
                model_vel(lam, gaussian_parms=params_1comp['gaussian_params'][i]) for i in range(len(params_1comp['gaussian_params']))
            ], axis=0)
            
            left_2comp = np.sum([
                model_vel(lam, gaussian_parms=params_2comp['left_comp'][i]) for i in range(len(params_2comp['left_comp']))
            ], axis=0)

            right_2comp = np.sum([
                model_vel(lam, gaussian_parms=params_2comp['right_comp'][i]) for i in range(len(params_2comp['right_comp']))
            ], axis=0)
            
            data = {
                   'TARGETID': target_id.astype(np.int64),
                         'RA': RA.astype(np.float32),
                        'DEC': DEC.astype(np.float32),
                          'Z': Z.astype(np.float32),
                       'LOGM': LOGM.astype(np.float32),
                     'LOGSFR': LOGSFR.astype(np.float32),
                       'DV_R': params_2comp['dv_r'].astype(np.float32),
                       'DV_L': params_2comp['dv_l'].astype(np.float32),
                    'SIGMA_R': params_2comp['sigma_r'].astype(np.float32),
                    'SIGMA_L': params_2comp['sigma_l'].astype(np.float32),
                'SIGMA_1COMP': params_1comp['sigma'].astype(np.float32),
                    'P_VALUE': p_value.astype(np.float32),
                'model_1comp': model_1comp.astype(np.float32),
                 'left_2comp': left_2comp.astype(np.float32),
                'right_2comp': right_2comp.astype(np.float32)
            }
            data.update(dict(zip(flux_1comp_cols, df_line['flux_1comp'].to_numpy().astype(np.float32))))
            data.update(dict(zip(flux_2compL_cols, df_line['flux_2compL'].to_numpy().astype(np.float32))))
            data.update(dict(zip(flux_2compR_cols, df_line['flux_2compR'].to_numpy().astype(np.float32))))
            data.update(dict(zip(dp_cols, df_line['dp'])))
            data.update(dict(zip(noise_b_cols, df_line['noise_b'])))
            data.update(dict(zip(dp_snr_cols, df_line['dp_snr'])))
            return data
        # Use joblib to parallelize the processing
        results = Parallel(n_jobs=n_jobs)(delayed(process_target)(target_id) for target_id in tqdm(data_class.targetID))
        
        dp_parent = pd.DataFrame(results)

        model_1comp = np.array(dp_parent['model_1comp'].to_list())
        left_2comp  = np.array(dp_parent['left_2comp'].to_list())
        right_2comp = np.array(dp_parent['right_2comp'].to_list())
        dp_parent.drop(columns=['model_1comp', 'left_2comp', 'right_2comp'], inplace=True)
        return dp_parent, model_1comp, left_2comp, right_2comp
    
    
    def select_dp_sample(self, dp_parent: pd.DataFrame, model_1comp, left_2comp, right_2comp):
        # Criteria 1: p_value < 0.05
        # Criteria 2: |dv_r - dv_l| > 3 * vel_resolution
        # criteria_1 = dp_parent['p_value'] < 0.05
        # criteria_2 = (dp_parent['dv_r'] - dp_parent['dv_l']).abs() > 3 * c * 0.8 / (Halpha_rest[0] * (1 + dp_parent['Z']))
        # dp_candidate = dp_parent[criteria_1 & criteria_2].copy()
        
        
        criteria_1 = dp_parent['P_VALUE'] < 0.05
        # criteria_2 = (dp_parent['dv_r'] - dp_parent['dv_l']).abs() > 3 * c * 0.8 / (Halpha_rest[0] * (1 + dp_parent['Z']))
        dp_candidate = dp_parent[criteria_1].copy()

        line_cols = ['OII3726', 'OII3729',
                    'Hbeta',
                    'OIII4959', 'OIII5007',
                    'NII6548', 'Halpha', 'NII6583', 
                    'SII6716', 'SII6731']
        dp_cols = [f'{col}_DP' for col in line_cols]
        dp_snr_cols = [f'{col}_SNR' for col in line_cols]


        def get_dp_info(row):
            """
            Identifies consecutive double peaks starting from the brightest emission line.
            - Filters out lines with a rank of -1 (not detected).
            - Sorts the remaining emission lines by rank (brightness).
            - If the brightest line does not have a double peak, returns 0 and an empty list.
            - Otherwise, counts consecutive lines with double peaks starting from the brightest
              and returns the count and a list of the corresponding line names.
            """
            # Create a list of (snr, dp_status, line_name) tuples for detected lines (snr != -1)
            line_info = []
            for snr_col, line_col in zip(dp_snr_cols, line_cols):
                snr = row[snr_col]
                if snr > -1:
                    line_info.append((snr, row[line_col+'_DP'], line_col))

            # If no lines were detected, there are no DPs to count.
            if not line_info:
                return 0, []
            
            # Sort by rank (the first element of the tuple)
            line_info.sort(reverse=True)

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
        dp_candidate['dp_count'], _ = zip(*dp_candidate.apply(get_dp_info, axis=1))
        dp_sample = dp_candidate[(dp_candidate['dp_count'] > 0)].copy()
        
        model_1comp = model_1comp[dp_sample.index]
        left_2comp = left_2comp[dp_sample.index]
        right_2comp = right_2comp[dp_sample.index]

        # dp_sample.drop(columns=dp_snr_cols, inplace=True)
        return dp_sample, model_1comp, left_2comp, right_2comp

    

    def select_nbcs(self, dp_parent: pd.DataFrame, dp_sample: pd.DataFrame):
        # control sample
        cs_df = dp_parent[~dp_parent.index.isin(dp_sample.index)].copy()

        # no-bias control sample
        z_bins  = np.linspace(dp_sample['Z'].min(), dp_sample['Z'].max(), 21)
        logm_bins = np.linspace(dp_sample['LOGM'].min(), dp_sample['LOGM'].max(), 21)

        H_dp, _, _ = np.histogram2d(dp_sample['Z'], dp_sample['LOGM'], bins=[z_bins, logm_bins])
        H_cs, _, _ = np.histogram2d(cs_df['Z'], cs_df['LOGM'], bins=[z_bins, logm_bins])

        H_cs_safe       = np.where(H_cs == 0, np.inf, H_cs)
        sampling_ratio  = np.minimum(H_dp / H_cs_safe, 1.0)
        
        z_bin_indices, logm_bin_indices = np.digitize(cs_df['Z'], bins=z_bins) - 1, np.digitize(cs_df['LOGM'], bins=logm_bins) - 1
        z_bin_indices   = np.clip(z_bin_indices, 0, len(z_bins) - 2)
        logm_bin_indices = np.clip(logm_bin_indices, 0, len(logm_bins) - 2)

        p               = sampling_ratio[z_bin_indices, logm_bin_indices]

        keep_mask       = np.random.rand(len(cs_df)) < p
        matched_cs_indices      = cs_df.index[keep_mask]
        unmatched_cs_indices    = cs_df.index[~keep_mask]
        nbcs_df                 = cs_df.loc[matched_cs_indices].copy()
        cs_nbcs_df              = cs_df.loc[unmatched_cs_indices].copy()
        
        return cs_df, nbcs_df, cs_nbcs_df

    def bpt_classification(self, df: pd.DataFrame, two_comp=True):
        classification_map = {
            1: 'SF', 4: 'COMP', 16: 'AGN', 64: 'LINER', 256: 'unclassified',
            2: 'double SF', 8: 'double COMP', 32: 'double AGN', 128: 'double LINER',
            5: 'SF+COMP', 17: 'SF+AGN', 65: 'SF+LINER',
            20: 'COMP+AGN', 68: 'COMP+LINER', 80: 'AGN+LINER',
            257: 'SF+uncertain', 260: 'COMP+uncertain', 272: 'AGN+uncertain', 320: 'LINER+uncertain',
            512: 'unclassified'
        }

        line_cols = ['OIII5007', 'Hbeta', 'NII6583', 'Halpha']
        
        
        def bpt(fluxes, snrs, noises):
            unavailable_lines = []
            line_fluxes = []
            for i, (snr, flux, noise) in enumerate(zip(snrs, fluxes, noises)):
                if snr == -1:
                    unavailable_lines.append(i)
                    line_fluxes.append(noise)
                else:
                    line_fluxes.append(flux if flux != 0 else noise)
                    
            if len(unavailable_lines) > 1:
                return 256, []

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                oiii_hbeta = np.log10(fluxes[0]/fluxes[1])
                nii_halpha = np.log10(fluxes[2]/fluxes[3])

            sf_boundary = 0.61/(nii_halpha-0.05)+1.30
            comp_boundary = 0.61/(nii_halpha-0.47)+1.19
            liner_boundary = 1.05*nii_halpha + 0.45
            if (sf_boundary > oiii_hbeta or comp_boundary > oiii_hbeta) and (nii_halpha < 0.47):
                if (sf_boundary > oiii_hbeta) and (nii_halpha < 0.05):
                    return 1, line_fluxes
                else:
                    return 4, line_fluxes
            elif (sf_boundary < oiii_hbeta or comp_boundary < oiii_hbeta):
                if liner_boundary > oiii_hbeta:
                    return 64, line_fluxes
                else:
                    return 16, line_fluxes
            else:
                return 256, line_fluxes


        df['BPT_1COMP'] = [0] * len(df)
        df['BPT_2COMP_L'] = [0] * len(df)
        df['BPT_2COMP_R'] = [0] * len(df)
        df['BPT_2COMP'] = [0] * len(df)

        bpt_1comp = []
        bpt_2compL = []
        bpt_2compR = []
        for i in range(len(df)):
            line_snr = [df.iloc[i][f'{col}_SNR'] for col in line_cols]
            line_noise_b = [df.iloc[i][f'{col}_3NOISE'] for col in line_cols]
            flux_1comp = [df.iloc[i][f'{col}_FLUX_1COMP'] for col in line_cols]
            flux_2compL = [df.iloc[i][f'{col}_FLUX_2COMP_L'] for col in line_cols]
            flux_2compR = [df.iloc[i][f'{col}_FLUX_2COMP_R'] for col in line_cols]
            bpt_class_val, _ = bpt(flux_1comp, line_snr, line_noise_b)
            bpt_1comp.append(bpt_class_val)

            bpt_class_val, _ = bpt(flux_2compL, line_snr, line_noise_b)
            bpt_2compL.append(bpt_class_val)

            bpt_class_val, _ = bpt(flux_2compR, line_snr, line_noise_b)
            bpt_2compR.append(bpt_class_val)

        df['BPT_1COMP'] = bpt_1comp
        if two_comp is True:
            df['BPT_2COMP_L'] = bpt_2compL
            df['BPT_2COMP_R'] = bpt_2compR
            df['BPT_2COMP'] = df['BPT_2COMP_L'] + df['BPT_2COMP_R']
        return df


    # def extinction(self, df:pd.DataFrame, model_1comp, left_2comp, right_2comp):
    #     # Placeholder for extinction correction logic
    #     # This function should apply extinction correction to the fluxes in the DataFrame
    #     # and update the model_1comp, left_2comp, and right_2comp arrays accordingly.
    #     # The actual implementation will depend on the specific extinction law and parameters used.
    #     return df, model_1comp, left_2comp, right_2comp

    def estimate_SFR(self, df: pd.DataFrame, two_comp=True):
        # Placeholder for SFR estimation logic
        # This function should estimate the SFR based on the provided DataFrame
        # and update the relevant columns accordingly.
        def sfr(flux_halpha, flux_hbeta, flux_hbeta_noise, z):
            """
            the unit of the flux is:
            1e-17 * erg / cm^2 / s
            """
            
            if flux_hbeta < flux_hbeta_noise:
                flux_hbeta = flux_hbeta_noise
            
            ebv = 1.97 * np.log10((flux_halpha/flux_hbeta) / 2.86)
            A_halpha = 3.33 * ebv
            flux_intrinsic = flux_halpha / (10**(A_halpha/-2.5)) * 1e-17
            flux_adjusted = flux_intrinsic * (1+z)
            
            luminous_distance = cosmo.luminosity_distance(z).to('cm').value
            lum_halpha = 4 * np.pi * (luminous_distance**2) * flux_adjusted
            sfr = 5.5e-42 * lum_halpha
            return sfr

        df['LOGSFR_1COMP'] = [np.nan] * len(df)
        df['LOGSFR_2COMP'] = [np.nan] * len(df)

        sfr_1comp = []
        sfr_2compL = []
        sfr_2compR = []
        for i in range(len(df)):
            z = df.iloc[i]['Z']

            if df.iloc[i]['Halpha_SNR'] > 0:
                sfr_1comp.append(sfr(df.iloc[i]['Halpha_FLUX_1COMP'], df.iloc[i]['Hbeta_FLUX_1COMP'], df.iloc[i]['Hbeta_3NOISE'], z))
            else:
                sfr_1comp.append(0)

            if two_comp:
                if df.iloc[i]['Halpha_SNR'] > 0:
                    sfr_2compL.append(sfr(df.iloc[i]['Halpha_FLUX_2COMP_L'], df.iloc[i]['Hbeta_FLUX_2COMP_L'], df.iloc[i]['Hbeta_3NOISE'], z))
                else:
                    sfr_2compL.append(0)

                if df.iloc[i]['Halpha_SNR'] > 0:
                    sfr_2compR.append(sfr(df.iloc[i]['Halpha_FLUX_2COMP_R'], df.iloc[i]['Hbeta_FLUX_2COMP_R'], df.iloc[i]['Hbeta_3NOISE'], z))
                else:
                    sfr_2compR.append(0)

        sfr_1comp = np.array(sfr_1comp)
        sfr_1comp = np.where((sfr_1comp > 0), np.log10(sfr_1comp), -15)
        sfr_1comp = np.where(sfr_1comp > -15, sfr_1comp, -15)
        df['LOGSFR_1COMP'] = sfr_1comp
        
        if two_comp:
            sfr_2comp = np.array(sfr_2compL) + np.array(sfr_2compR)
            sfr_2comp = np.where((sfr_2comp > 0), np.log10(sfr_2comp), -15)
            sfr_2comp = np.where(sfr_2comp > -15, sfr_2comp, -15)
            df['LOGSFR_2COMP'] = sfr_2comp
        
        
        return df

    def get_catalog(self, df: pd.DataFrame, fname: str, model_1comp=None, left_2comp=None, right_2comp=None):
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.BinTableHDU(data=df.to_records(index=False), name='DATA'))
        if model_1comp is not None:
            hdul.append(fits.ImageHDU(data=model_1comp.astype(np.float32), name='1COMP'))
        if left_2comp is not None:
            hdul.append(fits.ImageHDU(data=left_2comp.astype(np.float32), name='2COMP_L'))
        if right_2comp is not None:
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