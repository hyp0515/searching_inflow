import numpy as np
from .misc import *
from .SPECTRUM import *
from scipy.optimize import curve_fit
from scipy.stats import f


class FitSpectrum:
    def __init__(self,):
        pass

    def mask_bad_pixel(self, array_to_mask, mask):
        bad_mask = (mask != 0)
        array_clean = array_to_mask[~bad_mask]
        
        return array_clean

    def unmask_bad_pixel(self, array_clean, mask, fill_value=np.nan):
        """
        Reconstructs an array to its original shape before masking,
        inserting a fill value for the masked pixels.

        Parameters:
        array_clean : np.ndarray
            The 1D array of clean data (bad pixels removed).
        mask : np.ndarray
            The original mask array where non-zero values indicate bad pixels.
        fill_value : float, optional
            The value to insert for the bad pixels, by default np.nan.

        Returns:
        np.ndarray
            The reconstructed array with the same shape as the mask.
        """
        # Create an array of the original shape, filled with the fill_value
        unmasked_array = np.full(mask.shape, fill_value, dtype=np.float64)
        
        # Identify the locations of the good pixels
        good_mask = (mask == 0)
        
        # Place the clean data into the good pixel locations
        unmasked_array[good_mask] = array_clean
        
        return unmasked_array
    
    
    
    #
    # this is broken
    #
    def label_emission_lines(self, data_class:Spectrum, s_2_n=3):
        n_spectra = data_class.n_spectra
        
        fluxes = data_class.flux
        # fluxes = data_class.emission
        ivars = data_class.ivar
        masks = data_class.mask

        OII_labels = []
        OIII_labels = []
        Halpha_labels = []
        NII_labels = []
        SII_labels = []
        Hbeta_labels = []
        for idx in range(n_spectra):
            lam_rest = desi_wavelength / (1 + float(data_class.df.iloc[idx]['Z']))
            flux_ini = fluxes[idx, :]
            ivar_ini = ivars[idx, :]
            mask = masks[idx, :]
            for l0, label in zip([np.mean(OII_rest), OIII_rest[1], Halpha_rest[0], NII_rest[1], Hbeta_rest[0], SII_rest[0]],
                                [OII_labels, OIII_labels, Halpha_labels, NII_labels, Hbeta_labels, SII_labels]):
                lam         = self.mask_bad_pixel(lam_rest, mask)
                emission    = self.mask_bad_pixel(flux_ini, mask)
                ivar        = self.mask_bad_pixel(ivar_ini, mask)

                
                crop_flux = emission[(lam >= l0 - 2) & (lam <= l0 + 2)]
                crop_ivar = ivar[(lam >= l0 - 2) & (lam <= l0 + 2)]
                if len(crop_flux) > 0:
                    line_flux = np.max(crop_flux)
                    sigma_b = np.sqrt(np.mean((1/np.sqrt(crop_ivar))**2)) # mean of the background noise
                else:
                    line_flux = 0
                    sigma_b = 1e10

                if line_flux >= s_2_n*sigma_b:
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

    
    

    def fit_multi_emission_vel(self, data_class:Spectrum, 
                               id=None, two_component=False, w_dz=False):
        
        idx = data_class.id2index(id)
        df = data_class.df.iloc[idx]
        z = float(df['Z'])
        
        
        line_choices = {
            'OII'       : ([OII_rest[0]-40, OII_rest[1]+40], [(OII_rest[0], OII_rest[1])], 1/1.33),  # fixed ratio for [OII]3727/3729
            'Hbeta'     : ([Hbeta_rest[0]-40, Hbeta_rest[0]+40], [Hbeta_rest[0]], 0),
            'OIII'      : ([OIII_rest[0]-40, OIII_rest[1]+40], [(OIII_rest[0], OIII_rest[1])], 1/3.00),  # fixed ratio for [OIII]4959/5007
            'Halpha'    : ([NII_rest[0]-40, NII_rest[1]+40], [(NII_rest[0], NII_rest[1]), Halpha_rest[0]], 1/3.05), # fixed ratio for [NII]
            'SII'       : ([SII_rest[0]-40, SII_rest[1]+40], [SII_rest[0], SII_rest[1]], 0)
        }
        
        crop_region = []
        lines_to_fit = []
        line_ratios = []
        
        
        for detected_line in ['OII', 'Hbeta', 'OIII', 'Halpha', 'SII']:
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

        lam = desi_wavelength / (1+z)
        flux, ivar, conti = data_class.flux[idx, :], data_class.ivar[idx, :], data_class.continuum[idx, :]
        lam     = self.mask_bad_pixel(lam, data_class.mask[idx, :])
        flux    = self.mask_bad_pixel(flux, data_class.mask[idx, :])
        ivar    = self.mask_bad_pixel(ivar, data_class.mask[idx, :])
        conti   = self.mask_bad_pixel(conti, data_class.mask[idx, :])

        slice_indices = []
        lams = []
        fluxes = []
        sigmas = []
        contis = []
        conti_adjs = []
        for i in range(len(crop_region)):
            slice_mask = (lam >= crop_region[i][0]) & (lam <= crop_region[i][1])
            conti_adjust_estimate_mask = ((lam >= crop_region[i][0]) & (lam <= crop_region[i][0]+10)) | ((lam <= crop_region[i][1]) & (lam >= crop_region[i][1]-10))
            conti_adjust_estimate = np.median(flux[conti_adjust_estimate_mask]) if np.any(conti_adjust_estimate_mask) else 0
            lams.append(lam[slice_mask])
            fluxes.append(flux[slice_mask]-conti_adjust_estimate)
            sigmas.append(np.sqrt(1/np.abs(ivar[slice_mask])))
            contis.append(conti[slice_mask])
            conti_adjs.append(conti_adjust_estimate)
            slice_indices.append(np.sum(slice_mask))

        if len(slice_indices) > 1:
            slice_indices = np.cumsum(slice_indices)[:-1]
        else:
            slice_indices = []

        combine_lam     = np.concatenate(lams)
        combine_flux    = np.concatenate(fluxes)
        combine_sigma   = np.concatenate(sigmas)
        combine_conti   = np.concatenate(contis)

        def unpack_params(params):
            gaussian_parms = [[] for _ in range(len(crop_region))] # OII, Hbeta, OIII, Halpha, SII
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
                    
            gaussian_parms = [[] for _ in range(len(crop_region))]
            amps = [[] for _ in range(len(crop_region))]
            lam0s = [[] for _ in range(len(crop_region))]
            for idx_lines, lines in enumerate(lines_to_fit):
                for idx_line, line in enumerate(lines):
                    if not isinstance(line, tuple): # not doublet
                        if two_component:
                            # right component
                            
                            amp_r   = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_r  = line * lam0_adj
                            sigma_r_res     = c * 0.8/(lam0_r*(1+z))
                            sigma_r_v       = sigma_1
                            sigma_r_combine = np.sqrt(sigma_r_v**2 + sigma_r_res**2)
                            gaussian_parms[idx_lines].insert(0, (amp_r, lam0_r, dv_r, sigma_r_combine))
                            amps[idx_lines].insert(0, amp_r)
                            lam0s[idx_lines].insert(0, lam0_r)

                            # left component
                            
                            amp_l   = params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                            lam0_l  = line * lam0_adj
                            sigma_l_res     = c * 0.8/(lam0_l*(1+z))
                            sigma_l_v       = sigma_2
                            sigma_l_combine = np.sqrt(sigma_l_v**2 + sigma_l_res**2)
                            gaussian_parms[idx_lines].append((amp_l, lam0_l, dv_l, sigma_l_combine))
                            amps[idx_lines].append(amp_l)
                            lam0s[idx_lines].append(lam0_l)
                        else:
                            amp   = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0  = line * lam0_adj
                            sigma_res       = c * 0.8/(lam0*(1+z))
                            sigma_v         = sigma_1
                            sigma_combine   = np.sqrt(sigma_v**2 + sigma_res**2)
                            gaussian_parms[idx_lines].insert(0, (amp, lam0, 0, sigma_combine))
                            amps[idx_lines].insert(0, amp)
                            lam0s[idx_lines].insert(0, lam0)
                    else: # doublet
                        line1, line2 = line
                        line_ratio = line_ratios[idx_lines]
                            
                        if two_component:
                            
                            amp_1_r     = line_ratio * params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_1_r    = line1 * lam0_adj
                            sigma_1_r_res     = c * 0.8/(lam0_1_r*(1+z))
                            sigma_1_r_v       = sigma_1
                            sigma_1_r_combine = np.sqrt(sigma_1_r_v**2 + sigma_1_r_res**2)
                            gaussian_parms[idx_lines].insert(0, (amp_1_r, lam0_1_r, dv_r, sigma_1_r_combine))
                            amps[idx_lines].insert(0, amp_1_r)
                            lam0s[idx_lines].insert(0, lam0_1_r)

                            amp_2_r     = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_2_r    = line2 * lam0_adj
                            sigma_2_r_res     = c * 0.8/(lam0_2_r*(1+z))
                            sigma_2_r_v       = sigma_1
                            sigma_2_r_combine = np.sqrt(sigma_2_r_v**2 + sigma_2_r_res**2)
                            gaussian_parms[idx_lines].insert(0, (amp_2_r, lam0_2_r, dv_r, sigma_2_r_combine))
                            amps[idx_lines].insert(0, amp_2_r)
                            lam0s[idx_lines].insert(0, lam0_2_r)

                            
                            amp_1_l     = line_ratio * params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                            lam0_1_l    = line1 * lam0_adj
                            sigma_1_l_res     = c * 0.8/(lam0_1_l*(1+z))
                            sigma_1_l_v       = sigma_2
                            sigma_1_l_combine = np.sqrt(sigma_1_l_v**2 + sigma_1_l_res**2)
                            gaussian_parms[idx_lines].append((amp_1_l, lam0_1_l, dv_l, sigma_1_l_combine))
                            amps[idx_lines].append(amp_1_l)
                            lam0s[idx_lines].append(lam0_1_l)

                            amp_2_l     = params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                            lam0_2_l    = line2 * lam0_adj
                            sigma_2_l_res     = c * 0.8/(lam0_2_l*(1+z))
                            sigma_2_l_v       = sigma_2
                            sigma_2_l_combine = np.sqrt(sigma_2_l_v**2 + sigma_2_l_res**2)
                            gaussian_parms[idx_lines].append((amp_2_l, lam0_2_l, dv_l, sigma_2_l_combine))
                            amps[idx_lines].append(amp_2_l)
                            lam0s[idx_lines].append(lam0_2_l)
                        else:
                            amp_1       = line_ratio * params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_1      = line1 * lam0_adj
                            sigma_1_res     = c * 0.8/(lam0_1*(1+z))
                            sigma_1_v       = sigma_1
                            sigma_1_combine = np.sqrt(sigma_1_v**2 + sigma_1_res**2)
                            gaussian_parms[idx_lines].insert(0, (amp_1, lam0_1, 0, sigma_1_combine))
                            amps[idx_lines].insert(0, amp_1)
                            lam0s[idx_lines].insert(0, lam0_1)

                            amp_2       = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                            lam0_2      = line2 * lam0_adj
                            sigma_2_v       = sigma_1
                            sigma_2_res     = c * 0.8/(lam0_2*(1+z))
                            sigma_2_combine = np.sqrt(sigma_2_v**2 + sigma_2_res**2)
                            gaussian_parms[idx_lines].insert(0, (amp_2, lam0_2, 0, sigma_2_combine))
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
        sigma_1_init, sigma_1_upper, sigma_1_lower      = 30, 800, 0.001
        amp_init, amp_upper, amp_lower                  = [np.max(combine_flux+combine_conti)/2]*n_lines_fit, [np.max(combine_flux+combine_conti)]*n_lines_fit, [0]*n_lines_fit
        if two_component:
            sigma_2_init, sigma_2_upper, sigma_2_lower  = sigma_1_init, sigma_1_upper, sigma_1_lower
            dv_r_init, dv_r_upper, dv_r_lower           =  5, 800,    0     # right component
            dv_l_init, dv_l_upper, dv_l_lower           = -5,   0, -800     # left component
            amp_init, amp_upper, amp_lower              = [np.max(combine_flux+combine_conti)/2]*int(n_lines_fit*2), [np.max(combine_flux+combine_conti)]*int(n_lines_fit*2), [0]*int(n_lines_fit*2)

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
            params['sigma'] = (sigma_1, sigma_2)
            params['dv'] = (dv_r, dv_l)
            params['sigma_r'] = sigma_1
            params['sigma_l'] = sigma_2
            params['dv_r'] = dv_r
            params['dv_l'] = dv_l
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
        return params, (combine_lam, combine_flux, combine_sigma), slice_indices, n_lines_fit, conti_adjs