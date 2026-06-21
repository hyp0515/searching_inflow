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
                               id=None, n_components=1, w_dz=False, two_component=None):
        
        # Backward compatibility: convert two_component boolean to n_components
        if two_component is not None:
            n_components = 2 if two_component else 1
        
        idx = data_class.id2index(id)
        df = data_class.df.iloc[idx]
        z = float(df['Z'])
        
        
        line_choices = {
            'OII'       : ([OII_rest[0]-40, OII_rest[1]+40], [(OII_rest[0], OII_rest[1])], 1/1.33),  # fixed ratio for [OII]3727/3729
            'Hbeta'     : ([Hbeta_rest[0]-40, Hbeta_rest[0]+40], [Hbeta_rest[0]], 0),
            'OIII'      : ([OIII_rest[0]-40, OIII_rest[1]+40], [(OIII_rest[0], OIII_rest[1])], 1/3.00),  # fixed ratio for [OIII]4959/5007
            'Halpha'    : ([NII_rest[0]-60, NII_rest[1]+60], [(NII_rest[0], NII_rest[1]), Halpha_rest[0]], 1/3.05), # fixed ratio for [NII]
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
            conti_adjust_estimate_mask = ((lam >= crop_region[i][0]) & (lam <= crop_region[i][0]+15)) | ((lam <= crop_region[i][1]) & (lam >= crop_region[i][1]-15))
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
            """
            Unpack parameters for adaptive number of velocity components.
            
            Parameters are organized as:
            [optional dz], [sigma_1, ..., sigma_n], [dv_1, ..., dv_n], [amp_1, ..., amp_m]
            where n = n_components and m = n_lines_fit * n_components for doublets, else n_lines_fit
            """
            gaussian_parms = [[] for _ in range(len(crop_region))]
            amps = [[] for _ in range(len(crop_region))]
            lam0s = [[] for _ in range(len(crop_region))]
            lam0_adj = 1.0
            
            # Parse fixed parameters
            param_idx = 0
            dz = 0
            if w_dz:
                dz = params[param_idx]
                param_idx += 1
                lam0_adj += dz
            
            # Extract sigma values for each component
            sigmas = []
            for comp_idx in range(n_components):
                sigmas.append(params[param_idx])
                param_idx += 1
            
            # Extract velocity shifts for each component (if n_components > 1)
            dvs = []
            if n_components > 1:
                for comp_idx in range(n_components):
                    dvs.append(params[param_idx])
                    param_idx += 1
            else:
                dvs = [0]  # Single component has no velocity shift
            
            amp_start_index = param_idx
            
            # Build Gaussian parameters for each line region
            for idx_lines, lines in enumerate(lines_to_fit):
                for idx_line, line in enumerate(lines):
                    if not isinstance(line, tuple):  # singlet
                        for comp_idx in range(n_components):
                            amp_idx = idx_line + nline_start_indices[idx_lines] + comp_idx * n_lines_fit + amp_start_index
                            amp = params[amp_idx]
                            lam0 = line * lam0_adj
                            sigma_res = c * 0.8 / (lam0 * (1 + z))
                            sigma_v = sigmas[comp_idx]
                            sigma_combine = np.sqrt(sigma_v**2 + sigma_res**2)
                            dv = dvs[comp_idx]
                            gaussian_parms[idx_lines].append((amp, lam0, dv, sigma_combine))
                            amps[idx_lines].append(amp)
                            lam0s[idx_lines].append(lam0)
                    else:  # doublet
                        line1, line2 = line
                        line_ratio = line_ratios[idx_lines]
                        for comp_idx in range(n_components):
                            amp_idx = idx_line + nline_start_indices[idx_lines] + comp_idx * n_lines_fit + amp_start_index
                            # First line of doublet (scaled by ratio)
                            amp_1 = line_ratio * params[amp_idx]
                            lam0_1 = line1 * lam0_adj
                            sigma_1_res = c * 0.8 / (lam0_1 * (1 + z))
                            sigma_1_v = sigmas[comp_idx]
                            sigma_1_combine = np.sqrt(sigma_1_v**2 + sigma_1_res**2)
                            dv = dvs[comp_idx]
                            gaussian_parms[idx_lines].append((amp_1, lam0_1, dv, sigma_1_combine))
                            amps[idx_lines].append(amp_1)
                            lam0s[idx_lines].append(lam0_1)
                            
                            # Second line of doublet
                            amp_2 = params[amp_idx]
                            lam0_2 = line2 * lam0_adj
                            sigma_2_res = c * 0.8 / (lam0_2 * (1 + z))
                            sigma_2_v = sigmas[comp_idx]
                            sigma_2_combine = np.sqrt(sigma_2_v**2 + sigma_2_res**2)
                            gaussian_parms[idx_lines].append((amp_2, lam0_2, dv, sigma_2_combine))
                            amps[idx_lines].append(amp_2)
                            lam0s[idx_lines].append(lam0_2)
            
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
        sigma_init, sigma_upper, sigma_lower            = 20, 800, 0.001
        amp_init, amp_upper, amp_lower                  = [np.max(combine_flux+combine_conti)/2]*n_lines_fit*n_components, [np.max(combine_flux+combine_conti)]*n_lines_fit*n_components, [0]*n_lines_fit*n_components
        
        # Velocity shift parameters for multiple components
        dv_inits = []
        dv_uppers = []
        dv_lowers = []
        if n_components > 1:
            # Initialize velocity shifts symmetrically around 0
            for comp_idx in range(n_components):
                if comp_idx < n_components // 2:
                    # Left/blue-shifted components (negative velocities)
                    dv_inits.append(-5)
                    dv_lowers.append(-800)
                    dv_uppers.append(150)
                elif comp_idx == n_components // 2 and n_components % 2 == 1:
                    # Middle component (if odd number)
                    dv_inits.append(0)
                    dv_lowers.append(-300)
                    dv_uppers.append(300)
                else:
                    # Right/red-shifted components (positive velocities)
                    dv_inits.append(5)
                    dv_uppers.append(800)
                    dv_lowers.append(-150)
        
        # Build parameter lists
        p0_list = []
        bounds_lower_list = []
        bounds_upper_list = []
        
        if w_dz:
            p0_list.append(dz_init)
            bounds_lower_list.append(dz_lower)
            bounds_upper_list.append(dz_upper)
        
        # Add sigma parameters for each component
        for comp_idx in range(n_components):
            p0_list.append(sigma_init)
            bounds_lower_list.append(sigma_lower)
            bounds_upper_list.append(sigma_upper)
        
        # Add velocity shift parameters for each component (if multiple components)
        if n_components > 1:
            for comp_idx in range(n_components):
                p0_list.append(dv_inits[comp_idx])
                bounds_lower_list.append(dv_lowers[comp_idx])
                bounds_upper_list.append(dv_uppers[comp_idx])
        
        # Add amplitude parameters
        p0_list.extend(amp_init)
        bounds_lower_list.extend(amp_lower)
        bounds_upper_list.extend(amp_upper)
        
        p0 = p0_list
        bounds_lower = bounds_lower_list
        bounds_upper = bounds_upper_list

        popt, pcov = curve_fit(fitting_func, combine_lam, combine_flux, p0=p0, sigma=combine_sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
        # print(popt)
        params = {}
        
        # Parse fitted parameters
        param_idx = 0
        if w_dz:
            params['dz'] = popt[param_idx]
            param_idx += 1
        else:
            params['dz'] = 0
        
        # Extract sigma values for each component
        sigmas_fit = []
        for comp_idx in range(n_components):
            sigmas_fit.append(popt[param_idx])
            param_idx += 1
        
        # Extract velocity shifts for each component
        dvs_fit = []
        if n_components > 1:
            for comp_idx in range(n_components):
                dvs_fit.append(popt[param_idx])
                param_idx += 1
        else:
            dvs_fit = [0]
        
        # Store sigma and velocity parameters
        if n_components == 1:
            params['sigma'] = sigmas_fit[0]
            params['dv'] = None
        else:
            params['sigma'] = tuple(sigmas_fit)
            params['dv'] = tuple(dvs_fit)
            # Also store individual component parameters for backward compatibility
            for comp_idx, (sigma, dv) in enumerate(zip(sigmas_fit, dvs_fit)):
                params[f'sigma_{comp_idx}'] = sigma
                params[f'dv_{comp_idx}'] = dv
        
        params['n_components'] = n_components
        gaussian_parms, amps, lam0s = unpack_params(popt)
        params['gaussian_params'] = gaussian_parms
        params['amps'] = amps
        params['lam0s'] = lam0s
        
        # If multiple components, organize them into component groups
        if n_components > 1:
            # Split components by matching velocity shift (dv)
            # Each Gaussian in gaussian_parms contains (amp, lam0, dv, sigma)
            # We group by the dv value to identify which component each belongs to
            component_list = []
            for comp_idx in range(n_components):
                target_dv = dvs_fit[comp_idx]  # The velocity shift that identifies this component
                comp_dict = {
                    'index': comp_idx,
                    'sigma': sigmas_fit[comp_idx],
                    'dv': dvs_fit[comp_idx],
                    'gaussian_params': [],
                    'amps': [],
                    'lam0s': []
                }
                
                # Gather parameters for this component across all line regions
                for line_region_idx in range(len(gaussian_parms)):
                    region_gaussian_parms = []
                    region_amps = []
                    region_lam0s = []
                    
                    # Extract only Gaussians that belong to this component (matching dv)
                    for param_idx in range(len(gaussian_parms[line_region_idx])):
                        gaussian_param = gaussian_parms[line_region_idx][param_idx]
                        # gaussian_param = (amp, lam0, dv, sigma)
                        param_dv = gaussian_param[2]  # Extract dv from the tuple
                        
                        # Check if this Gaussian belongs to this component
                        if param_dv == target_dv:
                            region_gaussian_parms.append(gaussian_param)
                            region_amps.append(amps[line_region_idx][param_idx])
                            region_lam0s.append(lam0s[line_region_idx][param_idx])
                    
                    comp_dict['gaussian_params'].append(region_gaussian_parms)
                    comp_dict['amps'].append(region_amps)
                    comp_dict['lam0s'].append(region_lam0s)
                component_list.append(comp_dict)
            params['components'] = component_list
        combine_model = fitting_func(combine_lam, *popt)
        return params, (combine_lam, combine_flux, combine_sigma, combine_model), slice_indices, n_lines_fit, conti_adjs