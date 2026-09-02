import numpy as np
from .misc import *
from .SPECTRUM import *
from scipy.optimize import curve_fit
from scipy.stats import f


class FitSpectrum:
    def __init__(self,):
        # Optional hook used by the stability / convergence checks.
        # When set to a callable it is invoked just before curve_fit as
        #     p0_new = p0_perturb(np.asarray(p0), np.asarray(lower), np.asarray(upper))
        # and must return a starting vector of the same length (it is clipped
        # back inside the bounds regardless). Default None -> the fit uses the
        # deterministic default p0 exactly as before. It can be passed per-call
        # to fit_multi_emission_vel, or set once on the instance so that it also
        # takes effect through DP.fit_dp (which calls the shared module FIT).
        self.p0_perturb = None

        # curve_fit / TRF solver knobs (tunable per instance).
        #   fit_ftol, fit_xtol : stopping tolerances. Default 1e-8 keeps the fit
        #       numerically identical to the original code (only the vectorized
        #       model + analytic Jacobian differ, at the ~1e-9 float level). A
        #       10k cross-check showed that loosening to 1e-6 gains only ~2% at
        #       the pipeline level while roughly doubling the number of galaxies
        #       that shift N_COMP vs the original, so it is NOT the default; set
        #       both to 1e-6 per-run if you want the extra speed on a sample
        #       where model-selection reproducibility is not critical.
        #   fit_x_scale : parameter scaling for the trust region. Kept at 1.0
        #       (scipy default); 'jac' was benchmarked and found slightly slower
        #       here because the analytic Jacobian already conditions the steps.
        self.fit_ftol = 1e-8
        self.fit_xtol = 1e-8
        self.fit_x_scale = 1.0

        # Optimization toggles, mainly for cross-checking the speed changes
        # against the original code (e.g. which galaxies shift N_COMP). Both
        # default True (the optimized path). Setting them False restores the
        # original Python-loop model / finite-difference Jacobian. Only take
        # effect when w_dz is False.
        #   use_fast_model   : vectorized model with precomputed resolution
        #   use_analytic_jac : closed-form Jacobian (else curve_fit finite-diff)
        # The four benchmarked "versions" are:
        #   v0 original   : fast=False, ajac=False, ftol=xtol=1e-8
        #   v1 +vectorize : fast=True,  ajac=False, ftol=xtol=1e-8
        #   v2 +jacobian  : fast=True,  ajac=True,  ftol=xtol=1e-8  (current default)
        #   v3 +tolerance : fast=True,  ajac=True,  ftol=xtol=1e-6  (opt-in)
        self.use_fast_model = True
        self.use_analytic_jac = True

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
                               id=None, n_components=1, w_dz=False, two_component=None,
                               p0_perturb=None):

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
                            # Instrumental LSF sigma (km/s) from the DESI resolving
                            # power. R is evaluated at the OBSERVED wavelength
                            # lam0*(1+z); sigma_v itself is frame-invariant.
                            sigma_res = desi_sigma_resolution_vel(lam0 * (1 + z))
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
                            sigma_1_res = desi_sigma_resolution_vel(lam0_1 * (1 + z))
                            sigma_1_v = sigmas[comp_idx]
                            sigma_1_combine = np.sqrt(sigma_1_v**2 + sigma_1_res**2)
                            dv = dvs[comp_idx]
                            gaussian_parms[idx_lines].append((amp_1, lam0_1, dv, sigma_1_combine))
                            amps[idx_lines].append(amp_1)
                            lam0s[idx_lines].append(lam0_1)
                            
                            # Second line of doublet
                            amp_2 = params[amp_idx]
                            lam0_2 = line2 * lam0_adj
                            sigma_2_res = desi_sigma_resolution_vel(lam0_2 * (1 + z))
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

        # ------------------------------------------------------------------ #
        # Fast, vectorized model evaluation (used by curve_fit when w_dz is
        # False, i.e. the line centres are fixed -- which is every call made by
        # the DP pipeline).
        #
        # It is mathematically identical to fitting_func, but factors out the
        # quantities that are CONSTANT across optimizer iterations and computes
        # them once here instead of on every function evaluation:
        #   * each line's velocity grid  v(lam) = c*(lam - lam0)/lam0
        #   * the instrumental-resolution sigma  sigma_res(lam0)
        # Only amp, dv and sigma_v (the actual fit parameters) change per call,
        # and the Gaussians of each region are evaluated as a single broadcast
        # `exp` rather than a Python loop over `gaussian_vel`. This removes the
        # per-evaluation np.interp resolution chain and the per-Gaussian Python
        # overhead that dominate the profile.
        #
        # Layout mirrors unpack_params exactly, so the amp indices, the doublet
        # ratio scaling and the sigma-in-quadrature combination all match.
        # ------------------------------------------------------------------ #
        fast_static = None
        if not w_dz:
            amp_start_fast = n_components + (n_components if n_components > 1 else 0)
            reg_V, reg_sres2, reg_comp, reg_ampidx, reg_ratio = [], [], [], [], []
            for idx_lines, lines in enumerate(lines_to_fit):
                lam_reg = lams[idx_lines]
                Vs, sres2s, comps, ampidxs, ratios = [], [], [], [], []
                for idx_line, line in enumerate(lines):
                    if not isinstance(line, tuple):          # singlet
                        sub = [(line, 1.0)]
                    else:                                     # doublet: both share amp_idx
                        line1, line2 = line
                        sub = [(line1, line_ratios[idx_lines]), (line2, 1.0)]
                    for lam0, r in sub:
                        sres = desi_sigma_resolution_vel(lam0 * (1 + z))
                        v = c * (lam_reg - lam0) / lam0
                        for comp_idx in range(n_components):
                            amp_idx = (idx_line + nline_start_indices[idx_lines]
                                       + comp_idx * n_lines_fit + amp_start_fast)
                            Vs.append(v)
                            sres2s.append(sres * sres)
                            comps.append(comp_idx)
                            ampidxs.append(int(amp_idx))
                            ratios.append(r)
                reg_V.append(np.asarray(Vs, dtype=float))              # (G_i, n_i)
                reg_sres2.append(np.asarray(sres2s, dtype=float))      # (G_i,)
                reg_comp.append(np.asarray(comps, dtype=int))          # (G_i,)
                reg_ampidx.append(np.asarray(ampidxs, dtype=int))      # (G_i,)
                reg_ratio.append(np.asarray(ratios, dtype=float))      # (G_i,)
            fast_static = (reg_V, reg_sres2, reg_comp, reg_ampidx, reg_ratio)

        def fitting_func_fast(lam_grid, *params):
            reg_V, reg_sres2, reg_comp, reg_ampidx, reg_ratio = fast_static
            p = np.asarray(params, dtype=float)
            sig = p[0:n_components]                                    # sigma_v per component
            dvs = p[n_components:2 * n_components] if n_components > 1 else np.zeros(1)
            out = []
            for i in range(len(reg_V)):
                comp = reg_comp[i]
                sigma = np.sqrt(sig[comp] ** 2 + reg_sres2[i])        # (G_i,)
                dv = dvs[comp]                                         # (G_i,)
                amp = reg_ratio[i] * p[reg_ampidx[i]]                  # (G_i,)
                arg = (reg_V[i] - dv[:, None]) / sigma[:, None]        # (G_i, n_i)
                out.append(amp @ np.exp(-0.5 * arg * arg))            # (n_i,)
            return np.concatenate(out)

        # Analytic Jacobian d(model)/d(param), matching fitting_func_fast, so
        # curve_fit no longer needs ~n_params finite-difference model evaluations
        # per iteration. Returns the UNWEIGHTED model derivatives (curve_fit
        # applies the 1/sigma weighting itself). Column order matches p0:
        #   [sigma_v_0..n-1]  [dv_0..n-1 if n>1]  [amp params...]
        # For a Gaussian amp*exp(-0.5 u^2), u=(V-dv)/sigma, sigma=sqrt(sv^2+sres^2),
        # amp=ratio*a:  d/da = ratio*G ;  d/ddv = amp*G*u/sigma ;
        #               d/dsv = amp*G*u^2*sv/sigma^2.
        def jac_fast(lam_grid, *params):
            reg_V, reg_sres2, reg_comp, reg_ampidx, reg_ratio = fast_static
            p = np.asarray(params, dtype=float)
            nparm = p.size
            sig = p[0:n_components]
            multi = n_components > 1
            dvs = p[n_components:2 * n_components] if multi else np.zeros(1)
            rows = []
            for i in range(len(reg_V)):
                comp = reg_comp[i]
                V = reg_V[i]
                G_i, n_i = V.shape
                sigma = np.sqrt(sig[comp] ** 2 + reg_sres2[i])        # (G_i,)
                dv = dvs[comp]                                         # (G_i,)
                amp = reg_ratio[i] * p[reg_ampidx[i]]                  # (G_i,)
                U = (V - dv[:, None]) / sigma[:, None]                 # (G_i, n_i)
                Gmat = np.exp(-0.5 * U * U)                            # (G_i, n_i)
                AG = amp[:, None] * Gmat                               # (G_i, n_i)
                J = np.zeros((n_i, nparm))
                for g in range(G_i):
                    k = comp[g]
                    s_g = sigma[g]
                    J[:, reg_ampidx[i][g]] += reg_ratio[i][g] * Gmat[g]      # d/d amp
                    J[:, k] += AG[g] * U[g] * U[g] * sig[k] / (s_g * s_g)    # d/d sigma_v[k]
                    if multi:
                        J[:, n_components + k] += AG[g] * U[g] / s_g         # d/d dv[k]
                rows.append(J)
            return np.concatenate(rows, axis=0)

        # Select model / Jacobian implementation (toggles only apply when the
        # fast static structure was built, i.e. w_dz is False).
        _use_fast = (not w_dz) and getattr(self, 'use_fast_model', True)
        _use_ajac = (not w_dz) and getattr(self, 'use_analytic_jac', True)
        func_for_fit = fitting_func_fast if _use_fast else fitting_func
        jac_for_fit = jac_fast if _use_ajac else None


        dz_init, dz_upper, dz_lower                     = 0, 1e-3, -1e-3
        sigma_init, sigma_upper, sigma_lower            = 50, 700, 5
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
                    dv_inits.append(-20)
                    dv_lowers.append(-600)
                    dv_uppers.append(200)
                elif comp_idx == n_components // 2 and n_components % 2 == 1:
                    # Middle component (if odd number)
                    dv_inits.append(0)
                    dv_lowers.append(-300)
                    dv_uppers.append(300)
                else:
                    # Right/red-shifted components (positive velocities)
                    dv_inits.append(20)
                    dv_uppers.append(600)
                    dv_lowers.append(-200)
        
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

        # Optional randomized starting point for the convergence/stability tests.
        # Falls back to the per-call argument, then the instance attribute, then
        # None (deterministic default). The returned vector is clipped strictly
        # inside the bounds so curve_fit never rejects it.
        perturb = p0_perturb if p0_perturb is not None else getattr(self, 'p0_perturb', None)
        if perturb is not None:
            lo = np.asarray(bounds_lower, dtype=float)
            hi = np.asarray(bounds_upper, dtype=float)
            p0 = np.asarray(perturb(np.asarray(p0, dtype=float), lo, hi), dtype=float)
            eps = 1e-9 * (hi - lo)
            p0 = np.clip(p0, lo + eps, hi - eps)

        # Solver conditioning / stopping knobs (see notes in fit docs). Read from
        # instance attributes so they can be tuned/benchmarked without code edits;
        # defaults reproduce scipy's behaviour (x_scale=1.0, tol=1e-8).
        _x_scale = getattr(self, 'fit_x_scale', 1.0)
        _ftol = getattr(self, 'fit_ftol', 1e-8)
        _xtol = getattr(self, 'fit_xtol', 1e-8)
        popt, pcov = curve_fit(func_for_fit, combine_lam, combine_flux, p0=p0, sigma=combine_sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True, jac=jac_for_fit, x_scale=_x_scale, ftol=_ftol, xtol=_xtol)
        # print(popt)
        params = {}
        # Parse fitted parameters
        param_idx = 0
        if w_dz:
            params['dz'] = popt[param_idx]
            params['dz_err'] = np.sqrt(pcov[param_idx, param_idx])
            param_idx += 1
        else:
            params['dz'] = [0]
            params['dz_err'] = [0]

        # Extract sigma values for each component
        sigmas_fit = []
        sigma_errors = []
        for comp_idx in range(n_components):
            sigmas_fit.append(popt[param_idx])
            sigma_errors.append(np.sqrt(pcov[param_idx, param_idx]))
            param_idx += 1
        
        # Extract velocity shifts for each component
        dvs_fit = []
        dv_errors = []
        if n_components > 1:
            for comp_idx in range(n_components):
                dvs_fit.append(popt[param_idx])
                dv_errors.append(np.sqrt(pcov[param_idx, param_idx]))
                param_idx += 1
        else:
            dvs_fit = [0]
            dv_errors = [0]

        # Store sigma and velocity parameters
        if n_components == 1:
            params['sigma'] = sigmas_fit[0]
            params['sigma_err'] = sigma_errors[0]
            params['dv'] = (0)
            params['dv_err'] = (0)
        else:
            params['sigma'] = tuple(sigmas_fit)
            params['sigma_err'] = tuple(sigma_errors)
            params['dv'] = tuple(dvs_fit)
            params['dv_err'] = tuple(dv_errors)
            # Also store individual component parameters for backward compatibility
            # for comp_idx, (sigma, sigma_err, dv, dv_err) in enumerate(zip(sigmas_fit, sigma_errors, dvs_fit, dv_errors)):
            #     params[f'sigma_{comp_idx}'] = sigma
            #     params[f'sigma_err_{comp_idx}'] = sigma_err
            #     params[f'dv_{comp_idx}'] = dv
            #     params[f'dv_err_{comp_idx}'] = dv_err

        params['n_components'] = n_components
        gaussian_parms, amps, lam0s = unpack_params(popt)
        _, amps_error, _ = unpack_params(np.sqrt(np.diag(pcov)))
        params['gaussian_params'] = gaussian_parms
        params['amps'] = amps
        params['amps_err'] = amps_error
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
                    'sigma_err': sigma_errors[comp_idx],
                    'dv': dvs_fit[comp_idx],
                    'gaussian_params': [],
                    'amps': [],
                    'amps_err': [],
                    'lam0s': []
                }
                
                # Gather parameters for this component across all line regions
                for line_region_idx in range(len(gaussian_parms)):
                    region_gaussian_parms = []
                    region_amps = []
                    region_amps_err = []
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
                            region_amps_err.append(amps_error[line_region_idx][param_idx])
                            region_lam0s.append(lam0s[line_region_idx][param_idx])
                    
                    comp_dict['gaussian_params'].append(region_gaussian_parms)
                    comp_dict['amps'].append(region_amps)
                    comp_dict['amps_err'].append(region_amps_err)
                    comp_dict['lam0s'].append(region_lam0s)
                component_list.append(comp_dict)
            params['components'] = component_list
        combine_model = fitting_func(combine_lam, *popt)
        return params, (combine_lam, combine_flux, combine_sigma, combine_model), slice_indices, n_lines_fit, conti_adjs