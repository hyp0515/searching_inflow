import numpy as np
import astropy.constants as const
from scipy.optimize import curve_fit

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


def mask_bad_pixel(array_to_mask, mask):
    bad_mask = (mask != 0)
    array_clean = array_to_mask[~bad_mask]
    return array_clean

def unmask_bad_pixel(array_clean, mask, fill_value=np.nan):
    unmasked_array = np.full(mask.shape, fill_value, dtype=np.float64)
    good_mask = (mask == 0)
    unmasked_array[good_mask] = array_clean
    return unmasked_array


def fit_multi_emission(lam_grid, flux, ivar, mask, z, line_to_fit, two_component=False):
    
    
    line_to_fit = {
        'OII'       : ([OII_rest[0]-20, OII_rest[1]+20], [(OII_rest[0], OII_rest[1])], 1/1.33),  # fixed ratio for [OII]3727/3729
        'Hbeta'     : ([Hbeta_rest[0]-20, Hbeta_rest[0]+20], [Hbeta_rest[0]], 0),
        'OIII'      : ([OIII_rest[0]-20, OIII_rest[1]+20], [(OIII_rest[0], OIII_rest[1])], 1/3.00),  # fixed ratio for [OIII]4959/5007
        'Halpha'    : ([NII_rest[0]-20, NII_rest[1]+20], [NII_rest[0], Halpha_rest[0], NII_rest[1]], 0),
        'SII'       : ([SII_rest[0]-20, SII_rest[1]+20], [SII_rest[0], SII_rest[1]], 0)
    }
    
    crop_region = []
    lines_to_fit = []
    line_ratios = []
    for fit_line in line_to_fit.keys():
        crop_region.append(line_to_fit[fit_line][0])
        lines_to_fit.append(line_to_fit[fit_line][1])
        line_ratios.append(line_to_fit[fit_line][2])

    
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

    lam     = mask_bad_pixel(lam_grid, mask)
    flux    = mask_bad_pixel(    flux, mask)
    ivar    = mask_bad_pixel(    ivar, mask)

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
        if two_component:
            sigma_1, sigma_2, dv_r, dv_l = params[:4]
            amp_start_index = 4
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
                        
                        amp_r   = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                        lam0_r  = line
                        sigma_r_res     = c * 0.8/(lam0_r*(1+z)) # resolution
                        sigma_r_v       = sigma_1 # intrinsic
                        sigma_r_combine = np.sqrt(sigma_r_v**2 + sigma_r_res**2)
                        gaussian_parms[idx_lines].insert(0, (amp_r, lam0_r, dv_r, sigma_r_combine))
                        amps[idx_lines].insert(0, amp_r)
                        lam0s[idx_lines].insert(0, lam0_r)

                        # left component
                        
                        amp_l   = params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                        lam0_l  = line
                        sigma_l_res     = c * 0.8/(lam0_l*(1+z)) # resolution
                        sigma_l_v       = sigma_2 # intrinsic
                        sigma_l_combine = np.sqrt(sigma_l_v**2 + sigma_l_res**2)
                        gaussian_parms[idx_lines].append((amp_l, lam0_l, dv_l, sigma_l_combine))
                        amps[idx_lines].append(amp_l)
                        lam0s[idx_lines].append(lam0_l)
                    else:
                        amp   = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                        lam0  = line
                        sigma_res       = c * 0.8/(lam0*(1+z)) # resolution
                        sigma_v         = sigma_1 # intrinsic
                        sigma_combine   = np.sqrt(sigma_v**2 + sigma_res**2)
                        gaussian_parms[idx_lines].insert(0, (amp, lam0, 0, sigma_combine))
                        amps[idx_lines].insert(0, amp)
                        lam0s[idx_lines].insert(0, lam0)
                else: # doublet
                    line1, line2 = line
                    line_ratio = line_ratios[idx_lines]
                        
                    if two_component:
                        
                        amp_1_r     = line_ratio * params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                        lam0_1_r    = line1
                        sigma_1_r_res     = c * 0.8/(lam0_1_r*(1+z)) # resolution
                        sigma_1_r_v       = sigma_1 # intrinsic
                        sigma_1_r_combine = np.sqrt(sigma_1_r_v**2 + sigma_1_r_res**2)
                        gaussian_parms[idx_lines].insert(0, (amp_1_r, lam0_1_r, dv_r, sigma_1_r_combine))
                        amps[idx_lines].insert(0, amp_1_r)
                        lam0s[idx_lines].insert(0, lam0_1_r)

                        amp_2_r     = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                        lam0_2_r    = line2
                        sigma_2_r_res     = c * 0.8/(lam0_2_r*(1+z)) # resolution
                        sigma_2_r_v       = sigma_1 # intrinsic
                        sigma_2_r_combine = np.sqrt(sigma_2_r_v**2 + sigma_2_r_res**2)
                        gaussian_parms[idx_lines].insert(0, (amp_2_r, lam0_2_r, dv_r, sigma_2_r_combine))
                        amps[idx_lines].insert(0, amp_2_r)
                        lam0s[idx_lines].insert(0, lam0_2_r)

                        
                        amp_1_l     = line_ratio * params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                        lam0_1_l    = line1
                        sigma_1_l_res     = c * 0.8/(lam0_1_l*(1+z)) # resolution
                        sigma_1_l_v       = sigma_2 # intrinsic
                        sigma_1_l_combine = np.sqrt(sigma_1_l_v**2 + sigma_1_l_res**2)
                        gaussian_parms[idx_lines].append((amp_1_l, lam0_1_l, dv_l, sigma_1_l_combine))
                        amps[idx_lines].append(amp_1_l)
                        lam0s[idx_lines].append(lam0_1_l)

                        amp_2_l     = params[idx_line+nline_start_indices[idx_lines]+n_lines_fit+amp_start_index]
                        lam0_2_l    = line2
                        sigma_2_l_res     = c * 0.8/(lam0_2_l*(1+z)) # resolution
                        sigma_2_l_v       = sigma_2 # intrinsic
                        sigma_2_l_combine = np.sqrt(sigma_2_l_v**2 + sigma_2_l_res**2)
                        gaussian_parms[idx_lines].append((amp_2_l, lam0_2_l, dv_l, sigma_2_l_combine))
                        amps[idx_lines].append(amp_2_l)
                        lam0s[idx_lines].append(lam0_2_l)
                    else:
                        amp_1       = line_ratio * params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                        lam0_1      = line1
                        sigma_1_res     = c * 0.8/(lam0_1*(1+z)) # resolution
                        sigma_1_v       = sigma_1 # intrinsic
                        sigma_1_combine = np.sqrt(sigma_1_v**2 + sigma_1_res**2)
                        gaussian_parms[idx_lines].insert(0, (amp_1, lam0_1, 0, sigma_1_combine))
                        amps[idx_lines].insert(0, amp_1)
                        lam0s[idx_lines].insert(0, lam0_1)

                        amp_2       = params[idx_line+nline_start_indices[idx_lines]+amp_start_index]
                        lam0_2      = line2
                        sigma_2_v       = sigma_1 # intrinsic
                        sigma_2_res     = c * 0.8/(lam0_2*(1+z)) # resolution
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
    

    sigma_1_init, sigma_1_upper, sigma_1_lower      = 30, 300, 0.01
    amp_init, amp_upper, amp_lower                  = [np.max(combine_flux)/2]*n_lines_fit, [np.max(combine_flux)]*n_lines_fit, [0]*n_lines_fit
    if two_component:
        sigma_2_init, sigma_2_upper, sigma_2_lower  = sigma_1_init, sigma_1_upper, sigma_1_lower
        dv_r_init, dv_r_upper, dv_r_lower           =  5, 500,    0     # right component
        dv_l_init, dv_l_upper, dv_l_lower           = -5,   0, -500     # left component
        amp_init, amp_upper, amp_lower              = [np.max(combine_flux)/2]*int(n_lines_fit*2), [np.max(combine_flux)]*int(n_lines_fit*2), [0]*int(n_lines_fit*2)


    if two_component:
        p0 = [sigma_1_init, sigma_2_init, dv_r_init, dv_l_init] + amp_init
        bounds_lower = [sigma_1_lower, sigma_2_lower, dv_r_lower, dv_l_lower] + amp_lower
        bounds_upper = [sigma_1_upper, sigma_2_upper, dv_r_upper, dv_l_upper] + amp_upper
    else:
        p0 = [sigma_1_init] + amp_init
        bounds_lower = [sigma_1_lower] + amp_lower
        bounds_upper = [sigma_1_upper] + amp_upper

    popt, pcov = curve_fit(fitting_func, combine_lam, combine_flux, p0=p0, sigma=combine_sigma, bounds=(bounds_lower, bounds_upper), absolute_sigma=True)
    # print(popt)
    params = {}
    if two_component:
        sigma_1, sigma_2, dv_r, dv_l = popt[:4]
        params['dz'] = 0
        params['sigma'] = (sigma_1, sigma_2)
        params['dv'] = (dv_r, dv_l)
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
