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

# ---------------------------------------------------------------------------
# Canonical line bookkeeping.
#
# The order below MUST match the order in which FITSPECTRUM builds
# `gaussian_params` (region by region, line by line within a region):
#   region 0  OII    -> OII3726, OII3729      (doublet)
#   region 1  Hbeta  -> Hbeta
#   region 2  OIII   -> OIII4959, OIII5007    (doublet)
#   region 3  Halpha -> NII6548, NII6583, Halpha   (NII doublet, then Halpha)
#   region 4  SII    -> SII6716, SII6731
# ---------------------------------------------------------------------------
LINE_NAMES = ['OII3726', 'OII3729',
              'Hbeta',
              'OIII4959', 'OIII5007',
              'NII6548', 'NII6583', 'Halpha',
              'SII6716', 'SII6731']

# which fit region each flattened line lives in (for per-region background noise)
REGION_OF_LINE = [0, 0, 1, 2, 2, 3, 3, 3, 4, 4]

N_MODELS = 4          # try N = 1..4 components
PARAMS_PER_COMP = 9   # used only for the BIC penalty (matches your DP_new stub)

# --- model-selection thresholds -------------------------------------------
# A model is "decisively" preferred over another when their BIC differs by
# more than this. Below it, the two models are statistically indistinguishable
# and we let component-level SNR break the tie (see fit_dp).
DELTA_BIC_DECISIVE = 15.0
# A more-complex candidate model only "qualifies" over a simpler one if
# EVERY one of its components is jointly detected above SNR_SIG_THRESHOLD on
# at least one of these anchor lines (all components on OIII5007, OR all
# components on Halpha -- not a mix of different lines per component).
QUALIFY_ANCHOR_LINES = ['OIII5007', 'Halpha', 'NII6583']
SNR_SIG_THRESHOLD = 3.0
# FITSPECTRUM fits velocity sigma with hard bounds of 1-800 km/s (see
# sigma_lower / sigma_upper in FITSPECTRUM.fit_multi_emission_vel). A
# component whose fitted sigma is pinned at/near either bound is a sign the
# optimizer collapsed it against a fit limit rather than resolving a real
# component.
SIGMA_FLOOR = 5.0
SIGMA_FLOOR_ATOL = 0.001
SIGMA_CEIL = 699.0
# SIGMA_CEIL_ATOL = 0.001
# A model with too many broad (likely blended/degenerate) components is also
# rejected: more than MAX_WIDE_COMPONENTS components wider than
# WIDE_SIGMA_THRESHOLD km/s.
WIDE_SIGMA_THRESHOLD = 300.0
MAX_WIDE_COMPONENTS = 1
# A model is also rejected if any two of its components are practically the
# same component (degenerate split): their sigmas within a factor of
# SIGMA_RATIO_DEGENERATE of each other AND their dv within
# DV_DIFF_DEGENERATE km/s.
SIGMA_RATIO_DEGENERATE = 1.5
DV_DIFF_DEGENERATE = 60.0
# ...or, regardless of ratio, if both components are individually narrow
# (sigma below this) and within DV_DIFF_DEGENERATE km/s -- two narrow lines
# this close together aren't resolvable as distinct kinematic components.
NARROW_SIGMA_DEGENERATE = 60.0
# Between the 2- and 4-component models specifically: if their BIC is this
# close, the extra complexity of N=4 isn't decisively justified, so N=2 wins
# by Occam's razor regardless of the general qualify-based tie-break above.
DELTA_BIC_SIMPLICITY = 5.0

# --- Stage-4 kinematic (line-profile shape) classification -----------------
# Widths used here are the DECONVOLVED velocity sigmas stored in
# dp_components['SIGMA']. For classification we floor them at the DESI
# resolution: a component narrower than we can resolve is treated as
# "= resolution", so unresolved spikes (sigma pinned near SIGMA_FLOOR) cannot
# blow up width ratios.
CLASS_SIGMA_FLOOR = 30.0        # km/s, ~DESI resolution at these redshifts
# A component is "broad" (turbulent outflow phase) on an ABSOLUTE width cut,
# OR (for n>=3) on a RELATIVE cut: the single widest component counts as broad
# when widest / second-widest exceeds SD_WIDTH_RATIO_MAX, even below the
# absolute cut. The second-widest reference (not the narrowest) avoids the
# unresolved-narrow blow-up and won't mislabel a moderately-wide central head.
BROAD_SIGMA_THRESHOLD = 200.0   # km/s
# 'SD' (symmetric double): two NON-broad components of COMPARABLE width
# (max/min sigma below this, after flooring) that straddle the internal
# reference velocity. Flux ratio is intentionally NOT required. EXCEPTION: a
# pair with BOTH components in the narrow regime (sigma < WIDTH_NARROW_MAX)
# also counts as SD regardless of this ratio.
SD_WIDTH_RATIO_MAX = 2.0
# ...and the SD pair must be genuinely split, so a near-central/degenerate pair
# doesn't masquerade as a double. Measured differently by component count:
#   n == 2  : PEAK-TO-PEAK separation |dv_j - dv_i| must exceed this (v_ref is
#             the mean of just these two, so an unequal flux ratio would drag it
#             onto the brighter peak and break a per-component offset test).
#   n >= 3  : each component's own |dv - v_ref| must exceed this.
SD_MIN_SEP = 60.0               # km/s
# 'CEN' (central component): a non-broad component lying BETWEEN the two SD
# (rim) components -- the central head. No distance-to-reference tolerance is
# used (per Fred): sitting between the rotational pair is sufficient.
# '2-pairs': components split into two velocity groups separated by more than
# this, each group carrying an internal width contrast (a narrow + a broader
# component) -> two outflow systems.
TWO_SYS_MIN_SEP = 150.0         # km/s
TWO_SYS_CONTRAST = 2.0          # within-group max/min sigma to count as a pair
# '3-width': three mutually distinct widths -- every consecutive (sorted)
# width ratio must exceed this.
DISTINCT_WIDTH_RATIO = 2.0
# 2-component, non-SD systems are split further by the narrow/wide AMPLITUDE
# ratio, measured on the brightest line (a consistent anchor across components):
#   r < AMP_RATIO_AMBIG                  -> 'ambiguous' (the wide component
#                                           out-peaks the narrow one)
#   AMP_RATIO_AMBIG <= r <= AMP_RATIO_OUTFLOW -> '2-width'
#   r > AMP_RATIO_OUTFLOW                -> 'outflow' (sharp core + faint wing)
AMP_RATIO_AMBIG = 1/1.5
AMP_RATIO_OUTFLOW = 1.5
# A symmetric double ('rotation') is split further by the peak-amplitude ratio
# of its two components. The bands are symmetric about 1, so the result does
# not depend on which component is the numerator:
#   1/ROT_AMP_SYM  <= r <= ROT_AMP_SYM   -> 'rotation'
#   1/ROT_AMP_ASYM <= r <  1/ROT_AMP_SYM  (or the mirror band) -> 'asymmetry'
#   beyond ROT_AMP_ASYM                   -> 'ambiguous'
ROT_AMP_SYM = 2.0
ROT_AMP_ASYM = 4.0
# Descriptive width census reported alongside the class: every component is
# binned as narrow (N), medium (M) or broad (B) purely by width, so the three
# counts always partition the components (CNT_N + CNT_M + CNT_B == N_KIN).
# NB: independent of the is_broad flag used for classification (which also
# carries the RELATIVE broad rule) -- CNT_B is a pure width count.
WIDTH_NARROW_MAX = 90.0        # km/s, N: sigma < 90
WIDTH_MEDIUM_MAX = 200.0        # km/s, M: 90 <= sigma < 200 ; B: sigma >= 200


class DP:
    def __init__(self):
        pass

    # ------------------------------------------------------------------ I/O
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

    # ================================================================== #
    #  1. Generic component view                                         #
    # ================================================================== #
    @staticmethod
    def _flatten(region_nested):
        """[[a, b], [c], ...]  ->  [a, b, c, ...]"""
        return [x for region in region_nested for x in region]

    def _iter_components(self, params):
        """
        Normalise a FITSPECTRUM `params` dict (any N) into a list of
        per-component dicts, sorted blue -> red by velocity.

        Each dict has flat, length-10 arrays aligned with LINE_NAMES:
            comp_dv, comp_sigma,
            gp_flat      : [(amp, lam0, dv, sigma_combine), ...]  (for model building)
            amps_flat, amps_err_flat, lam0s_flat
        This is the ONLY place that needs to know about the N=1 vs N>1
        difference in the params layout, so everything downstream is
        component-count agnostic.
        """
        n = params.get('n_components', 1)

        if n == 1:
            comps = [{
                'dv': 0.0,
                'sigma': float(params['sigma']),
                'gaussian_params': params['gaussian_params'],
                'amps': params['amps'],
                'amps_err': params['amps_err'],
                'lam0s': params['lam0s'],
            }]
        else:
            comps = params['components']

        out = []
        for c in comps:
            out.append({
                'comp_dv':       float(c['dv']),
                'comp_sigma':    float(c['sigma']),
                'gp_flat':       self._flatten(c['gaussian_params']),
                'amps_flat':     self._flatten(c['amps']),
                'amps_err_flat': self._flatten(c['amps_err']),
                'lam0s_flat':    self._flatten(c['lam0s']),
            })

        # canonical ordering: bluest (most negative dv) first
        out.sort(key=lambda d: d['comp_dv'])
        return out

    @staticmethod
    def _per_region_noise(csigma, slice_idx):
        """RMS background noise in each fit region, from the per-pixel sigma."""
        noise_region = np.split(csigma, slice_idx)
        return [np.sqrt(np.mean(seg ** 2)) if seg.size > 1 else np.inf
                for seg in noise_region]

    def _component_sig_flags(self, params, sigmab_region):
        """
        For a given N-component fit, return a list (one entry per component,
        bluest first) of length-len(LINE_NAMES) boolean lists flagging which
        lines are individually detected above SNR_SIG_THRESHOLD.
        Used both by _model_qualifies (to decide whether a model "qualifies")
        and to later drop components with zero True entries.
        """
        comps = self._iter_components(params)
        flags = []
        for comp in comps:
            comp_flags = []
            for li in range(len(LINE_NAMES)):
                amp = comp['gp_flat'][li][0]
                sig_b = sigmab_region[REGION_OF_LINE[li]]
                comp_flags.append(bool(np.isfinite(sig_b) and amp > SNR_SIG_THRESHOLD * sig_b))
            flags.append(comp_flags)
        return flags

    def _component_sigmas(self, params):
        """Per-component velocity sigma (km/s), bluest first -- same order as _component_sig_flags."""
        return [comp['comp_sigma'] for comp in self._iter_components(params)]

    def _component_dvs(self, params):
        """Per-component velocity offset (km/s), bluest first -- same order as _component_sig_flags."""
        return [comp['comp_dv'] for comp in self._iter_components(params)]

    @staticmethod
    def _model_qualifies(flags):
        """
        `flags`: list of per-component boolean lists (len(LINE_NAMES)), from
        _component_sig_flags. The candidate model qualifies only if EVERY
        one of its components is significant (>3 SNR) on the same anchor
        line -- i.e. all components clear OIII5007, or all components clear
        Halpha (not a mix of different lines per component).
        """
        if not flags:
            return False
        anchor_idx = [LINE_NAMES.index(name) for name in QUALIFY_ANCHOR_LINES]
        return any(all(comp_flags[li] for comp_flags in flags) for li in anchor_idx)

    def _choose_n_components(self, bic_by_n, sig_flags_by_n, comp_sigmas_by_n, comp_dvs_by_n):
        """
        Model selection, repeated against a shrinking candidate pool:
          1. n_min = argmin(BIC) among remaining candidates. If every other
             candidate's BIC is more than DELTA_BIC_DECISIVE worse, tentatively
             accept n_min.
          2. Otherwise, among n_min and its "competitive" alternatives
             (deltaBIC < DELTA_BIC_DECISIVE), prefer the one with the most
             components that _model_qualifies (ALL of its components
             significant on OIII5007 or ALL significant on Halpha), falling
             back to n_min if none qualify.
          2b. Special case: if this lands on N=4 and N=2 is also a candidate,
              but their BIC differ by less than DELTA_BIC_SIMPLICITY, N=4's
              extra complexity isn't decisively justified over N=2 even
              though it "qualified" above -- fall back to N=2 outright.
          Whatever is tentatively chosen -- even a decisive BIC winner with
          no competitive alternatives -- is then rejected outright, dropped
          from the pool, and re-picked from what remains if it either:
            - does NOT _model_qualifies (fails the OIII5007/Halpha anchor
              check above), or
            - has ANY component whose fitted velocity sigma is pinned at/near
              SIGMA_FLOOR or SIGMA_CEIL, or
            - has more than MAX_WIDE_COMPONENTS components wider than
              WIDE_SIGMA_THRESHOLD, or
            - has any two components that are practically the same component
              -- |dv difference| < DV_DIFF_DEGENERATE km/s AND EITHER their
              sigma ratio < SIGMA_RATIO_DEGENERATE OR both are individually
              narrower than NARROW_SIGMA_DEGENERATE (two narrow lines that
              close together aren't resolvable regardless of their ratio) --
              a sign the fit split one real component into two near-identical
              ones rather than resolving
              two genuinely distinct components.
          This can cascade all the way down to n=1: N=1 is the floor of the
          model space (there's nothing simpler to fall back to), so it is
          always accepted once reached, regardless of these checks.
        """
        def has_pinned_sigma(sigmas):
            return sigmas and any((s < SIGMA_FLOOR + SIGMA_FLOOR_ATOL) or (s > SIGMA_CEIL) for s in sigmas)

        def too_many_wide(sigmas):
            return sigmas and sum(s > WIDE_SIGMA_THRESHOLD for s in sigmas) > MAX_WIDE_COMPONENTS

        def has_degenerate_pair(sigmas, dvs):
            for i in range(len(sigmas)):
                for j in range(i + 1, len(sigmas)):
                    if abs(dvs[i] - dvs[j]) >= DV_DIFF_DEGENERATE:
                        continue
                    if sigmas[i] < NARROW_SIGMA_DEGENERATE and sigmas[j] < NARROW_SIGMA_DEGENERATE:
                        return True
                    lo, hi = sorted((sigmas[i], sigmas[j]))
                    if lo > 0 and (hi / lo) < SIGMA_RATIO_DEGENERATE:
                        return True
            return False

        candidates = {n for n in range(1, N_MODELS + 1) if np.isfinite(bic_by_n[n - 1])}
        fallback = min(candidates, key=lambda n: bic_by_n[n - 1]) if candidates else 1

        while candidates:
            n_min = min(candidates, key=lambda n: bic_by_n[n - 1])
            competitive = [n for n in candidates
                           if n != n_min and (bic_by_n[n - 1] - bic_by_n[n_min - 1]) < DELTA_BIC_DECISIVE]

            chosen = n_min
            for n in sorted({n_min, *competitive}, reverse=True):
                if self._model_qualifies(sig_flags_by_n[n - 1]):
                    chosen = n
                    break

            # 2 vs 4 specifically: BIC too close to justify the extra
            # complexity -> prefer the simpler 2-component model outright.
            if chosen == 4 and 2 in candidates and abs(bic_by_n[3] - bic_by_n[1]) < DELTA_BIC_SIMPLICITY:
                chosen = 2

            sigmas_chosen = comp_sigmas_by_n[chosen - 1]
            dvs_chosen = comp_dvs_by_n[chosen - 1]
            fails_anchor_lines = not self._model_qualifies(sig_flags_by_n[chosen - 1])
            if chosen > 1 and (has_pinned_sigma(sigmas_chosen) or too_many_wide(sigmas_chosen)
                               or fails_anchor_lines or has_degenerate_pair(sigmas_chosen, dvs_chosen)):
                candidates.discard(chosen)
                continue

            return chosen

        return fallback

    # ================================================================== #
    #  2. Fit one target, choose N, emit tidy records                    #
    # ================================================================== #
    def fit_dp(self, data_class: Spectrum, id=None):
        """
        Returns
        -------
        parent_row : dict          one scalar row for this target
        comp_rows  : list[dict]    one row per (component, line)   [tidy/long]
        full_model : np.ndarray    best-fit total model on the full DESI grid
        """
        idx = data_class.id2index(id)
        row = data_class.df.iloc[idx]
        Z, RA, DEC, LOGM, LOGSFR = row[['Z', 'RA', 'DEC', 'LOGM', 'LOGSFR']]
        Z = float(Z)

        # --- fit N = 1..N_MODELS and score with BIC (your stub logic) -------
        params_by_n, chisq_by_n, bic_by_n = [], [], []
        best_extras = None  # keep combine arrays / slice_indices of each fit
        extras_by_n = []
        for i in range(N_MODELS):
            n_comp = i + 1
            try:
                param, (clam, cflux, csigma, cmodel), slice_idx, n_lines, conti_adjs = \
                    FIT.fit_multi_emission_vel(data_class=data_class, id=id,
                                               n_components=n_comp, w_dz=False)
                k = PARAMS_PER_COMP * n_comp
                chisq = np.sum(((cflux - cmodel) / csigma) ** 2)
                bic = k * np.log(len(cflux)) + chisq
                params_by_n.append(param)
                chisq_by_n.append(chisq)
                bic_by_n.append(bic)
                extras_by_n.append((clam, cflux, csigma, slice_idx))
            except Exception:
                params_by_n.append(None)
                chisq_by_n.append(np.inf)
                bic_by_n.append(np.inf)
                extras_by_n.append(None)

        # --- per-component SNR flags & sigmas for every N, then pick N -------
        sigmab_region_by_n = [None] * N_MODELS
        sig_flags_by_n = [None] * N_MODELS
        comp_sigmas_by_n = [None] * N_MODELS
        comp_dvs_by_n = [None] * N_MODELS
        for i in range(N_MODELS):
            if params_by_n[i] is None:
                continue
            _, _, csigma_i, slice_idx_i = extras_by_n[i]
            sigmab_region_by_n[i] = self._per_region_noise(csigma_i, slice_idx_i)
            sig_flags_by_n[i] = self._component_sig_flags(params_by_n[i], sigmab_region_by_n[i])
            comp_sigmas_by_n[i] = self._component_sigmas(params_by_n[i])
            comp_dvs_by_n[i] = self._component_dvs(params_by_n[i])

        best_n = self._choose_n_components(bic_by_n, sig_flags_by_n, comp_sigmas_by_n, comp_dvs_by_n)
        n_fit = best_n  # number of components actually fit (before cleanup)
        best_params = params_by_n[best_n - 1]
        clam, cflux, csigma, slice_idx = extras_by_n[best_n - 1]
        sigmab_region = sigmab_region_by_n[best_n - 1]
        sig_flags = sig_flags_by_n[best_n - 1]

        # --- flag (but keep) components with zero significantly-detected lines ---
        comps = self._iter_components(best_params)
        comp_insig = [sum(flags) == 0 for flags in sig_flags]  # no line above SNR_SIG_THRESHOLD
        best_n = len(comps)

        # --- build the tidy component rows ---------------------------------
        lam_full = desi_wavelength / (1 + Z)

        comp_rows = []
        full_model = np.zeros_like(lam_full)
        line_flux_total = {name: 0.0 for name in LINE_NAMES}

        for comp_idx, comp in enumerate(comps):
            # accumulate this component into the reconstructed spectrum
            full_model = full_model + model_vel(lam_full, gaussian_parms=comp['gp_flat'])

            for li, name in enumerate(LINE_NAMES):
                amp, lam0, dv, _sigma_comb = comp['gp_flat'][li]
                amp_err = comp['amps_err_flat'][li]
                reg = REGION_OF_LINE[li]
                sig_b = sigmab_region[reg]

                # integrated line flux of this component.
                # NB: uses the *velocity* sigma (comp_sigma), matching the
                # convention of the original two-component code.
                f_line = flux(amp, comp['comp_sigma'], lam0, dv=dv)
                f_err = flux(amp_err, comp['comp_sigma'], lam0, dv=dv)
                noise3 = flux(SNR_SIG_THRESHOLD * sig_b, comp['comp_sigma'], lam0, dv=dv)

                line_flux_total[name] += f_line

                comp_rows.append({
                    'TARGETID':  np.int64(row['TARGETID']),
                    'N_COMP':    best_n,
                    'COMP_IDX':  comp_idx,          # 0 = bluest
                    'LINE':      name,
                    'LAM0':      np.float32(lam0),
                    'DV':        np.float32(comp['comp_dv']),
                    'SIGMA':     np.float32(comp['comp_sigma']),
                    'AMP':       np.float32(amp),
                    'FLUX':      np.float32(f_line),
                    'FLUX_ERR':  np.float32(f_err),
                    'SNR':       np.float32(amp / sig_b if np.isfinite(sig_b) else 0.0),
                    'SIG':       bool(amp > SNR_SIG_THRESHOLD * sig_b),   # detection flag
                    'NOISE3':    np.float32(noise3),
                    'COMP_INSIG': bool(comp_insig[comp_idx]),  # whole component has no line above SNR_SIG_THRESHOLD
                })

        # --- parent scalar row (N-independent quantities) ------------------
        parent_row = {
            'TARGETID': np.int64(row['TARGETID']),
            'RA':       np.float32(RA),
            'DEC':      np.float32(DEC),
            'Z':        np.float32(Z),
            'LOGM':     np.float32(LOGM),
            'LOGSFR':   np.float32(LOGSFR),
            'N_COMP':     best_n,
            'N_COMP_FIT': n_fit,   # components in the chosen fit (== N_COMP; insignificant ones are flagged, not dropped)
        }
        for i in range(N_MODELS):
            parent_row[f'BIC_{i+1}'] = np.float32(bic_by_n[i])
            parent_row[f'CHISQ_{i+1}'] = np.float32(chisq_by_n[i])
        for name in LINE_NAMES:
            parent_row[f'{name}_FLUX_TOT'] = np.float32(line_flux_total[name])

        return parent_row, comp_rows, full_model.astype(np.float32)

    # ================================================================== #
    #  3. Fit the whole sample -> parent table + component table + cube   #
    # ================================================================== #
    def fit_all(self, data_class: Spectrum, n_jobs=5):
        def process(target_id):
            return self.fit_dp(data_class=data_class, id=target_id)

        results = Parallel(n_jobs=n_jobs)(
            delayed(process)(tid) for tid in tqdm(data_class.targetID)
        )

        parent_rows, comp_rows_nested, models = zip(*results)

        dp_parent = pd.DataFrame(list(parent_rows))
        dp_components = pd.DataFrame(
            [r for rows in comp_rows_nested for r in rows]
        )
        model_cube = np.array(models)  # [n_obj, n_wave], best-fit total model

        return dp_parent, dp_components, model_cube

    # ================================================================== #
    #  4. Downstream science on the tidy table                           #
    # ================================================================== #
    def bpt_classification(self, dp_components: pd.DataFrame):
        """
        Classify EACH component of EACH target on the BPT diagram.
        Returns a (TARGETID, COMP_IDX) -> class table, so a galaxy with
        3 components gets 3 classifications instead of a fixed L/R pair.

        A ratio's numerator or denominator line can be individually
        undetected (SNR < SNR_SIG_THRESHOLD) while still being included in
        the classification, so OIII_HB_BOUND / NII_HA_BOUND flag whether
        each BPT axis is a genuine measurement or only a limit:
          - numerator undetected, denominator detected  -> 'upper_limit'
          - denominator undetected, numerator detected   -> 'lower_limit'
          - both detected                                -> 'measured'
          - both undetected                              -> 'uncertain'
        """
        need = ['OIII5007', 'Hbeta', 'NII6583', 'Halpha']

        def bound(snr, num, den):
            num_ok = snr[num] >= SNR_SIG_THRESHOLD
            den_ok = snr[den] >= SNR_SIG_THRESHOLD
            if num_ok and den_ok:
                return 'measured'
            if den_ok:
                return 'upper_limit'
            if num_ok:
                return 'lower_limit'
            return 'uncertain'

        def classify(sub):
            f = sub.set_index('LINE')['FLUX']
            snr = sub.set_index('LINE')['SNR']
            if any(l not in f.index for l in need):
                return pd.Series({'BPT': 256, 'OIII_HB_BOUND': None, 'NII_HA_BOUND': None})

            oiii_hb_bound = bound(snr, 'OIII5007', 'Hbeta')
            nii_ha_bound = bound(snr, 'NII6583', 'Halpha')

            if (snr[need] <= 0).sum() > 1:
                return pd.Series({'BPT': 256, 'OIII_HB_BOUND': oiii_hb_bound,
                                   'NII_HA_BOUND': nii_ha_bound})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                oiii_hb = np.log10(f['OIII5007'] / f['Hbeta'])
                nii_ha = np.log10(f['NII6583'] / f['Halpha'])
            sf = 0.61 / (nii_ha - 0.05) + 1.30
            comp = 0.61 / (nii_ha - 0.47) + 1.19
            liner = 1.05 * nii_ha + 0.45
            if (sf > oiii_hb or comp > oiii_hb) and (nii_ha < 0.47):
                bpt = 1 if (sf > oiii_hb and nii_ha < 0.05) else 4
            elif (sf < oiii_hb or comp < oiii_hb):
                bpt = 64 if liner > oiii_hb else 16
            else:
                bpt = 256
            return pd.Series({'BPT': bpt, 'OIII_HB_BOUND': oiii_hb_bound,
                               'NII_HA_BOUND': nii_ha_bound})

        out = (dp_components
               .groupby(['TARGETID', 'COMP_IDX'])
               .apply(classify)
               .reset_index())
        return out

    # ================================================================== #
    #  4b. Stage-4 kinematic classification (line-profile shape)         #
    # ================================================================== #
    #
    # Design (matches the workflow):
    #   * SYSTEMIC-INDEPENDENT. Uses only relative velocities (via an internal
    #     flux-weighted reference V_REF), component widths, and an ABSOLUTE
    #     broad-width cut -- never the (possibly biased) pipeline redshift Z,
    #     and never a width ratio for the broad test.
    #   * 'SD' is gated on the width ratio only (comparable widths, straddling
    #     the reference); flux ratio is NOT required.
    #   * Widths are the deconvolved velocity sigmas, floored at the
    #     resolution so unresolved spikes can't drive the ratios.
    # Interpretation that needs a true systemic (blueshift, centring) is a
    # separate, later step -- not done here.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _distinct_widths(sig_sorted, ratio=DISTINCT_WIDTH_RATIO):
        """True if the (>=3) components have mutually distinct widths, i.e.
        every consecutive sorted width ratio exceeds `ratio` (stratified)."""
        s = np.sort(np.asarray(sig_sorted, float))
        if len(s) < 3 or np.any(s <= 0):
            return False
        return bool(np.all((s[1:] / s[:-1]) > ratio))

    @staticmethod
    def _is_two_systems(dv, sig_f):
        """Split components at their largest velocity gap; True if that yields
        two groups of >=2 separated by > TWO_SYS_MIN_SEP, each with an internal
        width contrast (a narrow + a broader component) -> two outflow systems.
        """
        dv = np.asarray(dv, float)
        sig_f = np.asarray(sig_f, float)
        if len(dv) < 4:
            return False
        order = np.argsort(dv)
        dv, sig_f = dv[order], sig_f[order]
        gaps = np.diff(dv)
        k = int(np.argmax(gaps))                     # split after index k
        if gaps[k] < TWO_SYS_MIN_SEP:
            return False
        g1, g2 = sig_f[:k + 1], sig_f[k + 1:]
        if len(g1) < 2 or len(g2) < 2:
            return False

        def has_contrast(s):
            lo, hi = np.min(s), np.max(s)
            return lo > 0 and (hi / lo) > TWO_SYS_CONTRAST

        return bool(has_contrast(g1) and has_contrast(g2))

    @staticmethod
    def _width_census(sig):
        """Count components by width class -- narrow / medium / broad. The
        three bins partition the components, so CNT_N + CNT_M + CNT_B == N_KIN.
        Purely descriptive: unlike N_BROAD this ignores the relative-broad rule."""
        s = np.asarray(sig, float)
        return {
            'CNT_N': int(np.sum(s < WIDTH_NARROW_MAX)),
            'CNT_M': int(np.sum((s >= WIDTH_NARROW_MAX) & (s < WIDTH_MEDIUM_MAX))),
            'CNT_B': int(np.sum(s >= WIDTH_MEDIUM_MAX)),
        }

    @staticmethod
    def _rotation_subclass(amp_ratio):
        """Split a symmetric double by the peak-amplitude ratio of its two
        components:
            1/ROT_AMP_SYM  <= r <= ROT_AMP_SYM   -> 'rotation'
            1/ROT_AMP_ASYM <= r <  1/ROT_AMP_SYM
              or ROT_AMP_SYM < r <= ROT_AMP_ASYM -> 'asymmetry'
            otherwise                            -> 'ambiguous'
        The ratio is folded to >= 1 first, so the bands are symmetric about 1
        and the answer does not depend on which component is the numerator.
        Falls back to 'rotation' when no amplitude information is available.
        """
        if not np.isfinite(amp_ratio) or amp_ratio <= 0:
            return 'rotation'
        r = amp_ratio if amp_ratio >= 1.0 else 1.0 / amp_ratio
        if r <= ROT_AMP_SYM:
            return 'rotation'
        if r <= ROT_AMP_ASYM:
            return 'asymmetry'
        return 'ambiguous'

    def _kinematic_class_one(self, dv, sigma, flux, amp=None):
        """Classify one target's components into a line-profile-shape class.
        Returns (class_str, flags_dict). `dv`, `sigma`, `flux`, `amp` are 1D
        arrays (any order); `sigma` are the deconvolved velocity sigmas [km/s].
        `amp` (peak amplitude on a common anchor line) is only used for the
        2-component, non-SD split; if omitted those fall back to 'outflow'."""
        dv = np.asarray(dv, float)
        sigma = np.asarray(sigma, float)
        flux = np.asarray(flux, float)
        amp = None if amp is None else np.asarray(amp, float)
        n = len(dv)
        if n == 0:
            return 'ambiguous', {'N_KIN': 0, 'HAS_SD': False, 'HAS_CEN': False,
                                 'HAS_BROAD': False, 'N_BROAD': 0,
                                 'TWO_SYS': False, 'V_REF': np.nan,
                                 'AMP_RATIO': np.nan,
                                 **self._width_census([])}
        if n == 1:
            return 'single', {'N_KIN': 1, 'V_REF': float(dv[0]),
                              'HAS_SD': False, 'HAS_CEN': False,
                              'HAS_BROAD': bool(sigma[0] >= BROAD_SIGMA_THRESHOLD),
                              'N_BROAD': int(sigma[0] >= BROAD_SIGMA_THRESHOLD),
                              'TWO_SYS': False, 'AMP_RATIO': np.nan,
                              **self._width_census(sigma)}

        order = np.argsort(dv)
        dv, sigma, flux = dv[order], sigma[order], flux[order]
        if amp is not None:
            amp = amp[order]
        # floor widths at the resolution so unresolved spikes can't drive ratios
        sig_f = np.maximum(sigma, CLASS_SIGMA_FLOOR)

        is_broad = sig_f >= BROAD_SIGMA_THRESHOLD
        # relative broad (n>=3): a single component whose width stands out from
        # the rest -- widest / second-widest above SD_WIDTH_RATIO_MAX -- counts
        # as a broad (outflow-like) component even if it is below the absolute
        # BROAD_SIGMA_THRESHOLD. Comparing to the SECOND-widest (not to the rims)
        # keeps a moderately-wide central-ISM head from being mislabelled.
        if n >= 3:
            wsort = np.sort(sig_f)
            if wsort[-2] > 0 and (wsort[-1] / wsort[-2]) > SD_WIDTH_RATIO_MAX:
                is_broad[int(np.argmax(sig_f))] = True
        has_broad = bool(is_broad.any())

        # internal, systemic-free reference: flux-weighted mean of non-broad comps
        ref_mask = ~is_broad if (~is_broad).any() else np.ones(n, bool)
        w = np.clip(flux[ref_mask], 0, None)
        if w.sum() > 0:
            v_ref = float(np.average(dv[ref_mask], weights=w))
        else:
            v_ref = float(np.mean(dv[ref_mask]))

        # --- SD: straddling pair of comparable-width, non-broad components ---
        cand = [i for i in range(n) if not is_broad[i]]
        sd_pair, best_asym = None, np.inf
        for a in range(len(cand)):
            for b in range(a + 1, len(cand)):
                i, j = cand[a], cand[b]
                if not (dv[i] < v_ref < dv[j]):               # must straddle
                    continue
                # comparable widths -- either the ratio is within
                # SD_WIDTH_RATIO_MAX, or BOTH components sit in the narrow
                # regime (sigma < WIDTH_NARROW_MAX). At small widths the ratio
                # is very sensitive (and close to the resolution floor), so two
                # genuinely narrow rims still count as a double even if their
                # ratio formally exceeds the limit.
                lo, hi = sorted((sig_f[i], sig_f[j]))
                both_narrow = (sig_f[i] < WIDTH_NARROW_MAX and
                               sig_f[j] < WIDTH_NARROW_MAX)
                if lo <= 0 or ((hi / lo) > SD_WIDTH_RATIO_MAX and not both_narrow):
                    continue
                # separation test. For a 2-component fit use the PEAK-TO-PEAK
                # separation: v_ref is the flux-weighted mean of just these two,
                # so an unequal flux ratio pulls it onto the brighter component
                # and its |dv - v_ref| would fail spuriously. For 3-/4-component
                # fits keep the per-component offset from the reference.
                if n == 2:
                    if abs(dv[j] - dv[i]) < SD_MIN_SEP:
                        continue
                elif abs(dv[i] - v_ref) < SD_MIN_SEP or abs(dv[j] - v_ref) < SD_MIN_SEP:
                    continue
                asym = abs(abs(dv[i] - v_ref) - abs(dv[j] - v_ref))
                if asym < best_asym:                          # keep most symmetric pair
                    best_asym, sd_pair = asym, (i, j)
        has_sd = sd_pair is not None

        # --- CEN: a non-broad component sitting BETWEEN the two SD (rim)
        #     components -- the central head. Only defined when an SD pair
        #     exists; no proximity-to-reference tolerance is required (per Fred). ---
        cen_idx = None
        if sd_pair is not None:
            lo_dv, hi_dv = sorted((dv[sd_pair[0]], dv[sd_pair[1]]))
            for i in range(n):
                if is_broad[i] or i in sd_pair:
                    continue
                if lo_dv < dv[i] < hi_dv:
                    cen_idx = i
                    break
        has_cen = cen_idx is not None

        two_sys = self._is_two_systems(dv, sig_f)
        n_broad = int(is_broad.sum())

        # peak-amplitude ratio.
        #   * when an SD pair exists -> ratio of the PAIR (blue/red), used to
        #     split 'rotation' into rotation / asymmetry / ambiguous. Those
        #     bands are symmetric about 1, so the numerator choice is irrelevant.
        #   * otherwise, for a 2-comp fit -> narrow/wide, used for the
        #     2-width / outflow split (those bands are NOT symmetric, so the
        #     narrow-over-wide order matters).
        amp_ratio = np.nan
        if amp is not None:
            if sd_pair is not None:
                i, j = sd_pair
                if np.isfinite(amp[i]) and np.isfinite(amp[j]) and amp[j] > 0:
                    amp_ratio = float(amp[i] / amp[j])
            elif n == 2:
                i_n, i_w = int(np.argmin(sig_f)), int(np.argmax(sig_f))
                if np.isfinite(amp[i_n]) and np.isfinite(amp[i_w]) and amp[i_w] > 0:
                    amp_ratio = float(amp[i_n] / amp[i_w])

        # --- priority decision tree --------------------------------------
        if two_sys:
            kin = '2-pairs'
        elif n in (2, 3) and n_broad >= 2:
            # two broad components in a 2- or 3-component fit is degenerate /
            # unreliable -- don't try to interpret the shape.
            kin = 'ambiguous'
        elif has_sd and has_cen and has_broad:
            kin = 'rotation+central+outflow'
        elif has_sd and has_broad:
            kin = 'rotation+outflow'
        elif has_sd and has_cen:
            kin = 'rotation+central'
        elif has_sd:
            kin = self._rotation_subclass(amp_ratio)
        elif n == 2:
            # not a symmetric double -> split on the narrow/wide amplitude
            # ratio (see AMP_RATIO_* above). Without amplitudes, fall back to
            # the previous behaviour.
            if amp is None:
                kin = 'outflow'
            elif not np.isfinite(amp_ratio) or amp_ratio < AMP_RATIO_AMBIG:
                kin = 'ambiguous'
            elif amp_ratio <= AMP_RATIO_OUTFLOW:
                kin = '2-width'
            else:
                kin = 'outflow'
        elif n >= 3 and self._distinct_widths(sig_f):
            kin = f'{n}-width'          # '3-width' (3 comp) or '4-width' (4 comp)
        elif has_broad:
            kin = 'outflow'
        else:
            kin = 'ambiguous'

        flags = {
            'N_KIN':     int(n),
            'HAS_SD':    bool(has_sd),
            'HAS_CEN':   bool(has_cen),
            'HAS_BROAD': bool(has_broad),
            'N_BROAD':   int(n_broad),
            'TWO_SYS':   bool(two_sys),
            'V_REF':     float(v_ref),
            'AMP_RATIO': float(amp_ratio),   # SD-pair ratio, else narrow/wide (2-comp)
            **self._width_census(sigma),     # CNT_N / CNT_M / CNT_B width census
        }
        return kin, flags

    @staticmethod
    def _pick_amp_line(sub, kept_index, sig_f):
        """
        Choose the line on which the narrow/wide peak-amplitude ratio is
        measured, and return (line_name, amp_array_aligned_to_kept_index).

        For a 2-component system: among QUALIFY_ANCHOR_LINES, keep only the
        lines in which BOTH components are individually detected (the SIG flag,
        i.e. amp > SNR_SIG_THRESHOLD * background), then pick the line with the
        LARGEST peak amplitude of the WIDE component. The wide component is the
        harder of the two to constrain, so measuring the ratio where it is
        strongest gives the most reliable value.

        Falls back to the brightest line when n != 2 or when no anchor line has
        both components detected.
        """
        def amps_for(line):
            s = sub[sub['LINE'] == line]
            if s.empty:
                return None, None
            s = s.set_index('COMP_IDX')
            a = s['AMP'].reindex(kept_index).to_numpy(float)
            d = (s['SIG'].reindex(kept_index).fillna(False).to_numpy(bool)
                 if 'SIG' in s.columns else np.ones(len(kept_index), bool))
            return a, d

        if len(kept_index) == 2:
            i_n, i_w = int(np.argmin(sig_f)), int(np.argmax(sig_f))
            best = None
            for line in QUALIFY_ANCHOR_LINES:
                a, d = amps_for(line)
                if a is None or not np.all(np.isfinite(a)):
                    continue
                if not (bool(d[i_n]) and bool(d[i_w])):   # both must be >3 S/N
                    continue
                if a[i_w] <= 0:
                    continue
                if best is None or a[i_w] > best[0]:      # rank by WIDE-comp amplitude
                    best = (a[i_w], line, a)
            if best is not None:
                return best[1], best[2]

        tot_by_line = sub.groupby('LINE')['FLUX'].sum()   # fallback: brightest
        if not len(tot_by_line):
            return None, None
        line = tot_by_line.idxmax()
        a, _ = amps_for(line)
        return line, a

    def classify_kinematics(self, dp_components: pd.DataFrame, drop_insig: bool = True):
        """
        Stage-4 line-profile-shape classification, one row per TARGETID.

        Parameters
        ----------
        dp_components : tidy component table (from fit_dp / fit_all).
        drop_insig    : if True, ignore components flagged COMP_INSIG (no line
                        detected above SNR_SIG_THRESHOLD) when classifying.

        Returns
        -------
        DataFrame: TARGETID, KIN_CLASS, N_KIN, HAS_SD, HAS_CEN, HAS_BROAD,
                   N_BROAD, TWO_SYS, V_REF, AMP_RATIO, AMP_LINE,
                   CNT_N, CNT_M, CNT_B.
        CNT_N / CNT_M / CNT_B are a width census of the components -- narrow
        (sigma < WIDTH_NARROW_MAX), medium, broad (sigma >= WIDTH_MEDIUM_MAX) --
        and always sum to N_KIN.
        KIN_CLASS in {'rotation', 'asymmetry', 'rotation+central',
        'rotation+outflow', 'rotation+central+outflow', 'outflow', '2-width',
        '3-width', '4-width', '2-pairs', 'single', 'ambiguous'}.
        ('asymmetry' is a symmetric double whose two peaks differ strongly in
        amplitude -- see _rotation_subclass.)

        For 2-component systems the amplitude ratio is measured on the anchor
        line chosen by _pick_amp_line: among QUALIFY_ANCHOR_LINES, the lines
        where BOTH components are detected (>3 S/N) are ranked by the peak
        amplitude of the WIDE component and the LARGEST is used. Otherwise the
        brightest line is used. The chosen line is reported as AMP_LINE.
        """
        def per_target(sub):
            g = sub.groupby('COMP_IDX')
            comp_index = g['DV'].first().index
            dv = g['DV'].first().to_numpy(float)
            sigma = g['SIGMA'].first().to_numpy(float)
            flux = g['FLUX'].sum().to_numpy(float)       # brightness weight

            keep = np.ones(len(dv), bool)
            if drop_insig and 'COMP_INSIG' in sub.columns:
                k = ~g['COMP_INSIG'].first().to_numpy(bool)
                if k.any():                               # never drop everything
                    keep = k
            dv, sigma, flux = dv[keep], sigma[keep], flux[keep]
            kept_index = comp_index[keep]

            # anchor line for the narrow/wide amplitude ratio (2-comp rule)
            sig_f = np.maximum(sigma, CLASS_SIGMA_FLOOR)
            anchor, amp = self._pick_amp_line(sub, kept_index, sig_f)

            kin, flags = self._kinematic_class_one(dv, sigma, flux, amp)
            return pd.Series({'KIN_CLASS': kin, **flags, 'AMP_LINE': anchor})

        out = (dp_components
               .groupby('TARGETID')
               .apply(per_target)
               .reset_index())
        return out

    def estimate_SFR(self, dp_components: pd.DataFrame, dp_parent: pd.DataFrame):
        """
        Per-component SFR from Halpha/Hbeta, plus a total SFR per galaxy
        (summed over components) written back onto the parent table.
        """
        z_map = dp_parent.set_index('TARGETID')['Z']

        def comp_sfr(sub):
            f = sub.set_index('LINE')['FLUX']
            noise = sub.set_index('LINE')['NOISE3']
            if 'Halpha' not in f.index or 'Hbeta' not in f.index:
                return 0.0
            f_ha, f_hb = f['Halpha'], max(f['Hbeta'], noise['Hbeta'] / 3)
            if f_ha <= 0 or f_hb <= 0:
                return 0.0
            z = float(z_map.loc[sub['TARGETID'].iloc[0]])
            return sfr(f_ha, f_hb, z)   # misc.sfr

        per_comp = (dp_components
                    .groupby(['TARGETID', 'COMP_IDX'])
                    .apply(comp_sfr)
                    .rename('SFR').reset_index())

        tot = per_comp.groupby('TARGETID')['SFR'].sum()
        dp_parent = dp_parent.copy()
        dp_parent['LOGSFR_FIT'] = np.log10(
            dp_parent['TARGETID'].map(tot).where(lambda s: s > 0)
        ).fillna(-15).astype(np.float32)

        per_comp['LOGSFR'] = np.where(per_comp['SFR'] > 0,
                                      np.log10(per_comp['SFR'].where(lambda s: s > 0)),
                                      -15)
        return per_comp, dp_parent

    # ================================================================== #
    #  5. Sample selection                                               #
    # ================================================================== #
    def select_dp_sample(self, dp_parent: pd.DataFrame, dp_components: pd.DataFrame):
        """
        A galaxy is a multi-component (DP+) candidate when the best model
        has >1 component AND at least two components are significantly
        detected (SIG) in the same line. Fully generalizes the old
        'brightest line is double-peaked' rule to arbitrary N.
        """
        multi = dp_parent[dp_parent['N_COMP'] > 1].copy()

        sig_counts = (dp_components[dp_components['SIG']]
                      .groupby(['TARGETID', 'LINE'])['COMP_IDX']
                      .nunique())
        # targets with >=2 significant components in at least one line
        good_ids = sig_counts[sig_counts >= 2].index.get_level_values(0).unique()

        dp_sample = multi[multi['TARGETID'].isin(good_ids)].copy()
        return dp_sample

    # ================================================================== #
    #  6. FITS persistence (variable N)                                  #
    # ================================================================== #
    @staticmethod
    def _to_fits_records(df):
        """
        Convert a DataFrame to a numpy record array FITS can write.
        Object columns (e.g. the string 'LINE' column) are given an explicit
        fixed-width unicode dtype, otherwise pandas leaves them as `object`
        and BinTableHDU raises 'unsupported object types or mixed types'.
        """
        col_dtypes = {}
        for col in df.columns:
            if df[col].dtype == object:
                maxlen = int(df[col].astype(str).str.len().max()) or 1
                col_dtypes[col] = f'U{maxlen}'
        return df.to_records(index=False, column_dtypes=col_dtypes)

    def get_catalog(self, dp_parent, dp_components, model_cube, fname):
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul.append(fits.BinTableHDU(self._to_fits_records(dp_parent), name='PARENT'))
        hdul.append(fits.BinTableHDU(self._to_fits_records(dp_components), name='COMPONENTS'))
        hdul.append(fits.ImageHDU(model_cube.astype(np.float32), name='MODEL'))
        hdul.writeto(fname, overwrite=True)

    @staticmethod
    def _decode_bytes(df):
        """FITS char columns come back as bytes; decode them to str in place."""
        for col in df.columns:
            if df[col].dtype == object and len(df) and isinstance(df[col].iloc[0], bytes):
                df[col] = df[col].str.decode('utf-8')
        return df

    def extract_fits_data(self, fname: str):
        with fits.open(fname) as hdul:
            dp_parent = self._decode_bytes(Table(hdul['PARENT'].data).to_pandas())
            dp_components = self._decode_bytes(Table(hdul['COMPONENTS'].data).to_pandas())
            model_cube = hdul['MODEL'].data
        return dp_parent, dp_components, model_cube

    # ================================================================== #
    #  7. Diagnostic plot: overlay the N components                      #
    # ================================================================== #
    @staticmethod
    def _rebuild_component(sub_comp, z):
        """
        Rebuild one component's (amp, lam0, dv, sigma_combine) tuples from a
        tidy sub-table, re-adding the instrumental resolution that was
        removed when we stored the *velocity* sigma. Mirrors FITSPECTRUM.
        """
        gp = []
        for _, r in sub_comp.iterrows():
            lam0 = float(r['LAM0'])
            sigma_res = c * 0.8 / (lam0 * (1 + z))
            sigma_comb = np.sqrt(float(r['SIGMA']) ** 2 + sigma_res ** 2)
            gp.append((float(r['AMP']), lam0, float(r['DV']), sigma_comb))
        return gp

    def plot_fit(self, data_class: Spectrum, dp_parent: pd.DataFrame,
                 dp_components: pd.DataFrame, id, save=None):
        """
        Panelled rest-frame plot for a single target: observed emission
        (flux - continuum) with the total best-fit model and every
        individual component overlaid (coloured blue->red by velocity).
        """
        import matplotlib.pyplot as plt

        idx = data_class.id2index(id)
        z = float(data_class.df.iloc[idx]['Z'])
        lam = desi_wavelength / (1 + z)
        # emission = data_class.flux[idx] - data_class.continuum[idx]
        emission = data_class.flux[idx]
        good = (data_class.mask[idx] == 0)

        comp_tbl = dp_components[dp_components['TARGETID'] == np.int64(id)]
        comp_ids = sorted(comp_tbl['COMP_IDX'].unique())
        n_comp = len(comp_ids)
        cmap = plt.get_cmap('coolwarm')
        colors = [cmap(i / max(n_comp - 1, 1)) for i in range(n_comp)]

        # zoom windows (rest frame) per line group
        panels = [
            ('[OII]',        OII_rest.min() - 25,  OII_rest.max() + 25),
            (r'H$\beta$',    Hbeta_rest[0] - 25,   Hbeta_rest[0] + 25),
            ('[OIII]',       OIII_rest.min() - 25, OIII_rest.max() + 25),
            (r'H$\alpha$+[NII]', NII_rest.min() - 25, NII_rest.max() + 25),
            ('[SII]',        SII_rest.min() - 25,  SII_rest.max() + 25),
        ]

        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
        total = np.zeros_like(lam)
        comp_curves = []
        for ci in comp_ids:
            gp = self._rebuild_component(comp_tbl[comp_tbl['COMP_IDX'] == ci], z)
            curve = model_vel(lam, gaussian_parms=gp)
            comp_curves.append(curve)
            total = total + curve

        for ax, (label, lo, hi) in zip(axes, panels):
            m = good & (lam >= lo) & (lam <= hi)
            ax.step(lam[m], emission[m], where='mid', color='0.4', lw=0.8, label='data')
            ax.plot(lam[m], total[m], color='k', lw=1.5, label='total')
            for ci, curve, col in zip(comp_ids, comp_curves, colors):
                ax.plot(lam[m], curve[m], color=col, lw=1.0, ls='--',
                        label=f'comp {ci}')
            ax.set_xlim(lo, hi)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel(r'rest $\lambda$ [$\AA$]')
        axes[0].set_ylabel('flux - continuum')
        axes[-1].legend(fontsize=7, frameon=False)
        n_best = int(dp_parent.loc[dp_parent['TARGETID'] == np.int64(id),
                                   'N_COMP'].iloc[0])
        fig.suptitle(f'TARGETID {id}   z={z:.4f}   N_COMP={n_best}', fontsize=11)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=130, bbox_inches='tight')
        return fig, axes
