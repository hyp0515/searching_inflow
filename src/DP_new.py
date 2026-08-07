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
# Amplitude-contrast tests applied on the anchor line (see _model_qualifies).
# S/N alone is a poor gate: what matters is whether the faintest component is
# comparable to the others.
#   * a model whose components ALL clear 3 S/N is still REJECTED when the
#     faintest is more than this many times fainter than the brightest -- a
#     formally-significant sliver is not a real kinematic component.
QUALIFY_AMP_RATIO_REJECT = 10.0
#   * a model with exactly ONE sub-threshold component is still ACCEPTED when
#     the faintest is within this factor of the brightest, i.e. it is
#     comparable to the rest and only marginally under the S/N cut.
QUALIFY_AMP_RATIO_RESCUE = 5.0
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

# --- kinematic (line-profile shape) classification -------------------------
# Bottom-up morphological scheme; see kinematic_classification_spec.md. All
# widths are the DECONVOLVED velocity sigmas from dp_components['SIGMA'].

# Widths are floored at the DESI resolution -- a component narrower than we can
# resolve is treated as "= resolution", so unresolved spikes can't drive ratios.
WIDTH_FLOOR = 30.0            # km/s, ~DESI resolution at these redshifts

# Two components are RESOLVED into distinct peaks when their velocity separation
# exceeds SEP_RATIO_MIN * (sigma_i + sigma_j) -- the Gaussian bimodality
# condition (Delta > 2 sigma for equal widths). With the floor this is ~60 km/s
# at the narrow end and scales up for wide pairs.
SEP_RATIO_MIN = 1.0

# A component is BROAD on this absolute width cut, or -- as the single widest,
# with the WIDTH_NARROW_MAX floor -- when it exceeds the second-widest by more
# than WIDTH_SIMILAR_RATIO. A broad component that is also faint is a "wing".
BROAD_WIDTH = 200.0          # km/s, absolute broad cut

# Two widths are SIMILAR (same tier, for doublet/triplet/quadruplet grouping)
# when their ratio <= WIDTH_SIMILAR_RATIO, or both are narrow (< WIDTH_NARROW_MAX,
# where the ratio is unreliable near the floor). The same ratio applied strictly
# (no narrow exception) defines DISTINCT width layers for the n-width classes.
WIDTH_SIMILAR_RATIO = 2.0

# A broad component is a faint WING (rather than a comparable broad LAYER, or a
# system-dominating STRONG broad) when the brightest core out-peaks it by at
# least this amplitude factor.
WING_WEAK_RATIO = 1.5

# SYM/ASYM amplitude-balance flag for the outer peak pair: folded amplitude
# ratio <= AMP_SYM_MAX -> SYM, <= AMP_ASYM_MAX -> ASYM, beyond -> ambiguous.
# AMP_ASYM_MAX also gates "amplitudes comparable" for triplet / NBN / BNB.
AMP_SYM_MAX = 2.0
AMP_ASYM_MAX = 4.0

# Descriptive width census reported alongside the class: narrow / medium / broad
# by width alone, so CNT_N + CNT_M + CNT_B == N_KIN (independent of the
# BROAD_WIDTH classification flag, which also carries the relative-broad rule).
WIDTH_NARROW_MAX = 90.0      # km/s, N: sigma < 90
WIDTH_MEDIUM_MAX = 200.0     # km/s, M: 90 <= sigma < 200 ; B: sigma >= 200


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

    def _component_amps(self, params):
        """Per-component amplitude on every line (flat, aligned with LINE_NAMES),
        bluest first -- same order as _component_sig_flags."""
        return [[comp['gp_flat'][li][0] for li in range(len(LINE_NAMES))]
                for comp in self._iter_components(params)]

    @staticmethod
    def _model_qualifies(flags, amps=None):
        """
        `flags`: list of per-component boolean lists (len(LINE_NAMES)), from
        _component_sig_flags.  `amps`: matching per-component amplitude lists
        from _component_amps.

        A candidate model qualifies if it passes the test below on at least
        ONE anchor line. All components are always judged on the SAME anchor
        line -- never a mix of different lines per component.

        On a given anchor line, with N components, n_sig of them above
        SNR_SIG_THRESHOLD, and contrast R = max(amp) / min(amp):
          n_sig == N      -> qualifies UNLESS R > QUALIFY_AMP_RATIO_REJECT.
                             A component that formally clears 3 S/N but is
                             >10x fainter than the brightest is a sliver, not
                             a distinct kinematic component.
          n_sig == N - 1  -> still qualifies IF R < QUALIFY_AMP_RATIO_RESCUE.
                             The single sub-threshold component is comparable
                             in amplitude to the rest, so it is likely real
                             and only marginally under the cut.
          n_sig < N - 1   -> does not qualify.

        Because every component of a line shares the same background sigma,
        S/N order == amplitude order on that line, so in the n_sig == N-1 case
        the faintest component is exactly the one that missed the cut.

        With `amps` omitted this degrades to the original "all components
        significant" rule.
        """
        if not flags:
            return False
        n_comp = len(flags)
        anchor_idx = [LINE_NAMES.index(name) for name in QUALIFY_ANCHOR_LINES]

        for li in anchor_idx:
            n_sig = sum(bool(comp_flags[li]) for comp_flags in flags)
            if n_sig == 0 or n_sig < n_comp - 1:
                continue

            if amps is None:                       # no amplitudes -> legacy rule
                if n_sig == n_comp:
                    return True
                continue

            a = [float(comp_amps[li]) for comp_amps in amps]
            a_min, a_max = min(a), max(a)
            if a_min <= 0 or not np.isfinite(a_min) or not np.isfinite(a_max):
                ratio = np.inf                     # degenerate/non-positive amp
            else:
                ratio = a_max / a_min

            if n_sig == n_comp:
                if ratio <= QUALIFY_AMP_RATIO_REJECT:
                    return True
            elif ratio < QUALIFY_AMP_RATIO_RESCUE:  # n_sig == n_comp - 1
                return True
        return False

    def _choose_n_components(self, bic_by_n, sig_flags_by_n, comp_sigmas_by_n,
                             comp_dvs_by_n, comp_amps_by_n=None):
        """
        Model selection, repeated against a shrinking candidate pool:
          1. n_min = argmin(BIC) among remaining candidates. If every other
             candidate's BIC is more than DELTA_BIC_DECISIVE worse, tentatively
             accept n_min.
          2. Otherwise, among n_min and its "competitive" alternatives
             (deltaBIC < DELTA_BIC_DECISIVE), prefer the one with the most
             components that _model_qualifies (on a single anchor line: all
             components significant, unless one is >10x fainter than the
             brightest; or all-but-one significant with a contrast <5x),
             falling back to n_min if none qualify.
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
                if self._model_qualifies(sig_flags_by_n[n - 1],
                                         None if comp_amps_by_n is None else comp_amps_by_n[n - 1]):
                    chosen = n
                    break

            # 2 vs 4 specifically: BIC too close to justify the extra
            # complexity -> prefer the simpler 2-component model outright.
            if chosen == 4 and 2 in candidates and abs(bic_by_n[3] - bic_by_n[1]) < DELTA_BIC_SIMPLICITY:
                chosen = 2

            sigmas_chosen = comp_sigmas_by_n[chosen - 1]
            dvs_chosen = comp_dvs_by_n[chosen - 1]
            fails_anchor_lines = not self._model_qualifies(
                sig_flags_by_n[chosen - 1],
                None if comp_amps_by_n is None else comp_amps_by_n[chosen - 1])
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
        comp_amps_by_n = [None] * N_MODELS
        for i in range(N_MODELS):
            if params_by_n[i] is None:
                continue
            _, _, csigma_i, slice_idx_i = extras_by_n[i]
            sigmab_region_by_n[i] = self._per_region_noise(csigma_i, slice_idx_i)
            sig_flags_by_n[i] = self._component_sig_flags(params_by_n[i], sigmab_region_by_n[i])
            comp_sigmas_by_n[i] = self._component_sigmas(params_by_n[i])
            comp_dvs_by_n[i] = self._component_dvs(params_by_n[i])
            comp_amps_by_n[i] = self._component_amps(params_by_n[i])

        best_n = self._choose_n_components(bic_by_n, sig_flags_by_n, comp_sigmas_by_n,
                                           comp_dvs_by_n, comp_amps_by_n)
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
    #  4b. Kinematic classification -- bottom-up morphological scheme     #
    # ================================================================== #
    #
    # Each component is an atom (width, amplitude, velocity); the system class
    # is composed from the atoms. SYSTEMIC-INDEPENDENT: uses only relative
    # velocities (an internal flux-weighted V_REF), widths and amplitudes --
    # never the pipeline redshift Z. Physical interpretation (rotation vs
    # outflow) is a separate, later step. See kinematic_classification_spec.md.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _width_census(sig):
        """Descriptive width census -- narrow / medium / broad by width alone.
        The three bins partition the components, so CNT_N + CNT_M + CNT_B ==
        N_KIN. Independent of the BROAD_WIDTH classification flag (which also
        carries the relative-broad rule)."""
        s = np.asarray(sig, float)
        return {
            'CNT_N': int(np.sum(s < WIDTH_NARROW_MAX)),
            'CNT_M': int(np.sum((s >= WIDTH_NARROW_MAX) & (s < WIDTH_MEDIUM_MAX))),
            'CNT_B': int(np.sum(s >= WIDTH_MEDIUM_MAX)),
        }

    @staticmethod
    def _similar(wi, wj):
        """Widths comparable for GROUPING: ratio within WIDTH_SIMILAR_RATIO, or
        both narrow (ratio unreliable near the resolution floor)."""
        lo, hi = (wi, wj) if wi <= wj else (wj, wi)
        if lo <= 0:
            return False
        return (hi / lo) <= WIDTH_SIMILAR_RATIO or (wi < WIDTH_NARROW_MAX and wj < WIDTH_NARROW_MAX)

    @staticmethod
    def _distinct_ratio(wi, wj):
        """Widths distinct for n-WIDTH layering: pure ratio > WIDTH_SIMILAR_RATIO
        (no narrow exception -- layers are defined by ratio only)."""
        lo, hi = (wi, wj) if wi <= wj else (wj, wi)
        return lo > 0 and (hi / lo) > WIDTH_SIMILAR_RATIO

    @staticmethod
    def _resolved(vi, vj, wi, wj):
        """Two Gaussians form distinct peaks iff |dv| > SEP_RATIO_MIN*(wi+wj)."""
        return abs(vi - vj) > SEP_RATIO_MIN * (wi + wj)

    def _kinematic_class_one(self, dv, sigma, flux, amp=None):
        """Bottom-up morphological classification of one target's components.
        See kinematic_classification_spec.md. Returns (class, flags)."""
        dv = np.asarray(dv, float)
        sigma = np.asarray(sigma, float)
        flux = np.asarray(flux, float)
        a = flux if amp is None else np.asarray(amp, float)
        n = len(dv)

        flags = {'N_KIN': int(n), 'N_WING': 0, 'WING_KIND': None, 'WING_DV': np.nan,
                 'WING_IDX': np.nan, 'WING_SIGMA': np.nan, 'SYM': None,
                 'BROAD_DOM': False, 'V_REF': np.nan, 'AMP_RATIO': np.nan,
                 'REASON': '', **self._width_census(sigma)}
        if n == 0:
            flags['REASON'] = 'no components'
            return 'ambiguous', flags
        if n == 1:
            flags['V_REF'] = float(dv[0])
            if sigma[0] >= BROAD_WIDTH:
                flags['REASON'] = f'single broad component (sigma={sigma[0]:.0f} >= {BROAD_WIDTH:g})'
                return 'broad-dominated', flags
            flags['REASON'] = 'single component below the broad cut'
            return 'single', flags

        # order by velocity, floor widths
        o = np.argsort(dv)
        dv, sigma, flux, a = dv[o], sigma[o], flux[o], a[o]
        w = np.maximum(sigma, WIDTH_FLOOR)

        # --- broadness ---
        # absolute-broad components are never "cores". A relative-broad candidate
        # (single widest, floored) becomes a WING only if it is also faint; a
        # comparable-amplitude wide component (e.g. an NBN central head) stays a core.
        broad_abs = w >= BROAD_WIDTH
        ws = np.sort(w)
        wi = int(np.argmax(w))
        rel = wi if (w[wi] >= WIDTH_NARROW_MAX and ws[-2] > 0
                     and (w[wi] / ws[-2]) > WIDTH_SIMILAR_RATIO) else -1
        a_core_max = float(np.max(a[~broad_abs])) if (~broad_abs).any() else float(np.max(a))

        wing = np.zeros(n, bool)
        strong = False
        for i in np.where(broad_abs)[0]:
            if a[i] > a_core_max:
                strong = True
            elif a[i] > 0 and (a_core_max / a[i]) >= WING_WEAK_RATIO:
                wing[i] = True
            # else: comparable absolute-broad LAYER (not core, not wing)
        if rel >= 0 and not broad_abs[rel] and a[rel] > 0 and (a_core_max / a[rel]) >= WING_WEAK_RATIO:
            wing[rel] = True

        core_mask = ~broad_abs & ~wing
        cw = np.clip(flux[core_mask], 0, None)
        if core_mask.any() and cw.sum() > 0:
            flags['V_REF'] = float(np.average(dv[core_mask], weights=cw))
        else:
            flags['V_REF'] = float(np.mean(dv))

        cores = list(np.where(core_mask)[0])
        wings = list(np.where(wing)[0])
        m, k = len(cores), len(wings)
        flags['N_WING'] = k
        # The wing of record is the WIDEST broad component (per Fred): when a
        # system has more than one wing, its velocity/kind/index are reported for
        # the widest one. WING_DV is in the same frame as DV; WING_IDX is the
        # position in the INPUT array (classify_kinematics maps it to COMP_IDX).
        if k:
            widest_wing = wings[int(np.argmax([w[i] for i in wings]))]
            flags['WING_DV'] = float(dv[widest_wing])
            flags['WING_SIGMA'] = float(sigma[widest_wing])   # deconvolved velocity sigma
            flags['WING_IDX'] = float(o[widest_wing])         # un-sort back to input order
            kinds = set('ABSOLUTE' if w[i] >= BROAD_WIDTH else 'RELATIVE' for i in wings)
            flags['WING_KIND'] = kinds.pop() if len(kinds) == 1 else 'MIXED'

        def done(kin, sym=None, ratio=np.nan, reason=''):
            flags['SYM'] = sym
            flags['AMP_RATIO'] = float(ratio)
            flags['REASON'] = reason
            return kin, flags

        if strong:
            flags['BROAD_DOM'] = True
            return done('broad-dominated',
                        reason='a broad component out-peaks the cores (not a faint wing)')

        # --- amplitude / width helpers ---
        def sym_of(i, j):
            ai, aj = a[i], a[j]
            if not (np.isfinite(ai) and np.isfinite(aj) and min(ai, aj) > 0):
                return None, np.nan
            r = max(ai, aj) / min(ai, aj)
            return ('SYM' if r <= AMP_SYM_MAX else ('ASYM' if r <= AMP_ASYM_MAX else 'TOO')), r

        def amp_comparable(idxs):
            # For >=3 peaks, judge on strongest / SECOND-weakest so a single
            # faint peak (e.g. a weak central component) can't veto the class
            # (per Fred); the amplitude imbalance still surfaces in SYM/ASYM.
            av = sorted(a[i] for i in idxs if np.isfinite(a[i]) and a[i] > 0)
            if len(av) < 2:
                return True
            denom = av[0] if len(av) == 2 else av[1]     # ignore the single faintest at n>=3
            return denom > 0 and (av[-1] / denom) <= AMP_ASYM_MAX

        def all_similar(idxs):
            return all(self._similar(w[i], w[j]) for x, i in enumerate(idxs) for j in idxs[x + 1:])

        def all_distinct(idxs):
            return all(self._distinct_ratio(w[i], w[j]) for x, i in enumerate(idxs) for j in idxs[x + 1:])

        def outer(idxs):
            s = sorted(idxs, key=lambda i: dv[i])
            return s[0], s[-1]

        def base_pattern(idxs):
            """Named base over CORE indices -> (name|None, tag, ratio, reason).
            `reason` explains the match, or the FIRST failed check for None."""
            mm = len(idxs)
            if mm == 1:
                return 'singlet', None, np.nan, 'single core'
            lo, hi = outer(idxs)
            tag, r = sym_of(lo, hi)
            res = self._resolved(dv[lo], dv[hi], w[lo], w[hi])
            sep = abs(dv[hi] - dv[lo]); wsum = w[lo] + w[hi]
            if mm == 2:
                i, j = idxs
                wr = max(w[i], w[j]) / max(min(w[i], w[j]), 1e-9)
                if not self._similar(w[i], w[j]):
                    return None, None, np.nan, f'2 cores not comparable in width (ratio {wr:.1f} > {WIDTH_SIMILAR_RATIO:g})'
                if not self._resolved(dv[i], dv[j], w[i], w[j]):
                    return None, None, np.nan, f'2 cores overlap (|dv|={sep:.0f} <= si+sj={wsum:.0f} km/s)'
                if tag == 'TOO':
                    return None, None, np.nan, f'pair too lopsided (amp ratio {r:.1f} > {AMP_ASYM_MAX:g})'
                return 'doublet', tag, r, f'comparable-width resolved pair ({tag}, amp ratio {r:.1f})'
            if mm == 3:
                s = sorted(idxs, key=lambda i: dv[i]); mid = s[1]
                if not res:
                    return None, None, np.nan, f'3 cores: outer pair overlaps (|dv|={sep:.0f} <= {wsum:.0f} km/s)'
                if not amp_comparable(idxs) or tag == 'TOO':
                    return None, None, np.nan, f'3 cores: peak amplitudes not comparable (> {AMP_ASYM_MAX:g}x)'
                if all_similar(idxs):
                    return 'triplet', tag, r, f'3 comparable-width resolved peaks ({tag})'
                if self._similar(w[lo], w[hi]) and not self._similar(w[mid], w[lo]) \
                   and not self._similar(w[mid], w[hi]):
                    nm2 = 'NBN' if w[mid] > w[lo] else 'BNB'
                    return nm2, tag, r, f'outer pair comparable, middle {"wider" if nm2=="NBN" else "narrower"} ({tag})'
                return None, None, np.nan, '3 cores: widths not uniform / NBN / BNB (odd-one-out on an edge, or chain)'
            if mm == 4:
                if not res:
                    return None, None, np.nan, f'4 cores: outer pair overlaps (|dv|={sep:.0f} <= {wsum:.0f} km/s)'
                if not amp_comparable(idxs) or tag == 'TOO':
                    return None, None, np.nan, f'4 cores: peak amplitudes not comparable (> {AMP_ASYM_MAX:g}x)'
                if all_similar(idxs):
                    return 'quadruplet', tag, r, f'4 comparable-width resolved peaks ({tag})'
                return None, None, np.nan, '4 cores: widths neither all-similar nor all-distinct'
            return None, None, np.nan, 'no base pattern'

        allidx = list(range(n))
        wingdesc = f'wing sigma={sigma[wings[0]]:.0f} ({flags["WING_KIND"]})' if k == 1 else f'{k} wings'

        # =============================== n == 2 ===============================
        if n == 2:
            if k == 1 and m == 1:
                return done('singlet+wing', reason=f'narrow core + faint broad {wingdesc}')
            # strong broad already handled above; the two components now have
            # comparable amplitude. Split on width (component-based, so a
            # comparable ABSOLUTE-broad "layer" is handled the same as a
            # medium one).
            i, j = 0, 1
            if self._similar(w[i], w[j]):
                if not self._resolved(dv[i], dv[j], w[i], w[j]):
                    return done('ambiguous',
                                reason=f'2 comparable-width components overlap '
                                       f'(|dv|={abs(dv[i]-dv[j]):.0f} <= si+sj={w[i]+w[j]:.0f} km/s)')
                tag, r = sym_of(i, j)
                if tag in ('SYM', 'ASYM'):
                    return done('doublet', tag, r, f'comparable-width resolved pair ({tag}, amp ratio {r:.1f})')
                return done('ambiguous', reason=f'pair too lopsided (amp ratio {r:.1f} > {AMP_ASYM_MAX:g})')
            return done('2width', reason='2 distinct-width, comparable-amplitude components')

        # =============================== n == 3 ===============================
        # base patterns (which use the both-narrow-aware `_similar`) are tried
        # BEFORE the pure-ratio n-width gate, so a both-narrow doublet + wing is
        # not stolen by 3width (per Fred).
        if n == 3:
            if k == 1 and m == 2:
                name, tag, r, why = base_pattern(cores)
                if name == 'doublet':
                    return done('doublet+wing', tag, r, f'{why} + faint broad {wingdesc}')
                if all_distinct(allidx):
                    return done('3width', reason='3 mutually distinct width layers')
                return done('ambiguous', reason=f'has a wing but the 2 cores are not a doublet: {why}')
            if k == 0 and m == 3:
                name, tag, r, why = base_pattern(cores)
                if name:
                    return done(name, tag, r, why)
                if all_distinct(allidx):
                    return done('3width', reason='3 mutually distinct width layers')
                return done('ambiguous', reason=why)
            if all_distinct(allidx):
                return done('3width', reason='3 mutually distinct width layers')
            return done('ambiguous', reason=f'{m} core(s) + {k} wing(s): no 3-component pattern')

        # =============================== n == 4 ===============================
        if n == 4:
            if self._is_two_pairs(dv, w, a):
                return done('2pairs', reason='two resolved core+wing systems')
            if k == 1 and m == 3:
                name, tag, r, why = base_pattern(cores)
                if name:
                    return done(f'{name}+wing', tag, r, f'{why} + faint broad {wingdesc}')
                if all_distinct(allidx):
                    return done('3width+wing', reason=f'3 distinct core layers + faint broad {wingdesc}')
                return done('ambiguous', reason=f'has a wing but the 3 cores are not a base: {why}')
            if k == 0 and m == 4:
                name, tag, r, why = base_pattern(cores)
                if name:
                    return done(name, tag, r, why)
                if all_distinct(allidx):
                    return done('4width', reason='4 mutually distinct width layers')
                return done('ambiguous', reason=why)
            if all_distinct(allidx):
                return done('3width+wing' if k == 1 else '4width',
                            reason='mutually distinct width layers')
            return done('ambiguous', reason=f'{m} core(s) + {k} wing(s): no 4-component pattern')

        return done('ambiguous', reason='unhandled component count')

    def _is_two_pairs(self, dv, w, a):
        """n==4 -> two 'singlet+wing' systems. Split at the largest velocity gap;
        require it to divide the four components 2|2, each group to be a local
        core+wing (a narrower core plus a broader, not-brighter partner), and the
        two group CORES to be resolved from each other. Group-local so it catches
        a system whose wing is only medium-width, and resolves on the cores (not
        the broad wings, which can bridge the gap)."""
        order = sorted(range(4), key=lambda i: dv[i])
        gaps = [dv[order[i + 1]] - dv[order[i]] for i in range(3)]
        if int(np.argmax(gaps)) != 1:                 # must split 2 | 2
            return False
        g1, g2 = order[:2], order[2:]

        def core_wing(grp):
            p, q = grp
            c, wg = (p, q) if w[p] <= w[q] else (q, p)      # core=narrower, wing=wider
            contrast = (w[wg] / w[c] > WIDTH_SIMILAR_RATIO) or (w[wg] >= BROAD_WIDTH)
            weaker = a[wg] <= a[c]                            # wing not brighter than its core
            return (c if contrast and weaker else None)

        c1, c2 = core_wing(g1), core_wing(g2)
        if c1 is None or c2 is None:
            return False
        return self._resolved(dv[c1], dv[c2], w[c1], w[c2])  # the two cores are distinct


    @staticmethod
    def _pick_amp_line(sub, kept_index, sig_f):
        """Anchor line on which component amplitudes are read, for ANY component
        count. Among QUALIFY_ANCHOR_LINES keep the lines where ALL components are
        detected (>3 S/N), then take the one with the LARGEST amplitude of the
        WIDEST component -- the broadest component is the hardest to constrain,
        so the ratio is most reliable where it is strongest. If no anchor line
        has every component detected, fall back to the brightest line. Returns
        (line_name, amp_array_aligned_to_kept_index)."""
        def amps_for(line):
            s = sub[sub['LINE'] == line]
            if s.empty:
                return None, None
            s = s.set_index('COMP_IDX')
            a = s['AMP'].reindex(kept_index).to_numpy(float)
            det = (s['SIG'].reindex(kept_index).fillna(False).to_numpy(bool)
                   if 'SIG' in s.columns else np.ones(len(kept_index), bool))
            return a, det

        if len(kept_index) >= 2:
            i_w = int(np.argmax(sig_f))                   # widest component
            best = None
            for line in QUALIFY_ANCHOR_LINES:
                a, det = amps_for(line)
                if a is None or not np.all(np.isfinite(a)):
                    continue
                if not np.all(det) or a[i_w] <= 0:        # every component must be detected
                    continue
                if best is None or a[i_w] > best[0]:      # rank by widest-comp amplitude
                    best = (a[i_w], line, a)
            if best is not None:
                return best[1], best[2]

        tot_by_line = sub.groupby('LINE')['FLUX'].sum()   # fallback: brightest line
        if not len(tot_by_line):
            return None, None
        line = tot_by_line.idxmax()
        a, _ = amps_for(line)
        return line, a

    # BPT code (from bpt_classification) -> per-target count column
    _BPT_COUNT_COL = {1: 'CNT_SF', 4: 'CNT_COMP', 16: 'CNT_AGN', 64: 'CNT_LINER'}

    def classify_kinematics(self, dp_components: pd.DataFrame, drop_insig: bool = True,
                            bpt: pd.DataFrame = None, dp_parent: pd.DataFrame = None):
        """
        Stage-4 line-profile-shape classification, one row per TARGETID.

        Parameters
        ----------
        dp_components : tidy component table (from fit_dp / fit_all).
        drop_insig    : if True, ignore components flagged COMP_INSIG (no line
                        detected above SNR_SIG_THRESHOLD) when classifying.
        bpt           : optional output of bpt_classification (columns TARGETID,
                        COMP_IDX, BPT). When given, four extra count columns
                        CNT_SF / CNT_COMP / CNT_AGN / CNT_LINER are added, tallying
                        the BPT class of each (kept) component per target
                        (codes 1/4/16/64; 256='uncertain' is not counted).
        dp_parent     : optional parent table (needs TARGETID, Z). When given
                        together with `bpt`, a LOGSFR column is added: the total
                        star-formation rate (log10, from Halpha/Hbeta) summed
                        over the components classified SF or COMP only. It is
                        None (NaN) when the system has no SF/COMP component.

        Returns
        -------
        DataFrame: TARGETID, KIN_CLASS, N_KIN, N_WING, WING_KIND, WING_DV,
                   WING_SIGMA, WING_IDX, SYM, BROAD_DOM, V_REF, AMP_RATIO,
                   AMP_LINE, CNT_N, CNT_M, CNT_B, REASON [, CNT_SF, CNT_COMP,
                   CNT_AGN, CNT_LINER if `bpt` is given] [, LOGSFR if `bpt`
                   and `dp_parent` are given].
        WING_DV / WING_SIGMA are the velocity offset and deconvolved sigma of
        the wing of record (widest wing); WING_IDX is its COMP_IDX (-1 if the
        system has no wing).
        REASON is a short human-readable string giving the rule that matched
        (for classified objects) or the first failed check (for 'ambiguous').
        WING_DV is the velocity offset of the wing (same frame as DV), reported
        only when there is exactly one wing (NaN otherwise); subtract V_REF for
        the wing's offset relative to the core reference.
        CNT_N / CNT_M / CNT_B are a width census of the components -- narrow
        (sigma < WIDTH_NARROW_MAX), medium, broad (sigma >= WIDTH_MEDIUM_MAX) --
        and always sum to N_KIN.
        KIN_CLASS is one of:
          n=2: 'doublet', 'singlet+wing', '2width'
          n=3: 'triplet', 'NBN', 'BNB', '3width', 'doublet+wing'
          n=4: 'quadruplet', 'triplet+wing', 'NBN+wing', 'BNB+wing',
               '3width+wing', '4width', '2pairs'
          any: 'broad-dominated', 'single' (n=1), 'ambiguous'
        SYM/ASYM (outer-pair amplitude balance) and WING_KIND (ABSOLUTE/
        RELATIVE/MIXED) ride along as orthogonal flags.

        Amplitudes are read on the anchor line chosen by _pick_amp_line: among
        QUALIFY_ANCHOR_LINES, the lines where ALL components are detected
        (>3 S/N) are ranked by the widest component's amplitude and the largest
        is used; otherwise the brightest line. Reported as AMP_LINE.
        """
        bpt_map = {}
        if bpt is not None:
            bpt_map = {(int(t), int(c)): int(b) for t, c, b
                       in zip(bpt['TARGETID'], bpt['COMP_IDX'], bpt['BPT'])}
        z_map = {}
        if dp_parent is not None:
            z_map = dict(zip(dp_parent['TARGETID'].astype(int), dp_parent['Z'].astype(float)))
        compute_sfr = bool(bpt_map) and bool(z_map)

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
            sig_f = np.maximum(sigma, WIDTH_FLOOR)
            anchor, amp = self._pick_amp_line(sub, kept_index, sig_f)

            kin, flags = self._kinematic_class_one(dv, sigma, flux, amp)
            # map the wing's input-array position back to its COMP_IDX
            wpos = flags.get('WING_IDX')
            flags['WING_IDX'] = (int(kept_index[int(wpos)])
                                 if wpos is not None and np.isfinite(wpos) else -1)
            row = {'KIN_CLASS': kin, **flags, 'AMP_LINE': anchor}
            tid = int(sub['TARGETID'].iloc[0])

            # per-target BPT census over the kept components (if BPT supplied)
            if bpt_map:
                counts = {col: 0 for col in self._BPT_COUNT_COL.values()}
                for ci in kept_index:
                    col = self._BPT_COUNT_COL.get(bpt_map.get((tid, int(ci))))
                    if col:
                        counts[col] += 1
                row.update(counts)

            # SFR from the SF/COMP components only (BPT codes 1, 4). None (NaN)
            # if the system has no SF/COMP component.
            if compute_sfr:
                sf_idx = [int(ci) for ci in kept_index
                          if bpt_map.get((tid, int(ci))) in (1, 4)]
                logsfr, z = np.nan, z_map.get(tid)
                if sf_idx and z is not None:
                    tot = 0.0
                    for ci in sf_idx:
                        cs = sub[sub['COMP_IDX'] == ci].set_index('LINE')
                        if 'Halpha' not in cs.index or 'Hbeta' not in cs.index:
                            continue
                        f_ha = float(cs.loc['Halpha', 'FLUX'])
                        f_hb = float(cs.loc['Hbeta', 'FLUX'])
                        if 'NOISE3' in cs.columns:
                            f_hb = max(f_hb, float(cs.loc['Hbeta', 'NOISE3']) / 3)
                        if f_ha > 0 and f_hb > 0:
                            tot += sfr(f_ha, f_hb, z)          # misc.sfr
                    if tot > 0:
                        logsfr = float(np.log10(tot))
                row['LOGSFR'] = logsfr
            return pd.Series(row)

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
