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

        best_n = int(np.argmin(bic_by_n)) + 1
        best_params = params_by_n[best_n - 1]
        clam, cflux, csigma, slice_idx = extras_by_n[best_n - 1]

        # --- per-region background noise (component-count independent) ------
        noise_region = np.split(csigma, slice_idx)
        sigmab_region = [np.sqrt(np.mean(seg ** 2)) if seg.size > 1 else np.inf
                         for seg in noise_region]

        # --- build the tidy component rows ---------------------------------
        comps = self._iter_components(best_params)
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
                noise3 = flux(3 * sig_b, comp['comp_sigma'], lam0, dv=dv)

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
                    'SIG':       bool(amp > 3 * sig_b),   # >3 sigma detection
                    'NOISE3':    np.float32(noise3),
                })

        # --- parent scalar row (N-independent quantities) ------------------
        parent_row = {
            'TARGETID': np.int64(row['TARGETID']),
            'RA':       np.float32(RA),
            'DEC':      np.float32(DEC),
            'Z':        np.float32(Z),
            'LOGM':     np.float32(LOGM),
            'LOGSFR':   np.float32(LOGSFR),
            'N_COMP':   best_n,
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
        """
        need = ['OIII5007', 'Hbeta', 'NII6583', 'Halpha']

        def classify(sub):
            f = sub.set_index('LINE')['FLUX']
            snr = sub.set_index('LINE')['SNR']
            if any(l not in f.index for l in need):
                return 256  # unclassified
            if (snr[need] <= 0).sum() > 1:
                return 256
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                oiii_hb = np.log10(f['OIII5007'] / f['Hbeta'])
                nii_ha = np.log10(f['NII6583'] / f['Halpha'])
            sf = 0.61 / (nii_ha - 0.05) + 1.30
            comp = 0.61 / (nii_ha - 0.47) + 1.19
            liner = 1.05 * nii_ha + 0.45
            if (sf > oiii_hb or comp > oiii_hb) and (nii_ha < 0.47):
                return 1 if (sf > oiii_hb and nii_ha < 0.05) else 4
            elif (sf < oiii_hb or comp < oiii_hb):
                return 64 if liner > oiii_hb else 16
            return 256

        out = (dp_components
               .groupby(['TARGETID', 'COMP_IDX'])
               .apply(classify)
               .rename('BPT')
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
