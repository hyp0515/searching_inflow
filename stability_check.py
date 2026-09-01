#!/usr/bin/env python
"""
stability_check.py
==================

Convergence / stability sanity checks for the emission-line fitting pipeline
(src/FITSPECTRUM.py + src/DP_new.py).

The production fit is *deterministic*: `curve_fit` is a local optimizer and the
starting point `p0` is fixed, so a galaxy always returns the same answer. These
checks ask the question that determinism hides: does that single starting point
find the *global* optimum, or could a different start land in a different local
minimum? We answer it by randomizing `p0` (via the `p0_perturb` hook added to
FitSpectrum) and looking at the spread of outcomes.

Two tiers
---------
Tier 1  (per-N multi-start).  For a galaxy and a fixed component count N, run K
        restarts from randomized starting points and measure how tightly the
        fits cluster: what fraction reach the best chi^2 found (the basin of
        attraction), how many distinct chi^2 plateaus appear (proxy for the
        number of local minima), and the scatter of the recovered kinematic
        parameters (sigma, dv). Components are sorted blue->red by dv before
        comparison so exchangeable-component "label switching" is not mistaken
        for instability.

Tier 2  (end-to-end model selection).  Run the FULL DP.fit_dp under the same
        randomized restarts and tabulate the distribution of the chosen N_COMP.
        This is the quantity the pipeline actually emits. Also records BIC_1..4
        each restart so the scatter of the decisive BIC differences can be
        compared against the selection thresholds (DELTA_BIC_DECISIVE = 15,
        DELTA_BIC_SIMPLICITY = 5).

Usage
-----
    python stability_check.py \
        --spectra /Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits \
        --cigale  /Users/hyp0515/data/IronPhysProp_v1.2_extracted.fits \
        --fastspec /Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_fastspecfit.fits \
        --targetids 39627... 39627... \
        --restarts 100 --mode uniform --plots --outdir stability_out

If --targetids is omitted, --n-sample galaxies are drawn at random from the
loaded sample (use --seed for reproducibility). Run it from the repo root so
that `from src...` imports resolve.

Nothing here modifies the pipeline's default behaviour; the randomization is
opt-in through the p0_perturb hook and is reset after every use.
"""

import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from astropy.io import fits

from src.SPECTRUM import Spectrum
from src.FITSPECTRUM import FitSpectrum
import src.DP_new as DP_new
from src.DP_new import DP, N_MODELS, PARAMS_PER_COMP, DELTA_BIC_DECISIVE, DELTA_BIC_SIMPLICITY


# --------------------------------------------------------------------------- #
#  Target-ID resolution (bare integers and/or CSV/txt files)                  #
# --------------------------------------------------------------------------- #
def load_ids_from_file(path):
    """
    Read TARGETIDs from a CSV/text file. Handles the repo's plain one-ID-per-line
    files (e.g. extreme_2_comp.csv, multi_comp.csv, 4_comp.csv -- no header), as
    well as files with a header and/or several columns (a 'TARGETID' column is
    used if present, otherwise the first column). Non-integer lines such as a
    header row are silently skipped.
    """
    def to_int(s):
        try:
            return int(str(s).strip())
        except (ValueError, TypeError):
            return None

    with open(path) as fh:
        rows = [ln.strip() for ln in fh if ln.strip()]
    if not rows:
        return []

    ids = []
    if ',' in rows[0]:                                   # multi-column
        cols = [c.strip() for c in rows[0].split(',')]
        # header row only if NO token is an integer (a data row always has at
        # least the integer TARGETID; float columns like RA/Z must not trigger it)
        header_present = all(to_int(c) is None for c in cols)
        upper = [c.upper() for c in cols]
        tid_col = upper.index('TARGETID') if 'TARGETID' in upper else 0
        data_rows = rows[1:] if header_present else rows
        for r in data_rows:
            parts = [p.strip() for p in r.split(',')]
            if tid_col < len(parts):
                v = to_int(parts[tid_col])
                if v is not None:
                    ids.append(v)
    else:                                                # single column
        for r in rows:                                   # a header line -> None -> skipped
            v = to_int(r)
            if v is not None:
                ids.append(v)
    return ids


def resolve_targetids(tokens):
    """
    Expand each --targetids token: existing file -> read IDs from it; otherwise
    parse as an integer. Order preserved, duplicates removed. Multiple files
    and loose integers can be mixed on one command line.
    """
    out = []
    for tok in tokens:
        if os.path.isfile(tok):
            found = load_ids_from_file(tok)
            print(f"  loaded {len(found)} TARGETIDs from {tok}")
            out.extend(found)
        else:
            v = None
            try:
                v = int(tok)
            except ValueError:
                print(f"  warning: --targetids entry {tok!r} is neither a file "
                      f"nor an integer; skipped.")
            if v is not None:
                out.append(v)
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# --------------------------------------------------------------------------- #
#  Randomized starting points                                                 #
# --------------------------------------------------------------------------- #
def make_perturber(mode, rng, scale=0.5):
    """
    Return a callable(p0, lo, hi) -> new_p0 that FitSpectrum will use as the
    curve_fit starting vector. Two strategies:

      'uniform'  draw every parameter uniformly across its full bound [lo, hi].
                 This maximally disperses the starts and is the most powerful
                 probe for *multiple* minima -- use it for the main test.
      'jitter'   multiplicative Gaussian noise (fractional sigma = `scale`)
                 around the pipeline's own default p0, plus a small additive
                 kick sized to the bound range so parameters whose default is 0
                 (e.g. dv) still move. Milder -- tests robustness to *local*
                 perturbations of the default start.

    The vector is clipped back inside the bounds by FitSpectrum regardless.
    """
    if mode == 'uniform':
        def perturb(p0, lo, hi):
            return rng.uniform(lo, hi)
        return perturb

    if mode == 'jitter':
        def perturb(p0, lo, hi):
            span = hi - lo
            mult = p0 * (1.0 + rng.normal(0.0, scale, size=p0.shape))
            add = rng.normal(0.0, 0.05 * scale, size=p0.shape) * span
            return mult + add
        return perturb

    raise ValueError(f"unknown perturb mode {mode!r} (use 'uniform' or 'jitter')")


# --------------------------------------------------------------------------- #
#  Small helpers                                                              #
# --------------------------------------------------------------------------- #
def chisq_of(cflux, cmodel, csigma):
    return float(np.sum(((cflux - cmodel) / csigma) ** 2))


def bic_of(chisq, n_comp, n_data):
    """Same definition DP.fit_dp uses: k*ln(N) + chi^2, k = PARAMS_PER_COMP*N."""
    k = PARAMS_PER_COMP * n_comp
    return float(k * np.log(n_data) + chisq)


def sorted_kinematics(params):
    """
    Extract (sigmas, dvs) as arrays sorted blue->red by dv, from a FITSPECTRUM
    `params` dict of any N. Sorting by dv makes exchangeable components line up
    across restarts, so genuine scatter is separated from label switching.
    """
    n = params.get('n_components', 1)
    if n == 1:
        return np.array([float(params['sigma'])]), np.array([0.0])
    sig = np.asarray(params['sigma'], dtype=float)
    dv = np.asarray(params['dv'], dtype=float)
    order = np.argsort(dv)
    return sig[order], dv[order]


def robust_std(x):
    """MAD-based robust sigma (0 for <2 points)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# --------------------------------------------------------------------------- #
#  Tier 1 -- per-N multi-start                                                #
# --------------------------------------------------------------------------- #
def tier1_for_target(FIT, data_class, tid, n_restarts, mode, scale,
                     base_seed, chisq_rtol=1e-3, chisq_atol=1.0):
    """
    Returns (summary_rows, raw_rows) for one target across N = 1..N_MODELS.
    """
    n_data = None
    summary_rows, raw_rows = [], []

    for n_comp in range(1, N_MODELS + 1):
        chisqs, bics = [], []
        sig_list, dv_list = [], []
        n_fail = 0

        for r in range(n_restarts):
            rng = np.random.default_rng((base_seed, int(tid), n_comp, r))
            perturb = make_perturber(mode, rng, scale)
            try:
                params, (clam, cflux, csigma, cmodel), slice_idx, n_lines, _ = \
                    FIT.fit_multi_emission_vel(data_class=data_class, id=tid,
                                               n_components=n_comp, w_dz=False,
                                               p0_perturb=perturb)
                n_data = len(cflux)
                cs = chisq_of(cflux, cmodel, csigma)
                sig, dv = sorted_kinematics(params)
                chisqs.append(cs)
                bics.append(bic_of(cs, n_comp, n_data))
                sig_list.append(sig)
                dv_list.append(dv)
                raw_rows.append({'TARGETID': int(tid), 'N': n_comp, 'restart': r,
                                 'chisq': cs, 'bic': bics[-1],
                                 'sigma': ','.join(f'{s:.1f}' for s in sig),
                                 'dv': ','.join(f'{d:.1f}' for d in dv)})
            except Exception as exc:  # bad start / no convergence
                n_fail += 1
                raw_rows.append({'TARGETID': int(tid), 'N': n_comp, 'restart': r,
                                 'chisq': np.nan, 'bic': np.nan,
                                 'sigma': '', 'dv': '', 'error': str(exc)[:80]})

        n_ok = len(chisqs)
        if n_ok == 0:
            summary_rows.append({'TARGETID': int(tid), 'N': n_comp, 'n_ok': 0,
                                 'n_fail': n_fail})
            continue

        chisqs = np.asarray(chisqs)
        best = float(chisqs.min())
        thr = max(best * (1.0 + chisq_rtol), best + chisq_atol)
        basin = chisqs <= thr
        basin_frac = float(basin.mean())

        # coarse count of distinct chi^2 plateaus (proxy for # local minima)
        tol = max(chisq_atol, chisq_rtol * best)
        n_clusters = int(np.unique(np.round(chisqs / tol)).size)

        # scatter of kinematics among the basin members (aligned by dv order)
        sig_arr = np.vstack([sig_list[i] for i in np.where(basin)[0]])
        dv_arr = np.vstack([dv_list[i] for i in np.where(basin)[0]])
        sigma_scatter = float(np.mean([robust_std(sig_arr[:, c])
                                       for c in range(sig_arr.shape[1])]))
        dv_scatter = float(np.mean([robust_std(dv_arr[:, c])
                                    for c in range(dv_arr.shape[1])]))
        sigma_med = ','.join(f'{v:.1f}' for v in np.median(sig_arr, axis=0))
        dv_med = ','.join(f'{v:.1f}' for v in np.median(dv_arr, axis=0))

        summary_rows.append({
            'TARGETID': int(tid), 'N': n_comp,
            'n_ok': n_ok, 'n_fail': n_fail,
            'best_chisq': best,
            'basin_frac': basin_frac,       # 1.0 = every start found the best min
            'n_chisq_clusters': n_clusters,  # >1 => multiple local minima
            'sigma_scatter': sigma_scatter,  # km/s, within-basin robust std
            'dv_scatter': dv_scatter,        # km/s
            'sigma_median': sigma_med,
            'dv_median': dv_med,
        })

    return summary_rows, raw_rows


# --------------------------------------------------------------------------- #
#  Tier 2 -- end-to-end model-selection stability                            #
# --------------------------------------------------------------------------- #
def tier2_for_target(dp, data_class, tid, n_restarts, mode, scale, base_seed):
    """
    Returns (summary_row, raw_rows). Randomizes the shared module-level FIT that
    DP.fit_dp uses, one restart at a time (kept sequential because the hook is
    shared mutable state), and records the chosen N_COMP + BIC_1..4 each time.
    """
    raw_rows = []
    best_ns, bic_cols = [], {n: [] for n in range(1, N_MODELS + 1)}

    prev = DP_new.FIT.p0_perturb
    try:
        for r in range(n_restarts):
            rng = np.random.default_rng((base_seed, int(tid), 999, r))
            DP_new.FIT.p0_perturb = make_perturber(mode, rng, scale)
            try:
                parent, comps, _model = dp.fit_dp(data_class=data_class, id=tid)
                bn = int(parent['N_COMP'])
                best_ns.append(bn)
                row = {'TARGETID': int(tid), 'restart': r, 'N_COMP': bn}
                for n in range(1, N_MODELS + 1):
                    v = float(parent.get(f'BIC_{n}', np.nan))
                    bic_cols[n].append(v)
                    row[f'BIC_{n}'] = v
                raw_rows.append(row)
            except Exception as exc:
                raw_rows.append({'TARGETID': int(tid), 'restart': r,
                                 'N_COMP': -1, 'error': str(exc)[:80]})
    finally:
        DP_new.FIT.p0_perturb = prev  # always restore

    if not best_ns:
        return {'TARGETID': int(tid), 'n_ok': 0}, raw_rows

    cnt = Counter(best_ns)
    modal_n, modal_hits = cnt.most_common(1)[0]
    n_ok = len(best_ns)

    def bic_arr(n):
        return np.asarray([v for v in bic_cols[n] if np.isfinite(v)], dtype=float)

    def delta_scatter(a, b):
        """robust std of (BIC_a - BIC_b) over restarts where both are finite."""
        x, y = np.asarray(bic_cols[a]), np.asarray(bic_cols[b])
        m = np.isfinite(x) & np.isfinite(y)
        return robust_std((x - y)[m]) if m.sum() >= 2 else 0.0

    summary = {
        'TARGETID': int(tid),
        'n_ok': n_ok,
        'modal_N_COMP': int(modal_n),
        'modal_frac': modal_hits / n_ok,            # 1.0 = perfectly stable choice
        'N_COMP_dist': dict(sorted(cnt.items())),   # e.g. {1: 3, 2: 97}
        # scatter of the two BIC differences the selector hinges on, vs their
        # decision thresholds: scatter >~ threshold means that call is fragile.
        'd(BIC2-BIC4)_scatter': delta_scatter(2, 4),
        'thr_simplicity': DELTA_BIC_SIMPLICITY,
        'd(BIC_min_gap)_scatter': _min_gap_scatter(bic_cols),
        'thr_decisive': DELTA_BIC_DECISIVE,
    }
    return summary, raw_rows


def _min_gap_scatter(bic_cols):
    """Robust std, over restarts, of the gap between the best and 2nd-best BIC
    (how much the decisive margin itself wobbles)."""
    ns = sorted(bic_cols)
    mat = np.vstack([np.asarray(bic_cols[n]) for n in ns]).T  # restarts x N
    gaps = []
    for row in mat:
        finite = np.sort(row[np.isfinite(row)])
        if finite.size >= 2:
            gaps.append(finite[1] - finite[0])
    return robust_std(gaps)


# --------------------------------------------------------------------------- #
#  Optional per-target diagnostic plots                                       #
# --------------------------------------------------------------------------- #
def plot_target(tid, t1_raw, t2_raw, outdir):
    import matplotlib.pyplot as plt
    t1 = pd.DataFrame(t1_raw)
    t2 = pd.DataFrame(t2_raw)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # BIC per restart, one series per N (the quantity model selection acts on)
    for n_comp, sub in t1.groupby('N'):
        axes[0].plot(sub['restart'], sub['bic'], '.', ms=4, label=f'N={n_comp}')
    axes[0].set_xlabel('restart'); axes[0].set_ylabel('BIC')
    axes[0].set_title('Tier 1: fit BIC across randomized starts')
    axes[0].legend(fontsize=8, frameon=False)

    # histogram of the chosen N_COMP
    if 'N_COMP' in t2.columns:
        good = t2[t2['N_COMP'] > 0]['N_COMP']
        bins = np.arange(0.5, N_MODELS + 1.5, 1)
        axes[1].hist(good, bins=bins, rwidth=0.8)
        axes[1].set_xticks(range(1, N_MODELS + 1))
    axes[1].set_xlabel('chosen N_COMP'); axes[1].set_ylabel('restarts')
    axes[1].set_title('Tier 2: model-selection stability')

    fig.suptitle(f'TARGETID {tid}')
    fig.tight_layout()
    path = os.path.join(outdir, f'stability_{tid}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spectra', default='/Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits')
    ap.add_argument('--cigale', default='/Users/hyp0515/data/IronPhysProp_v1.2_extracted.fits')
    ap.add_argument('--fastspec', default='/Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_fastspecfit.fits')
    ap.add_argument('--subtype-filter', default='QSO',
                    help="SPECTYPE to exclude when loading (matches the notebook).")
    ap.add_argument('--targetids', type=str, nargs='*', default=None,
                    help="TARGETIDs to test: bare integers and/or paths to a "
                         "CSV/txt list of IDs (e.g. ./extreme_2_comp.csv, "
                         "./multi_comp.csv). Files and integers may be mixed. "
                         "If omitted, draw --n-sample at random.")
    ap.add_argument('--n-sample', type=int, default=8,
                    help="How many random targets to test when --targetids is not given.")
    ap.add_argument('--restarts', type=int, default=100, help="Randomized starts per target.")
    ap.add_argument('--mode', choices=['uniform', 'jitter'], default='uniform')
    ap.add_argument('--scale', type=float, default=0.5, help="Fractional jitter (mode='jitter').")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tiers', default='12', help="Which tiers to run: '1', '2', or '12'.")
    ap.add_argument('--plots', action='store_true', help="Save a diagnostic PNG per target.")
    ap.add_argument('--outdir', default='stability_out')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading spectra ...")
    spectra_data = fits.open(args.spectra)
    cigale_data = fits.open(args.cigale)
    fastspec = fits.open(args.fastspec)
    data = Spectrum(spectra_data, cigale_data, fastspec,
                    load_targetID=None, subtype_filter=args.subtype_filter)

    if args.targetids:
        requested = resolve_targetids(args.targetids)
        tids = [t for t in requested if int(t) in data._id_to_idx]
        missing = set(requested) - set(tids)
        if missing:
            print(f"  warning: {len(missing)} requested TARGETIDs not in sample, skipped.")
    else:
        rng = np.random.default_rng(args.seed)
        tids = list(rng.choice(data.targetID, size=min(args.n_sample, data.n_spectra),
                               replace=False).astype(np.int64))
    print(f"Testing {len(tids)} targets, {args.restarts} restarts each, mode={args.mode}.\n")

    FIT = FitSpectrum()
    dp = DP()

    t1_summary, t1_raw_all = [], []
    t2_summary, t2_raw_all = [], []

    for tid in tids:
        print(f"--- TARGETID {tid} ---")
        t1_raw = t2_raw = []
        if '1' in args.tiers:
            s1, t1_raw = tier1_for_target(FIT, data, tid, args.restarts,
                                          args.mode, args.scale, args.seed)
            t1_summary.extend(s1); t1_raw_all.extend(t1_raw)
            for row in s1:
                if row.get('n_ok'):
                    print(f"  N={row['N']}: basin={row['basin_frac']:.2f} "
                          f"clusters={row['n_chisq_clusters']} "
                          f"sigma_scatter={row['sigma_scatter']:.1f} "
                          f"dv_scatter={row['dv_scatter']:.1f} "
                          f"({row['n_fail']} failed starts)")
        if '2' in args.tiers:
            s2, t2_raw = tier2_for_target(dp, data, tid, args.restarts,
                                          args.mode, args.scale, args.seed)
            t2_summary.append(s2); t2_raw_all.extend(t2_raw)
            if s2.get('n_ok'):
                print(f"  model selection: modal N_COMP={s2['modal_N_COMP']} "
                      f"stable in {s2['modal_frac']*100:.0f}% of starts, "
                      f"dist={s2['N_COMP_dist']}")
                print(f"    d(BIC2-BIC4) scatter={s2['d(BIC2-BIC4)_scatter']:.1f} "
                      f"(vs simplicity thr {s2['thr_simplicity']}); "
                      f"decisive-gap scatter={s2['d(BIC_min_gap)_scatter']:.1f} "
                      f"(vs decisive thr {s2['thr_decisive']})")
        if args.plots and (t1_raw or t2_raw):
            p = plot_target(tid, t1_raw, t2_raw, args.outdir)
            print(f"  plot -> {p}")
        print()

    # ------ persist ------
    def dump(rows, name):
        if rows:
            path = os.path.join(args.outdir, name)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"wrote {path}")

    dump(t1_summary, 'tier1_summary.csv')
    dump(t1_raw_all, 'tier1_raw.csv')
    dump(t2_summary, 'tier2_summary.csv')
    dump(t2_raw_all, 'tier2_raw.csv')

    # ------ headline read ------
    if t2_summary:
        fr = np.array([s['modal_frac'] for s in t2_summary if s.get('n_ok')])
        if fr.size:
            print(f"\nTier 2 headline: median model-selection stability "
                  f"{np.median(fr)*100:.0f}%; "
                  f"{int((fr < 0.9).sum())}/{fr.size} targets flip N_COMP in >10% of starts.")
    if t1_summary:
        bf = np.array([s['basin_frac'] for s in t1_summary if s.get('n_ok')])
        if bf.size:
            print(f"Tier 1 headline: median basin fraction {np.median(bf):.2f} "
                  f"across all (target, N); low values flag multi-minimum fits.")


if __name__ == '__main__':
    main()
