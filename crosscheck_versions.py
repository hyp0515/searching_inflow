#!/usr/bin/env python
"""
crosscheck_versions.py
======================

Cross-check the fitting-speed optimizations against the original code by
running the FULL DP pipeline under each "version" and diffing the chosen
N_COMP per galaxy. The optimizations are mathematically equivalent per fit,
but the model-selection rule is a discontinuous function of BIC, so a handful
of galaxies sitting on a decision boundary can flip N_COMP between versions.
Those are exactly the fragile galaxies the stability check is meant to flag --
this script finds them and (optionally) hands them straight to stability_check.

All four versions come from the SINGLE src/FITSPECTRUM.py via its toggles, so
there is nothing to keep in sync:

    v0_original    fast=False ajac=False ftol=xtol=1e-8   (pre-optimization)
    v1_vectorize   fast=True  ajac=False ftol=xtol=1e-8
    v2_jacobian    fast=True  ajac=True  ftol=xtol=1e-8
    v3_tolerance   fast=True  ajac=True  ftol=xtol=1e-6   (current default)

Usage
-----
    python crosscheck_versions.py --targetids ./multi_comp.csv --n-jobs 8
    python crosscheck_versions.py --n-sample 500 --versions v0_original v3_tolerance

Outputs (in --outdir, default crosscheck_out/):
    crosscheck_Ncomp.csv   TARGETID x N_COMP for each version + SHIFTED flag
    shifted_targetids.txt   IDs whose N_COMP differs across versions
                            (feed to: stability_check.py --targetids <file>)

Run from the repo root so `from src...` resolves.
"""

import os
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm import tqdm

from src.SPECTRUM import Spectrum
import src.DP_new as DP_new
from src.DP_new import DP

# reuse the target-id resolver (files + bare ints) from the stability script
from stability_check import resolve_targetids


# The version -> FITSPECTRUM-attribute mapping. Order defines column order.
VERSIONS = {
    'v0_original':  dict(use_fast_model=False, use_analytic_jac=False, fit_ftol=1e-8, fit_xtol=1e-8, fit_x_scale=1.0),
    'v1_vectorize': dict(use_fast_model=True,  use_analytic_jac=False, fit_ftol=1e-8, fit_xtol=1e-8, fit_x_scale=1.0),
    'v2_jacobian':  dict(use_fast_model=True,  use_analytic_jac=True,  fit_ftol=1e-8, fit_xtol=1e-8, fit_x_scale=1.0),
    'v3_tolerance': dict(use_fast_model=True,  use_analytic_jac=True,  fit_ftol=1e-6, fit_xtol=1e-6, fit_x_scale=1.0),
}


def apply_version(cfg):
    """Set the toggles on the shared module-level FIT that DP.fit_dp uses.
    Done inside each worker call so it is correct under joblib processes too."""
    for k, v in cfg.items():
        setattr(DP_new.FIT, k, v)


_DP = DP()


def ncomp_for(data, tid, cfg):
    """Chosen N_COMP for one target under one version (-1 on failure)."""
    apply_version(cfg)
    try:
        parent, _comps, _model = _DP.fit_dp(data_class=data, id=tid)
        return int(parent['N_COMP'])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--spectra', default='/Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits')
    ap.add_argument('--cigale', default='/Users/hyp0515/data/IronPhysProp_v1.2_extracted.fits')
    ap.add_argument('--fastspec', default='/Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_fastspecfit.fits')
    ap.add_argument('--subtype-filter', default='QSO')
    ap.add_argument('--targetids', type=str, nargs='*', default=None,
                    help="Bare integers and/or CSV/txt ID files (e.g. ./multi_comp.csv). "
                         "If omitted, draw --n-sample at random.")
    ap.add_argument('--n-sample', type=int, default=300)
    ap.add_argument('--versions', nargs='*', default=list(VERSIONS),
                    help=f"Subset/order of versions to compare. Choices: {list(VERSIONS)}")
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default='crosscheck_out')
    args = ap.parse_args()

    versions = [v for v in args.versions if v in VERSIONS]
    if len(versions) < 2:
        ap.error("need at least two valid --versions to compare")
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading spectra ...")
    data = Spectrum(fits.open(args.spectra), fits.open(args.cigale), fits.open(args.fastspec),
                    load_targetID=None, subtype_filter=args.subtype_filter)

    if args.targetids:
        requested = resolve_targetids(args.targetids)
        tids = [t for t in requested if int(t) in data._id_to_idx]
        skipped = len(requested) - len(tids)
        if skipped:
            print(f"  {skipped} requested IDs not in sample, skipped.")
    else:
        rng = np.random.default_rng(args.seed)
        tids = list(rng.choice(data.targetID, size=min(args.n_sample, data.n_spectra),
                               replace=False).astype(np.int64))
    print(f"Comparing {len(versions)} versions on {len(tids)} targets "
          f"(n_jobs={args.n_jobs}).\n")

    # N_COMP for every (target, version)
    table = {'TARGETID': [int(t) for t in tids]}
    for v in versions:
        cfg = VERSIONS[v]
        col = Parallel(n_jobs=args.n_jobs)(
            delayed(ncomp_for)(data, tid, cfg) for tid in tqdm(tids, desc=v))
        table[v] = col

    df = pd.DataFrame(table)
    ncols = df[versions]
    df['SHIFTED'] = ncols.nunique(axis=1) > 1          # any disagreement across versions
    df['N_COMP_RANGE'] = ncols.max(axis=1) - ncols.min(axis=1)

    path = os.path.join(args.outdir, 'crosscheck_Ncomp.csv')
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")

    shifted = df[df['SHIFTED']]
    idfile = os.path.join(args.outdir, 'shifted_targetids.txt')
    with open(idfile, 'w') as fh:
        for t in shifted['TARGETID']:
            fh.write(f"{t}\n")
    print(f"wrote {idfile}  ({len(shifted)} shifted IDs)")

    # ---- summary ----
    n = len(df)
    print(f"\n{len(shifted)}/{n} galaxies ({100*len(shifted)/max(n,1):.1f}%) shift N_COMP "
          f"across the compared versions.")
    # pairwise disagreement counts (esp. original vs current)
    print("pairwise N_COMP disagreements:")
    for a, b in combinations(versions, 2):
        d = int((df[a] != df[b]).sum())
        print(f"  {a:14s} vs {b:14s}: {d}")
    # how the current version's N_COMP differs from the original
    if 'v0_original' in versions and 'v3_tolerance' in versions:
        delta = (df['v3_tolerance'] - df['v0_original'])
        moved = delta[delta != 0]
        if len(moved):
            print("\nv3_tolerance - v0_original transitions (count):")
            print(moved.value_counts().sort_index().to_string())
    print(f"\nNext: cross-check these against the stability tool:\n"
          f"  python stability_check.py --targetids {idfile} --restarts 100 --tiers 2\n"
          f"Expectation: the shifted galaxies show modal_frac < 1 (fragile model choice).")


if __name__ == '__main__':
    main()
