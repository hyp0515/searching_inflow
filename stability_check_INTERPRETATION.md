# Interpreting `stability_check.py` output

A field guide to reading the convergence / stability diagnostics. The script
randomizes the `curve_fit` starting point (`p0`) many times per galaxy and asks
two questions:

- **Tier 1 — is each N-component fit well-determined?** Given a fixed component
  count N, do independent random starts converge to the same solution, or to
  several competing local minima?
- **Tier 2 — is the *chosen* N_COMP stable?** When the full `DP.fit_dp`
  pipeline (all N + the BIC/qualify selection logic) is run from random starts,
  does it keep picking the same number of components?

Keep the distinction in mind: Tier 1 measures *fitting* robustness at fixed N,
Tier 2 measures *model-selection* robustness. A galaxy can be perfectly stable
in Tier 1 yet flip in Tier 2 (two models nearly tied in BIC), or vice versa.

The production pipeline is deterministic (fixed `p0`), so none of this changes
your catalog. It tells you whether that one fixed start is trustworthy.

---

## Files produced (in `--outdir`, default `stability_out/`)

| File | One row per | Purpose |
|------|-------------|---------|
| `tier1_summary.csv` | (target, N) | headline Tier-1 metrics |
| `tier1_raw.csv` | (target, N, restart) | every restart's χ², BIC, σ, dv — for your own plots |
| `tier2_summary.csv` | target | headline Tier-2 metrics |
| `tier2_raw.csv` | (target, restart) | chosen N_COMP and BIC_1..4 each restart |
| `stability_<TARGETID>.png` | target (with `--plots`) | χ² across restarts (left) + N_COMP histogram (right) |

---

## Tier 1 — `tier1_summary.csv`

| Column | Meaning |
|--------|---------|
| `n_ok` / `n_fail` | restarts that converged / raised. Many failures means many random starts can't fit at all — usually a low-S/N line or a start colliding with a bound. |
| `best_chisq` | lowest χ² found across all restarts for that N — the best candidate for the global optimum. |
| `basin_frac` | **the headline number.** Fraction of successful restarts that reached `best_chisq`. 1.0 = every start finds the same (global) minimum. |
| `n_chisq_clusters` | count of distinct χ² plateaus. 1 = unimodal surface; >1 = that many local minima. |
| `sigma_scatter` | within-basin robust σ (km/s) of the fitted velocity dispersions, averaged over components (sorted blue→red). Spread of the *kinematics* among starts that agree on χ². |
| `dv_scatter` | same for the velocity offsets (km/s). |
| `sigma_median` / `dv_median` | the median recovered values, for context. |

**How to read it**

- `basin_frac` ≈ 1.0 and `n_chisq_clusters` = 1 → that N-component fit is
  well-posed; the fixed-`p0` production answer is the global optimum. Nothing to worry about.
- `basin_frac` 0.5–0.9 → a shallower secondary minimum exists but most starts
  still find the global one. The shipped fit is *probably* right but not
  guaranteed. Worth spot-checking.
- `basin_frac` < 0.5 → the global optimum is a *minority* basin. A fixed start
  can easily miss it. For these, consider a multi-start-and-keep-best strategy
  in production, or treat the parameters as start-dependent.
- `sigma_scatter` / `dv_scatter`: judge against physical scale, not absolute
  size. A few km/s is negligible. Scatter approaching the DESI resolution
  (~30 km/s) or the component separations you care about means the kinematics
  are not pinned down even when χ² agrees — a degeneracy the χ² surface is flat
  along.

**Two traps**

- *Higher N is expected to look worse.* N=3, 4 are non-convex; low `basin_frac`
  there is normal and is partly *why* you don't want to over-fit. The number
  that matters for your science is Tier 1 at the N the pipeline actually chose
  (see Tier 2's `modal_N_COMP`).
- *Pinned components fake stability.* If `sigma_median` for a component sits at
  ~5 (SIGMA_FLOOR) or ~699 (SIGMA_CEIL), its `sigma_scatter` will look tiny —
  but that's a wall, not a resolved parameter. The pipeline already rejects
  these via `has_pinned_sigma`; here they're a sign that N is too high for that
  galaxy.

---

## Tier 2 — `tier2_summary.csv`

| Column | Meaning |
|--------|---------|
| `modal_N_COMP` | the most frequently chosen component count across restarts. |
| `modal_frac` | **the headline number.** Fraction of restarts that chose `modal_N_COMP`. 1.0 = the model choice is rock-solid; 0.5 = essentially a coin flip. |
| `N_COMP_dist` | the full tally, e.g. `{1: 3, 2: 97}` or `{2: 55, 4: 45}`. Shows *which* models it flips between. |
| `d(BIC2-BIC4)_scatter` | robust σ of `BIC_2 − BIC_4` across restarts. |
| `thr_simplicity` | the `DELTA_BIC_SIMPLICITY` constant (5) that the 2-vs-4 decision compares against. |
| `d(BIC_min_gap)_scatter` | robust σ of the gap between the best and second-best BIC — how much the "winning margin" itself wobbles. |
| `thr_decisive` | the `DELTA_BIC_DECISIVE` constant (15). |

**How to read it**

- `modal_frac` ≥ 0.9 → the pipeline's N_COMP (and therefore the downstream
  BPT / kinematic class for that galaxy) is robust to the starting point. Trust it.
- `modal_frac` < 0.9 → **flagged.** The classification is start-dependent.
  These are exactly the galaxies to send to MCMC (Tier 3): the disagreement
  usually means two models are near-degenerate and BIC alone can't separate
  them. Look at `N_COMP_dist` to see the competing hypotheses (a 1-vs-2 flip is
  a "is there a second component at all" question; a 2-vs-4 flip is a
  "how many" question).

**Are the BIC decisions fragile?** Compare each scatter to its threshold on the
same row:

- `d(BIC2-BIC4)_scatter` ≳ `thr_simplicity` (5) → the 2-vs-4 tie-break is
  inside its own noise. That specific rule is fragile for this galaxy by
  construction, independent of anything else.
- `d(BIC_min_gap)_scatter` ≳ `thr_decisive` (15) → what looks like a "decisive"
  BIC winner isn't decisive once you account for start-to-start variation.

If these scatters are large across *many* galaxies, the thresholds themselves
(not just individual fits) deserve a second look.

---

## The plots (`--plots`)

- **Left (χ² vs restart, one series per N):** flat horizontal bands = a unique
  minimum per N. Vertically split points = multiple minima (each band is one).
  The vertical ordering of the bands is your BIC race in raw-χ² form.
- **Right (histogram of chosen N_COMP):** a single tall bar = stable model
  choice (`modal_frac` ≈ 1). Two comparable bars = the flipping galaxies to
  escalate.

---

## Recommended workflow

1. **Sample health check.** Read the two headline lines the script prints:
   median `basin_frac` (Tier 1) and median `modal_frac` (Tier 2), plus the count
   of targets that flip N_COMP in >10% of starts. This is your one-line verdict
   on whether the fixed-`p0` pipeline is trustworthy sample-wide.
2. **Triage.** Sort `tier2_summary.csv` by `modal_frac` ascending. The bottom of
   that list is your MCMC shortlist — the hard cases where the deterministic
   answer is a genuine gamble.
3. **Diagnose each flagged case.** For a flagged target, open its Tier-1 rows at
   the competing N values: is the flip because a fit is multi-minimum
   (`basin_frac` low → *fitting* problem, fixable with multi-start) or because
   the BIC margin is inside its scatter (`basin_frac` high but BIC scatters
   large → *selection* problem, needs a better criterion or MCMC evidence)?

---

## Modes: `--mode uniform` vs `--mode jitter`

- **`uniform`** draws every parameter across its full bound — a global stress
  test. Low `basin_frac` here is a *hard* bar and the most honest probe for
  multiple minima. Use this for the main run.
- **`jitter`** perturbs the pipeline's own default start by a fractional amount
  (`--scale`). This asks the narrower question "is the *shipped* start locally
  robust?" — a high `jitter` `basin_frac` with a low `uniform` one means the
  default start sits in the global basin even though other basins exist (good
  news for the current catalog).

A useful manual cross-check: run the ordinary deterministic fit and compare its
χ² to `best_chisq` from the uniform run. If they match, the shipped answer *is*
the global optimum despite any competing basins.

---

## Quick reference — thresholds

| Metric | Solid | Watch | Concern |
|--------|-------|-------|---------|
| `basin_frac` (at the chosen N) | ≥ 0.9 | 0.5–0.9 | < 0.5 |
| `n_chisq_clusters` | 1 | 2–3 | many |
| `modal_frac` | ≥ 0.9 | 0.7–0.9 | < 0.7 |
| `sigma_scatter` / `dv_scatter` | ≪ resolution (~30 km/s) | ~ resolution | ≳ component separation |
| BIC-diff scatter vs its threshold | ≪ threshold | ~ half | ≳ threshold |

These are rules of thumb, not hard cuts — tune them to your science tolerance.
