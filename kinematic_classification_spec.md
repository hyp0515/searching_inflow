# Kinematic (line-profile shape) classification — spec

Bottom-up, morphological. Each component is an *atom*; a system's class is composed
from the atoms. Physical interpretation (rotation vs outflow) is applied later, from
the morphology + BPT — it is **not** part of these labels.

## Atoms (per component)
- **w** = velocity dispersion σ, floored at `CLASS_SIGMA_FLOOR` (30 km/s).
- **a** = peak amplitude on one consistent anchor line (same line for all components).
- **v** = velocity offset.

## Pairwise predicates
- **similar(i,j)**: `max(w)/min(w) ≤ SIM` **or** both `w < NARROW`. (`SIM=2.0`, `NARROW=90`)
- **resolved(i,j)**: `|v_i − v_j| > SEP · (w_i + w_j)`. (`SEP=1.0`)
  Rationale: two equal Gaussians of width σ show a central dip (two distinct peaks)
  iff their separation `Δ > 2σ`. The symmetric generalisation `Δ > w_i + w_j` reduces
  to that for equal widths and, with the 30 km/s floor, gives a 60 km/s minimum at the
  narrow end (matching the old absolute `SD_MIN_SEP`) while scaling up for wide pairs.
  Only used where amplitudes are comparable (the regime in which the dip test is valid).

## Broadness and wings
- **broad(i)** = `w_i ≥ BROAD_ABS` (200; provenance **ABSOLUTE**) **or** i is the single
  widest with `w_i ≥ NARROW` and `w_i / w_2nd > SIM` (provenance **RELATIVE**).
  The `≥ NARROW` floor prevents a narrow rim from ever being called broad.
- **core** = not broad. `a_core_max` = brightest core amplitude.
- Each broad component is:
  - **strong** if `a_i > a_core_max`  → the system is **`broad-dominated`** (stop).
  - **wing** (faint) if `a_core_max / a_i ≥ WING_WEAK` (1.5).
  - **comparable** otherwise (kept only as an n-width layer; see below).
- The **wing of record** (reported `WING_DV`/`WING_KIND`) is the **widest** broad
  component when a system has more than one.
- `WING_KIND` per system = ABSOLUTE / RELATIVE / MIXED over its wings.

## Amplitude balance (orthogonal flag, not a class)
Folded ratio `r = max(a)/min(a)` over the relevant peak set:
`r ≤ AMP_SYM (2)` → **SYM**; `≤ AMP_ASYM (4)` → **ASYM**; `> AMP_ASYM` → the peak set is
too lopsided → **ambiguous**. `SYM/ASYM` is read off the outer (extreme-velocity) pair.
For the **amp-comparable gate** of triplet/NBN/BNB/quadruplet (≥3 peaks) the ratio
is `strongest / second-weakest`, so a single faint peak does not veto the class.

## Ordering note
Within each n, the **grouping/base patterns are tried before the pure-ratio
n-width gate** (n-width is the fallback), so a both-narrow doublet is not stolen
by 3width/4width.

## Per-n decision tables (first match wins; else `ambiguous`)

**n = 2**
| condition | class |
|---|---|
| a strong broad present | `broad-dominated` |
| 1 core + 1 faint broad wing | `singlet+wing` |
| 2 comparable-amp, similar-width, resolved | `doublet` (SYM/ASYM) |
| 2 comparable-amp, distinct-width | `2width` |

**n = 3** (base patterns first, then n-width)
| condition | class |
|---|---|
| a strong broad present | `broad-dominated` |
| a similar+resolved core pair (doublet) + third is faint broad wing | `doublet+wing` |
| 3 cores, all similar, amp-comparable, outer resolved | `triplet` |
| 3 cores, outer pair similar & middle width distinct, amp-comparable | `NBN` (middle wider) / `BNB` (middle narrower) |
| all 3 widths mutually distinct | `3width` |

**n = 4** (2pairs, then base patterns, then n-width)
| condition | class |
|---|---|
| a strong broad present | `broad-dominated` |
| 2 core+wing groups, two resolved velocity groups (cores resolved) | `2pairs` |
| exactly 1 faint broad wing + other 3 form a base | `triplet+wing` / `NBN+wing` / `BNB+wing` / `3width+wing` |
| 4 cores, all similar, amp-comparable, outer resolved | `quadruplet` (SYM/ASYM) |
| all 4 widths mutually distinct | `4width` |

**2pairs separation:** split the 4 components at their largest velocity gap into two
pairs; require each pair = one core + one wing, and the two groups **resolved** across
the gap (`resolved` on the innermost members). Otherwise it is not `2pairs`.

## Notes
- `n-width` (3width/4width) is checked over the width sequence and admits a *comparable*
  broad layer; a *faint* broad component instead becomes a `+wing` suffix.
- Every unmatched configuration (odd-one-out on an edge, chain-similarity, ≥2
  non-grouped broads, unresolved blends) → **`ambiguous`**.
- Outputs: `KIN_CLASS`, `N_KIN`, `N_WING`, `WING_KIND`, `SYM`, `BROAD_DOM`, `V_REF`,
  `CNT_N/CNT_M/CNT_B`, `AMP_LINE`.

## Thresholds (reuse current values)
`CLASS_SIGMA_FLOOR=30`, `BROAD_ABS=200`, `SIM=2.0`, `NARROW=90`, `SEP=1.0`,
`WING_WEAK=1.5`, `AMP_SYM=2.0`, `AMP_ASYM=4.0`.
