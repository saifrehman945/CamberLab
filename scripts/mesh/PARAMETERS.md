# `REGIME_MESH` Parameter Reference

This document explains every entry in `REGIME_MESH["A"]` (in
`regime_parameters.py`): what it controls, why the chosen value, and the
trade-off if you change it. Each parameter is grouped by what it physically
controls.

Only Regime A is populated today. Regimes B / C / D inherit the same key
shape — when you fill them in, mirror this structure.

---

## Wall / boundary-layer block

### `y_plus_target = 30.0`
**What it controls.** Target dimensionless wall distance of the first
cell-centre adjacent to the airfoil. It enters the mesh via
`first_cell_height(Re, y_plus_target)` which sets the smallest cell height
("h1") next to the wall.

**Why 30.** Regime A uses **wall functions** (`nutUSpaldingWallFunction` in the
template). Wall functions assume the first cell sits in the **log-law layer**,
roughly y+ ∈ [30, 300].

- y+ < 30 — first cell is in the buffer / viscous layer where the log law is
  invalid; wall function over/underestimates τ_w → wrong Cl/Cd.
- y+ > 80 — too coarse near the wall; smeared BL profile.

30 is the lower edge of the safe band → best resolution while keeping the
wall function valid.

**Trade-off.** Lower y+ → more cells in the BL (expensive but accurate).
Higher y+ → cheaper but risks losing log-law validity.

---

### `bl_growth_ratio = 1.20`
**What it controls.** Geometric ratio between successive cell heights moving
away from the wall.

**Why 1.20.** Industry-standard upper bound for external aerodynamics.

- r > 1.3 — abrupt cell-size jumps smear the BL.
- r < 1.1 — enormous cell counts to reach the farfield.

**Important caveat in `topology.py`.** This number is **not** used as the
actual progression. The code calls `solve_progression(h1, 20c, n)` to derive
the exact ratio that covers the whole 20c domain in `normal_pts - 1` cells.
The dict value is a **soft hint** — a warning fires if the derived ratio
deviates by more than 15%.

For case_0013 (Re=1.64e6, y+=30, n=79, L=20m) the derived ratio is ≈1.165,
inside the band. Raise `normal_pts` and the derived ratio drops; lower
`normal_pts` and it rises. Treat `bl_growth_ratio` as a sanity-check ceiling.

---

### `bl_layers = 22`
**What it controls.** Currently **unused by `topology.py`**. The wall-normal
direction uses a single progression across all `normal_pts - 1` cells.
`bl_layers` is metadata: how many of those cells sit inside δ_99.

**Why 22.** Wall-function convention is 15–25 cells inside δ_99. With h1 ≈
1e-5 m, the cell index where cumulative distance reaches δ_99 ≈ 0.022 m is
around 18–22.

**If you want this to actually drive the mesh** (two-region grading: BL stack
with one r, farfield with a different r), it's a v2 change to `topology.py`.

---

### `wall_treatment = "wall_function"`
**What it controls.** Documentation tag, **not consumed by the mesher**.
Read by the CFD stage to pick which boundary-condition templates to
instantiate (`nutUSpaldingWallFunction` vs `nutLowReWallFunction`).

Keeping it in the regime dict prevents a future regime from accidentally
pairing fully-resolved walls with a wall-function-targeted mesh.

---

## Surface discretisation

### `surface_points = 300`
**What it controls.** Number of cosine-clustered points used to define the
airfoil shape per side (upper and lower). Read by `02_geometry.py` to build
`aerofoil_surfaces.npz`; `topology.py` feeds these to the gmsh spline.

**Why 300.** Raw **geometric resolution** of the airfoil — how faithfully the
spline represents curvature, especially at LE where cosine clustering puts
~30% of points in the first 5% of chord. 300/side ≫ the chord_pts the mesh
actually uses (160), so the spline is over-sampled — the mesh never sees
surface kinks from too few input points.

**Trade-off.**

- < 150 — LE radius under-resolved for thicker NACAs (t > 0.18).
- > 500 — wasted; spline becomes numerically unstable with densely clustered
  control points.

---

## Blunt trailing edge

The NACA-4 thickness equation gives a tiny but non-zero half-thickness at
x=1 (~0.0021·t per side). Forcing it to zero — the legacy "sharp closed TE"
— makes the airfoil splines converge tangentially at the TE, which is the
classic *quasi-sharp* tip that produces sliver cells under any structured
mesher. The blunt-TE topology truncates the airfoil and treats the resulting
flat back as a wall, so the wake mesh wraps around an honest corner instead
of sneaking into a near-singular tip.

### `te_chord_fraction = 0.97`
**What it controls.** Where the airfoil is truncated, in chord units. The
NACA-4 half-thickness at that x becomes the TE half-thickness `h_te`. The
TE column (TOP_TE, BOT_TE) is anchored at `x = te_chord_fraction·chord`, so
the wall-normal seams `te_nu`/`te_nl` remain vertical and the wake-centreline
edge `wake_trans` starts at `TE_MID = (te_chord_fraction·chord, 0)`.

**Why 0.97.** For NACA0012 this gives `h_te ≈ 0.0017c` per side (3.3‰ of
chord total blunt-back height) — small enough to be physically negligible
(production TE thickness is often 0.5–1% of chord on real airfoils) yet
large enough that the blunt-back wall is meshed by `te_blunt_pts` honest
cells with a meaningful aspect ratio. Scales with airfoil thickness: NACA0024
gets ~0.0033c, NACA0008 gets ~0.0011c.

**Trade-off.**

- 1.0 — sharp closed TE; **forbidden** by `topology.py` because the topology
  degenerates (TE_UP = TE_MID = TE_LO).
- 0.99 — barely-blunt; `h_te` so small that `r_blunt` is steep and the
  blunt-back cells are almost-degenerate slivers.
- 0.97 — current sweet spot; matches NASA TMR convention for NACA0012.
- 0.95 — noticeably truncated airfoil (`h_te ≈ 0.005c` for NACA0012);
  detectable Cl/Cd shift vs the closed-TE reference.
- < 0.93 — measurable lift loss; only justified if you're modelling a real
  airfoil with a fabricated blunt back.

**Validation.** If you change this, re-check your NACA0012 probe Cl/Cd
against XFOIL or the NASA TMR reference. Phase-1 validation tolerances in
CLAUDE.md §13 (ΔCl ≤ 5%, ΔCd ≤ 10%) must still hold.

---

### `te_blunt_pts = 10`
**What it controls.** Nodes per blunt-back half: 10 on `te_blunt_up`
(TE_UP → TE_MID) and 10 on `te_blunt_lo` (TE_MID → TE_LO), giving 9 cells
per half (18 cells across the full blunt back). The blunt-back curve is
graded with `r_blunt` so the first cell at the airfoil corner (TE_UP/TE_LO)
matches `te_nu`/`te_nl`'s first cell at that corner — smooth cell-size
transition through the intermediate point on UTW/LTW's compound west side.

**Cascading effect on the seam edges.** Because UTW/LTW's west side is now
*compound* (`te_nu` + `te_blunt_*`, total `normal_pts + te_blunt_pts − 1`
nodes), the opposing east side `t_seam_up`/`t_seam_lo` and the outlet edges
`c_outU`/`c_outL` get the same node count. Their progression `r_seam` is
re-solved to span the full transverse extent (`Ht`) with that larger cell
count while keeping the first cell at the wake-centre end matched to `h1`.

**Why 10.** For NACA0012 at `te_chord_fraction = 0.97`, `h_te ≈ 0.0017c`
and `h_nu_first ≈ 5×10⁻⁵ m`. With 9 cells, `solve_progression` gives
`r_blunt ≈ 1.4` — within `solve_progression`'s safe band [1.0001, 10].

**Trade-off.**

- < 5 — too few cells; `r_blunt` exceeds 2 and corner cells become tall
  slivers.
- 10 — current balance.
- > 20 — wasted; `r_blunt` shrinks below 1.1 but the blunt back is only
  ~1% of chord, so extra cells there can't pay back their cost in the rest
  of the mesh.

**Coupling.** Doubling `te_blunt_pts` adds `2 × (te_blunt_pts − 1)` cells per
seam edge — small additive impact on total cell count but it propagates
through every wall-normal column in UMW and LMW. If you raise it, expect
total cell count to climb by ~10–20% per +5 nodes.

---

## Transfinite point counts (resolution knobs)

### `chord_pts_upper = 160`, `chord_pts_lower = 160`
**What they control.** Nodes (cells = nodes − 1) along the airfoil curves.
The same count is shared with the north edge of the same block (arc + horiz)
— the consistency constraint of the topology.

**Why 160.** 159 cells / surface = 318 cells around the airfoil. With Bump
β=0.05 on a unit chord, cell sizes span ~3×10⁻⁴ m at LE/TE to ~6×10⁻³ m
mid-chord — fine enough to resolve the suction peak.

**Trade-off.**

- < 80 — LE suction peak aliased; Cl drops a few %.
- 160 — Regime A sweet spot.
- 220+ — useful for stall (Regime B), excessive here.

**Must be equal** for the symmetric-airfoil topology; `topology.py` raises if
they differ. Cambered profiles (future) will need topology relaxation.

---

### `normal_pts = 80`
**What it controls.** Nodes (79 cells) in the wall-normal j-direction on
`le_rad`, `te_nu`, `te_nl`. The seam/outlet edges (`t_seam_up`, `t_seam_lo`,
`c_outU`, `c_outL`) carry **`normal_pts + te_blunt_pts − 1`** nodes instead,
since they sit opposite UTW/LTW's compound west side — see the blunt-TE
section above.

**Why 80.** With h1 ≈ 1e-5 m from y+=30 and a 20c total length, 80 nodes
gives derived progression ≈1.165 — comfortably inside [1.10, 1.30].

**Trade-off.**

- < 50 — derived r jumps past 1.25; cell-size jumps become visible in the
  outer BL.
- 80 — r in textbook range, cell count ~50k.
- 120+ — derived r drops below 1.10; outer field over-refined, wasted cells.

---

### `wake_pts = 220`
**What it controls.** Nodes (219 cells) along the **main** wake direction in
blocks UMW and LMW. Same count is required on `wake_main`, `top_main`,
`bot_main`. Note that this is the main-wake count only — the short
transition wake immediately downstream of TE is sized separately by
`transition_wake_pts`.

**Why 220.** With `wake_progression = 1.015` and a main-wake length of
(30 − 0.4)c = 29.6c, this gives first-main-wake cell ≈ 1.8×10⁻² m, growing
gently outward to ≈0.6 m at the outlet. The transition wake block on its
west side delivers cells of comparable size to this, so the main-wake first
cell is no longer a "first cell at TE" problem — it's the handoff from the
transition block, which by construction matches.

**Trade-off.**

- < 150 — main wake under-resolved within the first 5–10c downstream.
- 220 — current sweet spot for `wake_progression = 1.015`.
- 300+ — wasted; outer-wake cells get unnecessarily small.

---

## Wake transition block

### `transition_wake_length = 0.4`
**What it controls.** Chord-multiple length of the short structured block
sitting immediately downstream of the TE. Three new points (`T_TOP`,
`T_BOT`, `T_MID`) at `x = 1 + transition_wake_length` define the vertical
seam between the transition wake and the main wake.

**Why 0.4.** Standard production C-grid practice is 0.3–0.5c. The block
needs to be **long enough** to grade smoothly from airfoil TE spacing
(~5×10⁻⁴ m) to main-wake spacing (~1.8×10⁻² m) without an aspect-ratio
discontinuity, and **short enough** that it doesn't waste cells far
downstream.

**Trade-off.**

- < 0.2c — too short; transition progression has to be aggressive (r > 1.1)
  to span the cell-size jump → cell-size gradients become large.
- 0.3–0.5c — clean handoff.
- > 0.7c — wasted; the main wake block could cover the same span more
  cheaply.

---

### `transition_wake_pts = 80`
**What it controls.** Nodes (79 cells) inside the transition block along
the wake direction. Same count is shared by `wake_trans`, `top_trans`,
`bot_trans`.

**Why 80.** With `transition_wake_length = 0.4` and a derived progression
that matches airfoil-TE spacing at the TE end, 80 nodes give a healthy
~1.05 growth ratio across the block.

**Trade-off.** Lower counts force more aggressive grading; higher counts
just refine an already-thin region.

---

### `transition_wake_progression = None`
**What it controls.** Geometric progression ratio along the transition
block. **None** (the default) → derive automatically by solving
`h_TE_target * (rⁿ - 1) / (r - 1) = transition_wake_length`, where
`h_TE_target = le_te_cluster * chord / (chord_pts - 1)` is the estimated
Bump-endpoint cell size on the airfoil at TE. Set a float (e.g. `1.045`) to
override.

**Why auto-derive.** This is the entire point of having the transition
block — it absorbs the cell-size jump between the airfoil and the main
wake. Hard-coding a ratio defeats the purpose; the regime params can be
tuned freely and the transition block tracks automatically.

**Diagnostic logging.** `topology.py` emits a one-line summary per case:
`h_TE_target | trans_first (ratio) | trans_last | main_first (ratio)`.
The first ratio should be ≈1 (transition matches airfoil); the second ratio
should be small (smooth handoff to main wake).

---

## Transfinite grading laws

### `le_te_cluster = 0.09`
**What it controls.** gmsh **Bump** law β-coefficient on the airfoil splines.
Smaller β → tighter clustering at both endpoints (LE and TE simultaneously);
β=1 → uniform. **Also drives the auto-derivation of
`transition_wake_progression`** via the Bump-endpoint estimate.

**Why 0.09.** Endpoint cells ~11× smaller than mid-chord cells. For
chord_pts=160 on a unit chord: ~5.6×10⁻⁴ m at LE/TE vs ~6×10⁻³ m mid-chord.
The TE cell size sets the transition wake's first-cell target — too-small
forces an aggressive transition progression, too-large smears the Cp peak.

**Trade-off.**

- 0.02 — very tight LE/TE cluster (sharp Cp peak), aggressive transition
  progression required.
- 0.09 — current balance, plays cleanly with the transition wake.
- 0.20 — mild cluster; faster but smeared Cp peak.

**Coupling.** Single parameter governs both LE and TE clustering on the same
curve. Cannot be tuned independently with current topology. If we need that
later (transonic sharp TE), split the airfoil curve at mid-chord.

---

### `wake_progression = 1.015`
**What it controls.** Geometric growth ratio along the **main** wake block
(blocks UMW and LMW), applied to `wake_main`, `top_main`, `bot_main`. First
cell at the transition→main interface (`T_*` points); each downstream cell
is 1.5% longer than the previous.

**Why 1.015.** With `wake_pts = 220` over a ~29.6c main-wake length, this
gives a near-uniform main-wake spacing (first ≈ 1.8×10⁻² m, last ≈ 0.6 m).
Because the transition wake block now hands off cells of comparable size to
this, there is no longer a cell-size cliff at TE.

**Trade-off.**

- 1.005 — near-uniform wake (lots of cells far downstream where they're
  wasted).
- 1.015 — current balance.
- 1.03+ — sharper outer-wake coarsening but reopens a size mismatch at the
  transition→main interface (large jump from transition-block last cell to
  main-block first cell).

---

### `north_arc_to_horiz_ratio = 0.65`
**What it controls.** Partition of the airfoil-block's "north" edge between
the arc sub-curve (from LE_FAR to TOP_MID) and the horizontal sub-curve (from
TOP_MID to TOP_TE). The cell-count split is `arc : horiz = ratio : 1`.

**Why 0.65.** Arc spans π/2 × 20c ≈ 31.4 m arc length; horizontal spans
chord = 1 m. With ratio 0.65 the arc gets ~63 cells and the horizontal gets
~96 cells (out of 159 total). The horizontal — which sits directly above the
airfoil and matters more for force prediction — gets denser nodes than the
arc; the upstream arc, sitting 20c away from the wall, can be coarser
without harm.

**Trade-off.**

- ratio > 1 — more nodes on the arc, fewer on the horizontal; weakens
  resolution above the airfoil. **Hurts** force coefficients.
- 0.65 — current weighting; favours the horizontal slightly.
- ratio < 0.4 — most nodes on horizontal, very coarse arc; risks introducing
  visible cell-size gradient at TOP_MID/BOT_MID corner.

---

## Farfield extents (chord multiples)

### `upstream_radius = 20.0`, `transverse_extent = 20.0`, `downstream_length = 30.0`
**What they control.** Domain size. Upstream and transverse **must be equal**
(enforced in `farfield_points`) so the upstream cap is a clean semicircle.

**Why 20c / 20c / 30c.** CLAUDE.md §10 spec, derived from AIAA Drag
Prediction Workshop and NASA TMR practice for external aerodynamics:

- 20c upstream — pressure perturbation decays as 1/r² for lift and 1/r³ for
  thickness; by 20c residual is < 0.5% of free-stream q.
- 30c downstream — wake decayed enough that the outflow BC doesn't reflect
  spurious pressure.
- 20c transverse — same rationale as upstream.

**Trade-off.**

- 50% smaller — halves cell count, introduces ~2–3% blockage error in Cd.
- 2× larger — negligible accuracy improvement, significant cost.

---

## 2D-quasi-3D extrusion

### `spanwise_thickness = 0.05`, `spanwise_layers = 1`
**What they control.** Even with OpenFOAM's `empty` front/back patches, the
mesh is 3D — one cell thick in z. `spanwise_thickness` is the absolute
thickness (chord units → 0.05 m); `spanwise_layers` is the number of cells
in z.

**Why 0.05 and 1.**

- Thickness must be **small enough** that the spanwise face doesn't dominate
  cell shape (≤ 0.1 c typical).
- **Large enough** that floating-point rounding doesn't degenerate the cells
  (≥ 1e-3 m).
- 0.05 is the convention used by OpenFOAM's own airFoil2D tutorial.
- One layer is mandatory for quasi-2D RANS; more layers bloat cell count
  without changing the solution.

---

## Topology dispatch

### `topology = "c_grid_6block"`
**What it controls.** Selects which builder in `topology.py` to dispatch to.
Today only one builder exists — the C+H 6-block transfinite mesh with a
transition-wake block downstream of the TE and a blunt trailing-edge wall.
This is the hook for plugging in a different topology for Regime B (low-y+
near-stall) or Regime C (transitional) without changing the orchestrator.

---

## Quality gates

### `non_orthogonality_max = 70.0`
**What it controls.** Hard reject threshold in `validate_quality`. Any face
with non-orthogonality ≥ this is treated as a fatal mesh failure.

**Why 70 — and the current issue.** CLAUDE.md §7 quotes 70° as "acceptable."
But OpenFOAM 12's `checkMesh` considers up to ~75° fine, and with
`nNonOrthogonalCorrectors ≥ 1` (CLAUDE.md §5) it solves cleanly to ~85°. The
smoke test produced 89.4° on 0.2% of faces — a known C-grid LE artefact. 70°
is stricter than necessary; 75–80° would still produce solver-acceptable
meshes.

---

### `skewness_max = 4.0`
**What it controls.** Hard reject on cell skewness.

**Why 4.** OpenFOAM convention; structured grids typically run 1–2; > 4
harms convergence.

---

### `aspect_ratio_max = 5000.0`
**What it controls.** Hard reject on per-cell length/width ratio.

**Why 5000.** Wall-function meshes have h1 ≈ 1e-5 m and chord cells
~6×10⁻³ m → aspect ratios ~600 mid-chord and ~17× higher at the TE wake
transition. OpenFOAM's own warning fires at 1000 but it's only a warning;
correct numerics depend more on aspect-ratio *gradients* than absolute
values.

---

### `min_hex_fraction = 0.999`
**What it controls.** Reject if any block failed to recombine into hexes
(gmsh fell back to triangles). A "did the transfinite work at all?" check.

---

## Advisory cell-count band

### `target_cells_min = 40_000`, `target_cells_max = 200_000`
**What they control.** Soft warning thresholds (no rejection). Logs a
warning if realised cell count falls outside.

**Why 40k–200k.** Current defaults produce ~47k cells. CLAUDE.md §10
specifies 80k–200k for Regime A; lower bound is at 40k so the current config
doesn't trigger a spurious warning. Treat 80k as a target rather than a
floor — raising counts is cheap if force coefficients turn out
under-converged.

---

## Cross-parameter coupling — what changes together

When you tune one knob, others may need to follow:

| Change | Likely also-need-to-change |
|---|---|
| `y_plus_target` ↓ (e.g., 30 → 1 for low-Re regime) | `wall_treatment` → `"low_re"`; OpenFOAM template directory swap |
| `chord_pts_upper/lower` ↑ | `le_te_cluster` ↑ (less aggressive cluster); transition wake re-derives automatically |
| `le_te_cluster` ↑ | airfoil TE cell grows → transition wake progression eases automatically (auto-derived) |
| `normal_pts` ↑↑ | derived progression may dip below `bl_growth_ratio`; warning will fire |
| `wake_pts` ↑ | `wake_progression` ↓ to keep main-wake first cell aligned with transition's last cell |
| `transition_wake_length` ↓ | `transition_wake_pts` ↓ proportionally to avoid over-refining a short region; auto-derived ratio sharpens |
| `transition_wake_progression` set explicitly | be aware you've broken the airfoil-TE matching invariant — check the per-case `wake handoff` log line |
| `te_chord_fraction` ↓ (blunter TE) | re-validate NACA0012 Cl/Cd vs the closed-TE reference; `r_blunt` eases (good); `h_te_target` shifts because the airfoil curve is shorter |
| `te_blunt_pts` ↑ | seam/outlet edges (`t_seam_*`, `c_out*`) inherit the larger node count; total cell count climbs noticeably (every wall-normal column in UMW/LMW gains cells) |
| `upstream_radius` ↑ | `normal_pts` ↑ to keep derived progression in band |
| `non_orthogonality_max` ↑ (relax gate) | CFD `nNonOrthogonalCorrectors` ↑ to handle skewed faces |

---

## Quick reference: which parameters drive which outcomes

| Outcome you care about | Primary knobs |
|---|---|
| Cl accuracy near LE | `le_te_cluster`, `chord_pts_*` |
| Cp suction-peak resolution | `le_te_cluster`, `chord_pts_*` |
| Cd accuracy | `wake_pts`, `wake_progression`, `transition_wake_*`, `chord_pts_*` (TE end) |
| Smooth TE handoff (no skewed corner cells) | `transition_wake_length`, `transition_wake_pts`, `le_te_cluster` |
| Avoiding sliver cells at the TE tip | `te_chord_fraction`, `te_blunt_pts` |
| Convergence robustness | `non_orthogonality_max`, `bl_growth_ratio`, `aspect_ratio_max` |
| Total cell count | `chord_pts_*`, `normal_pts`, `wake_pts`, `transition_wake_pts`, `te_blunt_pts` |
| Farfield-influence error | `upstream_radius`, `transverse_extent`, `downstream_length` |
| BL profile fidelity | `y_plus_target`, `normal_pts`, derived progression ratio |
