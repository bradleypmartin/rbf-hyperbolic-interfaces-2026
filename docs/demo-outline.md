# Demo outline — 2026-09-30, 30 minutes

Audience: Ziff Davis coworkers from several parts of the org. Bright tech
workers; assume no background in numerical PDEs or PDEs at all. This is a
high-level share-out about mathematical simulation and what a domain expert
experiences collaborating with current AI. **No live coding**: everything is
prepared in advance (two videos + a short slide deck PDF). Thesis of the talk:
"AI in research mathematics" currently means two very different things, and it
is worth seeing both.

1. Autonomous, frontier-scale proof generation (OpenAI / Navier–Stokes).
2. Collaborative, verifiable, human-in-the-loop computational work (Claude
   Code + my dissertation methods).

## Part 1 — Navier–Stokes (≈15 min)

### Verified facts (source in parentheses; PDFs in `papers/`)

- 2026-09-08: OpenAI posts *On the Navier–Stokes Millennium Prize Problem*
  linking a 166-page manuscript, *Finite Time Blowup for Navier–Stokes*, and a
  Lean 4 repository. (Blog URL and PDF; PDF creation date 2026-09-08.)
- Abstract, verbatim: "For every positive viscosity, we construct a solution
  of the three-dimensional incompressible Navier–Stokes equations that starts
  from rest and develops unbounded velocity in finite time while maintaining
  uniformly bounded kinetic energy." (manuscript p. 1)
- Theorem 1.1: for every ν > 0 there is a smooth force f, compactly supported
  in space and time, and smooth (u, p) on R³ × [0, 1) with u(·, 0) = 0,
  support in a fixed compact set K, sup‖u(t)‖_L² < ∞ and
  limsup_{t↑1} ‖u(t)‖_L∞ = ∞. Hence no global smooth finite-energy solution
  with that force and datum. This is alternative **(C)** of Fefferman's Clay
  statement; compact support gives the torus case, alternative **(D)**
  (Corollary 10.6). (manuscript p. 1; Clay statement p. 2)
- Mechanism (manuscript §2): an axisymmetric, self-similar collapsing vortex
  (radial width shrinks faster than axial length; inward spiral + axial
  outflow spins it up). The background alone leaves a momentum residual that
  blows up. Spatially oscillatory pulses are added whose nonlinear momentum
  flux cancels the singular part of that residual, and further corrections
  make the leftover force extend smoothly through t = 1.
- Companion result: unforced Euler blowup from smooth compactly supported
  divergence-free data on R³. (Lean repo README; Euler PDF linked there.)
- Lean 4 formalization: `openai/NavierStokesAndEuler`, Apache-2.0, created
  2026-09-08, Lean 4.34.0-rc2 + Mathlib, with instructions for independent
  checking via "Comparator". ~1.9k stars on 2026-09-17. (GitHub API)
- Prior work the paper positions itself against (manuscript §1.1): Leray 1934
  weak solutions; Caffarelli–Kohn–Nirenberg partial regularity; Escauriaza–
  Seregin–Šverák; Tao's blowup for an *averaged* NS; Buckmaster–Vicol
  nonuniqueness of weak solutions; Albritton–Brué–Colombo nonunique
  Leray–Hopf solutions with forcing; Córdoba–Martínez-Zoroa forced Euler
  blowup and the hypodissipative NS extension with Zheng.

### Reported in press

Verified against CNBC, Fortune, and The Week on 2026-09-17; see
`docs/navier-stokes-notes.md` for the attributed details and what remains
unread (Nature, New Scientist, Axios, the Alpöge–Buckmaster paper).

### Discussion prompts

- **Letter vs. spirit.** Fefferman's (C)/(D) explicitly allow a smooth force.
  The result is squarely within the official statement, but the intuition
  most people have about "does turbulence blow up" is about unforced flow.
  Where does that leave the Millennium problem?
- **Machine-checked ≠ accepted.** Lean certifies the formal theorem. Whether
  the formal statement is the Clay statement, and whether the formalization
  is faithful, is still a human review step. "Independent judgment pending"
  is the honest status.
- **Credit and conduct** when labs race on rumors of each other's results.
- **Compute concentration.** Whatever the exact numbers, this scale of search
  is not available to an academic group. What does that do to the field?
- **What is still human here?** Problem selection, the physical picture in
  §2, judging significance, deciding what to formalize.

### Slides (`slides/talk.tex`; culled 2026-09-19 for #18, 19 pages, no backup)

Budget: title, a framing slide, two section slides, seven content slides per
part, and a links slide. About two minutes a slide. Backup slides were dropped;
the equations, the stability figure, and the numbers tables live in the repo.

1. Title. 2. Why this talk (thesis, roadmap).
3. Fluids, Newton, and a question from 1934 (no equations; flow diagram of
   the two outcomes; "a counterexample may use a smooth force" is the hinge).
4. What happened, in two weeks (TikZ timeline, Aug 15 – Sep 17, colour by
   party).
5. The claim, and the picture behind it (abstract verbatim; vortex schematic
   after Fig. 1; what it is not).
6. How it was produced, in OpenAI's own account (number tiles: agents, hours,
   tokens, Lean, cost; press discrepancy in the source line).
7. The parallel story, and the dispute (Alpöge–Buckmaster; two columns, both
   primary sources).
8. What "machine-checked" does and does not mean (Lean vs human checks).
9. Reactions, and questions to argue about (four quotes, three prompts).
10–16. Part 2 (section; my corner + problem in one picture; 1-D ringing +
   clip; 2-D nodes; 2-D snapshot + clips; 1-D and 2-D convergence on one
   slide, left panels cropped by `build.sh`; the collaboration with number
   tiles; comparison table).
17. For the curious (links; on screen during questions).

Speaker script with timings: `slides/notes.md`.

## Part 2 — Working with Claude on my dissertation (≈15 min)

### Narrative

"In 2016 I wrote this in MATLAB. Today I don't have a MATLAB license, and
the code is nine years old. With Claude Code I re-derived the method, ported
it to Python, and verified it against analytic results in an afternoon."

### Deliverables

- `outputs/wave1d_naive_vs_aware.mp4`: two panels, same discretization.
  Left: naive FD straight across the interfaces. Right: interface-aware FD.
  The naive panel should visibly ring / get the reflection wrong.
- `outputs/wave2d_naive_vs_aware.mp4`: same idea for 2-D elastic waves on a
  scattered node set (naive RBF-FD vs interface-aware), with error maps under
  the two |v| panels. `outputs/wave2d_naive_vs_aware_curved.mp4` is the
  curved-interface variant (`--amplitude 0.02`).
- `slides/`: brief deck as PDF covering both halves. Tooling: `tectonic` is
  installed (Beamer works offline); Marp via `npx` is the alternative (Google Chrome
  is installed, so PDF export works).

### 1-D problem (must-have)

Two-way wave equation on periodic [-1, 1) in first-order form (u = particle
velocity, f = stress): ρ u_t = f_x, f_t = ρ c² u_x. Gaussian pulse starts at
x = -0.5 moving right; heterogeneous layer on [0, w) with (c₂, ρ₂). Equispaced
4th-order FD in space, RK4 in time.

Demo beats:

1. Baseline: naive FD straight across the interface. Show the spurious
   oscillation / wrong reflection amplitude.
2. Interface-aware stencils: piecewise polynomials that satisfy the PDE's
   continuity conditions across the jump; weights come from solving a small
   dense system per affected node.
3. The thin-layer "double-cross": one stencil spanning both sides of a layer
   thinner than the stencil. Compare naive vs. treated as w shrinks below h.
4. Verification: reflected/transmitted amplitudes vs. the analytic impedance
   formulas (Z = ρc); convergence order plot.
5. Animation.

### 1-D results (2026-09-17, `scripts/wave1d_convergence.py`)

Relative l2 error in stress at t = 1 against the exact ray-sum solution,
c: 1 -> 2, rho: 1 -> 1, RK4 at CFL 0.4. Matches dissertation Fig. 2-8:
naive is first order, interface-aware is fourth order, also when the layer
(width 0.01) is thinner than the stencil.

| nodes | naive, layer 0.5 | aware, layer 0.5 | naive, layer 0.01 | aware, layer 0.01 |
| --- | --- | --- | --- | --- |
| 200 | 1.2e-1 | 6.4e-2 | 1.3e-1 | 7.5e-2 |
| 400 | 5.3e-2 | 4.5e-3 | 1.4e-2 | 5.5e-3 |
| 800 | 2.6e-2 | 2.9e-4 | 6.1e-3 | 3.5e-4 |
| 1600 | 1.3e-2 | 1.8e-5 | 3.0e-3 | 2.2e-5 |

At 400 nodes (the dissertation's resolution) the naive solution's ringing is
visible but not dramatic to a lay eye, so the video carries an error strip
under each panel. Running longer does not widen the gap much (both errors
grow; ratio stays ~5-10x), so the clip stops at t = 1.25, just before the
transmitted pulse wraps around the periodic domain.

**Coarse clip for the talk** (`outputs/wave1d_naive_vs_aware_coarse.mp4`,
Brad's suggestion: ~100 points with a 2x layer is where the naive ringing is
obvious). 100 nodes and a wider pulse, `--sharpness 150` (about 4 nodes
across; the default 600 is only 2 nodes wide at this spacing and dispersion
then swamps both panels). Brad's call: plain curves, no node markers. At
t = 1 the naive
solution trails a sawtooth of 6% of the pulse height behind the reflected
pulse and through the layer (relative error 10%); the interface-aware
solution has 0.6% ringing and 3.3% error, i.e. it sits on the exact curve.
At 200 nodes with the default pulse the ringing is finer (5%) but still
clear; at 100 nodes with the default pulse both panels are underresolved.

### 2-D problem (stretch)

RBF-FD on a scattered, repulsion-relaxed node set in the doubly periodic unit
square, with curved interfaces. Pieces: node generation, periodic kNN by
tiling + `cKDTree`, per-stencil dense solves (~30 nodes + polynomial terms,
so ~50×50), sparse differentiation matrices, hyperviscosity, RK4.

Scope (decided 2026-09-17): the full 5-field **elastic** system (u, v,
s1, s2, s3), as in the MATLAB and the dissertation, conditional on the 1-D
work and the 2-D prep (node set, kNN, sparse operators) going well. Brad's
view: the understanding/implementation hurdles are equivalent between
acoustic and elastic once we align on one; acoustic is only cheaper to run.
RBF flavor is Claude's call, to be made after reading the interface
treatment in the 2017 JCP preprint: default to polyharmonic splines +
polynomials (shape-parameter free) unless the interface treatment leans on
the IMQ/GA formulation used in 2016.

### 2-D results (2026-09-17)

Node sets: fixed hex rows straddling each interface plus a repulsion-relaxed
field (`scripts/wave2d_nodes.py`). Operators: 30-node Gaussian RBF-FD
stencils with degree-4 polynomials, `eps = 0.4 / d_3`, `Delta^3`
hyperviscosity with the MATLAB amplitude `gamma = 2.4e-11 (h/0.02)^5`, RK4 at
CFL 0.5 capped by the hyperviscosity spectrum.

- Spectrum (`scripts/wave2d_eigenvalues.py`, 900 nodes): max Re(lambda) goes
  from +26 without hyperviscosity to +0.01 with it; 10x gamma pushes the
  damped modes past RK4's real-axis limit. In a uniform medium the MATLAB
  gamma is exactly the smallest value that puts every eigenvalue in the
  left half-plane (`scripts/wave2d_hyperviscosity.py`).
- Uniform medium, plane P-wave (sigma = 10), t = 0.2, relative error in v
  against the exact solution: 4.2e-3 (1600 nodes), 1.9e-3 (2500), 9.9e-4
  (3600), 5.5e-4 (4900); rates 3.6-3.7. Errors are insensitive to gamma
  between 0.25x and 4x the MATLAB value; larger gamma only shrinks the time
  step.
- Flat two-interface problem (dissertation §3.4.1, sigma = 23), t = 0.3,
  relative error in v vs the ray-sum reference. "Floor" is the same pulse
  in a uniform medium, i.e. pure resolution error with no interface:

  | nodes | naive | interface-aware | floor (no interface) |
  | --- | --- | --- | --- |
  | 2500 | 1.1e-1 | 9.9e-2 | 8.2e-2 |
  | 4900 | 6.9e-2 | 3.4e-2 | 3.3e-2 |
  | 10000 | 3.1e-2 | 9.6e-3 | 1.1e-2 |
  | 19600 | 1.7e-2 | 2.7e-3 | 3.0e-3 |

  Naive converges at 1.5-2.2 (the "FD4" behaviour of Fig. 3-5); the
  interface-aware stencils (19 nodes, degree 3, coupled piecewise bases,
  locally flat interface, MATLAB's 4h band) remove the interface error
  entirely at these resolutions: the error equals the resolution floor,
  which itself converges at 3.2-3.5 towards 4th order. At 19600 nodes the
  aware solution is 6x more accurate than naive.
- Curved interfaces (§3.4.2 geometry, amplitude 0.02, constant Lamé
  parameters in the band), t = 0.3, relative error in v against a
  40000-node interface-aware run resampled onto each node set
  (`scripts/wave2d_convergence.py`):

  | nodes | naive | interface-aware |
  | --- | --- | --- |
  | 2500 | 1.2e-1 | 9.8e-2 |
  | 4900 | 6.5e-2 | 3.3e-2 |
  | 10000 | 3.0e-2 | 8.9e-3 |

  Same picture as the flat case, so the locally flat interface
  approximation costs nothing at these resolutions (JCP Fig. 10 puts the
  crossover near 40000 nodes). Both runs are stable to t = 1.5 at 2500
  nodes with the default hyperviscosity.
- Measuring errors against a resampled reference: the interpolant uses
  one-sided stencils (source nodes from the target's own material only,
  `wave2d/resample.py`). v has a kink at an interface, and a stencil that
  mixes both sides adds its own error there: with the pulse centred on
  y = 0.5 the two-sided interpolant's error in the interface band is 40x
  the one-sided one's (3.0e-3 vs 7.6e-5 from 10000 to 2500 nodes), while
  away from the band the two agree. One-sided, the reference is usable at
  any time, not only when the pulses are clear of the interfaces. For the
  record, at the dissertation's t = 0.3 only 0.6% of ||v||^2 lies within
  4h of an interface (h = 0.01); at t = 0.15 it is 75%, and the reflections
  bouncing inside the band keep it at a few percent later on.
- Video (`scripts/wave2d_demo.py`, 10000 nodes, 250 frames to t = 0.5,
  36 s end to end flat, 67 s curved including the 40000-node reference).
  Relative error in v, flat / curved:

  | t | naive | aware |
  | --- | --- | --- |
  | 0.2 | 2.4e-2 / 2.3e-2 | 7.6e-3 / 6.9e-3 |
  | 0.3 | 3.1e-2 / 3.0e-2 | 9.5e-3 / 8.9e-3 |
  | 0.4 | 5.0e-2 / 4.6e-2 | 2.1e-2 / 1.5e-2 |
  | 0.5 | 3.2e-2 / 3.0e-2 | 1.3e-2 / 1.2e-2 |

  The clip ends at t = 0.5, after the pulse has split at the upper
  interface, crossed the band, split at the lower one and the in-band
  reflection has split again at the upper interface. 2500 nodes would
  render in a few seconds but both methods sit at the resolution floor
  there (11% vs 10%), so the panels would look alike; at 10000 the naive
  error is 3x the aware one for the whole clip and the error maps show it.

## Timeline (13 days from 2026-09-17)

| Dates | Work |
| --- | --- |
| Sep 17 | Environment, papers, plan (done) |
| Sep 18–19 | 1-D: Fornberg weights, interface stencils, double-cross, RK4, tests |
| Sep 20–22 | 1-D two-panel video; read NS manuscript §1–3, Clay statement, press |
| Sep 23–26 | 2-D RBF-FD prep (node set, periodic kNN, weights, sparse ops, hyperviscosity), elastic time stepping |
| Sep 17 (actual) | 1-D, 2-D, Navier–Stokes reading, and the first full deck all landed on day one |
| Sep 18–26 | Slack: read the still-unread sources (Nature controversy section, Economist, NYT); follow Clay and OpenAI for updates; polish wording |
| Sep 27–28 | Final deck pass; regenerate clips; check every quote against `docs/navier-stokes-notes.md` |
| Sep 29 | Rehearsal with `slides/notes.md`; freeze the repo |
| Sep 30 | Demo |

## Decisions log

- 2026-09-17: 1-D is a faithful port of the FD interface method in
  `FD4wave1DAC.m` (not RBF-FD in 1-D).
- 2026-09-17: 2-D targets the full elastic system, conditional on 1-D and
  2-D prep going well.
- 2026-09-17: Delivery is two prepared two-panel videos (naive vs
  interface-aware, same resolution) plus a short slide deck PDF. No live
  coding. Audience assumed to have no PDE background.
- 2026-09-17: Repo is intentionally public. Nothing in Brad's own work is
  secret; keep it in mind, don't commit PDFs or anything employer-related.
- 2026-09-17: 2-D work is an epic (#2) with sub-issues #5 domain, #6
  operators, #7 naive simulation, #8 interface-aware, #9 video; one PR each.
- 2026-09-17: RBF flavour is **Gaussian with a relative shape parameter**
  (`eps = 0.4 / d`), Brad's call and what the MATLAB uses; not PHS.
- 2026-09-17: 2-D goes naive-first: plain RBF-FD everywhere plus
  hyperviscosity, validated against the flat-interface analytic solution,
  before any interface-aware stencils. No Cartesian/FD hybrid far field for
  now ("all RBF"); it is a runtime optimisation we can add if needed.
- 2026-09-17: Node sets keep fixed hex-staggered rows straddling every
  interface orthogonally (Brad: empirically the key to 2-D stability).
- 2026-09-17: Errors against a resampled reference, and all colour-map
  rendering, use one-sided interpolation stencils that never cross an
  interface (Brad: an interpolant straddling the interface would need the
  interface-aware treatment itself, or the comparison must stay well away
  from the interfaces). Two-sided stencils are kept only for the test that
  demonstrates the problem.
- 2026-09-17: The 2-D video defaults to 10000 nodes (2 s per run) rather
  than the 2500 of the issue text: at 2500 both methods are at the
  resolution floor and the panels look identical. Flat interfaces are the
  default clip because the error maps are then against the exact solution;
  the curved clip is behind `--amplitude 0.02` with a 4x finer reference.
- 2026-09-17: 2-D colour maps use two single-hue sequential ramps from the
  1-D palette, blue for |v| and orange for error; the orange steps were
  derived from the blue ramp's OKLCH lightness ladder so the two read alike.
- 2026-09-17: Slides are **Beamer (metropolis) built with tectonic**, not
  Marp: fully offline, math and figures are trivial, and the PDF is the
  deliverable anyway. Avenir Next on macOS with a Latin Modern fallback.
- 2026-09-17: `slides/figures/*.png` (2.5 MB), `slides/videos/*.mp4`
  (~10 MB), and `slides/talk.pdf` are all committed so the talk is
  self-contained from a fresh clone. Brad's call: commit `talk.pdf` on every
  change; re-commit a clip only when its content changed.
- 2026-09-17: The dispute gets one slide, two columns, one primary source
  each (Buckmaster's statement; OpenAI's post), no adjudication. Cost
  figures come from Science (Chen: "millions of dollars"); the New Scientist
  "$15 million" is not used because the primary source gives no cost.
- 2026-09-17: Audience framing for Part 2: no equations on the main path,
  one TikZ picture of a stencil straddling a corner, results as "halve the
  error" vs "divide by 16". Beginner links (3Blue1Brown, Khan Academy,
  Wikipedia) on the last slide.
