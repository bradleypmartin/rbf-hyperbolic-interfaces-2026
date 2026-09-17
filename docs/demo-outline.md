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

### Slide sketch (≈8 slides)

1. The problem in one slide (NS equations; Clay's four alternatives).
2. What was announced, when, by whom; the three artifacts (paper, Lean, post).
3. The theorem, in words.
4. The physical picture (collapsing vortex + oscillatory pulses).
5. How it was produced (agents, time; flagged as reported).
6. The parallel story (Alpöge–Buckmaster; Anthropic models; the dispute).
7. Verification status (Lean, Clay, community).
8. Questions for discussion.

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
  scattered node set with curved interfaces (naive RBF-FD vs interface-aware).
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

### 2-D results so far (2026-09-17, naive RBF-FD everywhere)

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

## Timeline (13 days from 2026-09-17)

| Dates | Work |
| --- | --- |
| Sep 17 | Environment, papers, plan (done) |
| Sep 18–19 | 1-D: Fornberg weights, interface stencils, double-cross, RK4, tests |
| Sep 20–22 | 1-D two-panel video; read NS manuscript §1–3, Clay statement, press |
| Sep 23–26 | 2-D RBF-FD prep (node set, periodic kNN, weights, sparse ops, hyperviscosity), elastic time stepping |
| Sep 27–28 | 2-D interfaces + two-panel video if on track; slide deck PDF for both halves |
| Sep 29 | Rehearsal; freeze the repo |
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
