# Decisions log

The record of how this repository's code got to where it is: the problem
statements and results of the dissertation replication (Part 2), what
happened when, and the decisions behind the code, the notes and the
manuscript. Part 3's results live in
[`stiff-features.md`](stiff-features.md), not here, and from demo#42 on the
choices are recorded where they were made: the curved feature and the
comparators in the notes (§5.6, §5.7, §2.1), the manuscript's in
[`paper/README.md`](../paper/README.md) (*Pre-submission decisions*).

Carried from `docs/demo-outline.md` in
[`bradleypmartin/20260930-zd-ai-pdes-demo`](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo),
the talk repository Parts 2 and 3 were built in; the demo tag
`part3-pre-split` holds the original, talk outline included. Part 1 (the
Navier–Stokes half of the talk), the slide list and the talk schedule stayed
there, and so did the decisions that only concern the deck.

## Part 2: the dissertation replication

### 1-D problem

Two-way wave equation on periodic [-1, 1) in first-order form (u = particle
velocity, f = stress): ρ u_t = f_x, f_t = ρ c² u_x. Gaussian pulse starts at
x = -0.5 moving right; heterogeneous layer on [0, w) with (c₂, ρ₂). Equispaced
4th-order FD in space, RK4 in time.

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

**Coarse clip** (`outputs/wave1d_naive_vs_aware_coarse.mp4`,
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

### 2-D problem

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

## What happened

| Date | Work |
| --- | --- |
| Sep 17 | Scaffold, papers, plan. 1-D port with the exact ray-sum reference (demo#4). 2-D in five PRs: node sets (demo#10), RBF-FD weights and sparse operators (demo#11), RK4 with analytic validation and the hyperviscosity study (demo#12), interface-aware stencils (demo#14), clips, convergence figure and one-sided resampler (demo#15). Coarse 1-D clip (demo#16). Four bugs found and fixed the same day: a ray-pruning sign error and two latent exact-solver bugs in 1-D, a driver crash in 2-D. |
| Sep 19 | Part 3 after the talk's freeze (demo#27): smooth tanh edges, spectral reference, ODE-continued seed stencils, knee experiment, notes with verified related work and a 2-D design (demo#28–demo#34, one PR). The 2-D chain, flat: smooth flat edges (demo#36), naive baseline (demo#37), elastic seeds (demo#38), seed weights and the stability question (demo#39), the δ sweep (demo#40), oblique incidence (demo#41). |
| Sep 20 | Curved edges (demo#42), the last of the 2-D chain. The standing alternative measured head to head (demo#69). The manuscript (demo#51): scaffold (demo#52), literature pass and `LITERATURE.md` (demo#53), results cache, figures and tables (demo#54), §1–§7 (demo#55–demo#60), assembly (demo#61), arXiv packaging (demo#62). |
| Sep 22 | Split out of the talk repository into this one (#1): history import (#2), package rename and `demo#N` rewrite (#3), docs (#4), manuscript link (#5). |

## Decisions log

- 2026-09-17: 1-D is a faithful port of the FD interface method in
  `FD4wave1DAC.m` (not RBF-FD in 1-D).
- 2026-09-17: 2-D targets the full elastic system, conditional on 1-D and
  2-D prep going well.
- 2026-09-17: Repo is intentionally public. Nothing in Brad's own work is
  secret; keep it in mind, don't commit PDFs or anything employer-related.
- 2026-09-17: 2-D work is an epic (demo#2) with sub-issues demo#5 domain, demo#6
  operators, demo#7 naive simulation, demo#8 interface-aware, demo#9 video; one PR each.
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
- 2026-09-19: 2-D colour maps are aqua (field) and violet (error), derived
  from the same OKLCH lightness ladder, so blue and orange mean "which
  method" everywhere. Supersedes the blue/orange maps above.
- 2026-09-19: The 1-D clip draws no exact curve and no legends; the error
  strips carry the comparison. The 2-D still shows the reference wave once, then
  each method's error map, because the two solvers' waves cannot be told
  apart by eye at 10,000 nodes; the clips keep both waves. Clips stay at
  120 dpi and the 2-D ones square.
- 2026-09-19: Part 3 (demo#27) is an exploration after the freeze and is not in
  the talk, the deck or the clips. It lives in `docs/stiff-features.md`,
  `wave1d/stiff.py`, `wave1d/spectral.py`, `LayeredMedium(edge_width=...)`
  and `scripts/wave1d_stiff.py`; sub-issues demo#28–demo#34 land in one PR with one
  commit each (nobody is merging in real time), not the second docs PR
  announced on demo#27.
- 2026-09-19: The seeds are the t = 0 profiles of solutions polynomial in
  time (Brad's bullets with ∂ₜᵏu = C, C ≠ 0), which makes the dissertation's
  jump construction the δ → 0 limit of the same chain; `mode="aware"`
  covers both, algebra for a jump and an ODE march for a smooth edge.
- 2026-09-19: References for smooth edges are pseudo-spectral (the ray sum
  needs jumps). The knee experiment uses a wider pulse (sharpness 60,
  centre −0.6) so the coarse grids resolve the pulse and only the edge is
  under test; with the dissertation pulse the interior dispersion error
  hides the edge effect below n = 400.
- 2026-09-19: Part 3 figures are committed under `docs/figures/` (525 KB);
  `outputs/` stays gitignored, including the cached spectral references.
- 2026-09-19: The 2-D proof of concept for Part 3 goes flat first (Brad):
  sub-issues demo#36–demo#42 in order, one PR each, so the principles and the
  RBF-FD stability question are settled on a case with an independent
  reference before curvature enters. demo#36: `LayeredMedium2D(edge_width=…)`
  blends λ, μ, ρ linearly in the tanh weight (so K = λ + 2μ is linear and
  c_p is not, unlike the 1-D medium, which blends c and ρ); the
  normal-incidence reference `spectral_plane_wave` therefore maps the 2-D
  profiles pointwise into the 1-D spectral solver instead of building a
  `LayeredMedium(edge_width = 2δ)`, and hands it the exact image of the 2-D
  initial state so that f follows from f = f₀ + λ/(λ+2μ) (h − h₀) exactly.
- 2026-09-19: The 2-D naive baseline (demo#37) reports both the dissertation
  pulse and a wider one (sharpness 15, centre 0.875), reads the wider one,
  and uses fixed δ columns as in 1-D rather than δ ∝ h. The resolution
  floor hides most of the edge error in v at our node counts, so the
  spurious u (exactly zero in the true solution) is the primary measure of
  edge error for the rest of the chain; the eigenvalue driver takes
  `--edge-width` and defaults to flat interfaces with it.
- 2026-09-19: The 2-D seeds (demo#38, `wave2d/seeds.py`) are anchored at the
  evaluation node in both coordinates and scaled by r_max, the 1-D lesson
  for the normal coordinate; the polynomial basis of `interface.py` keeps
  its interface origin and is untouched, since moving the origin is a
  change of basis within the same span (`shift_matrix`), so the jump path
  stays bit for bit. All 30 seeds of a stencil march as one linear ODE
  system, restarted at every node and at the edge flanks (a single solve
  without the flanks is off by 4e-10 at δ = 1e-4). The stress seeds drop
  the same three rigid-motion columns as the polynomial basis because the
  shear pair shares its stress through any edge, not just for constant
  coefficients. Hyperviscosity rows for seed stencils (impose Δ³ = 0 on
  the seed space, or not) are an eigenvalue question deferred to demo#39.
- 2026-09-19: Seed-augmented stencils (demo#39). Hyperviscosity annihilates
  the seed space (Brad's call), but on the naive 30-node footprint, not
  the 19-node interface stencil: a 19-node Δ³ row carrying 20 coupled
  constraints has half the naive damping and turns the operator unstable
  at the MATLAB γ once the seed rows fill the domain (δ ≥ h/2), which a
  plain 19-node scheme does even without seeds. The fix, a second seed
  march on 30 nodes for the Δ³ rows, is stable at every δ at the standard
  γ with the straddling rows kept; the 30-node degree-4 stencils are
  unstable across a sharp feature for the jump path too, so the elastic
  rows stay at 19 nodes and degree 3. `seed_weights` lives in
  `wave2d/seeds.py` (not `interface.py` as the issue said) to avoid a
  circular import; `interface.py` exposes `gaussian_rows` and
  `coupled_weights` for any augmenting basis, jump path bit for bit.
- 2026-09-19: Flat δ sweep (demo#40). Seed every row that sees the edge at
  every (n, δ) and let the sweep set the rule, instead of hard-coding the
  δ ≤ h/4 guess of demo#39: the crossover sits at h ≈ δ (the δ = 0.01
  panel), so the rule is "seed when δ ≤ h, naive otherwise", per edge.
  Through an edge the nodes never resolve (δ = 0.0025) the seeds are
  fourth order at every n and land at their own resolution floor, which
  is half the naive scheme's (measured by seeding through a 10⁻⁶
  contrast); why the mix of 19-node seed rows and 30-node annihilating
  Δ³ rows beats 30/4 on this pulse is left open (§5.4). The seed marches
  run on a process pool (`build_operators(workers=...)`, bit for bit the
  serial weights) and the driver caches seed operators under `outputs/`,
  which turns an hour of marches into ten minutes once and four from
  then on.
- 2026-09-19: Oblique incidence (demo#41). The pulse is the doubly periodic
  plane-wave train along a lattice direction ((1, 2), 26.6°), because a
  tilted plane pulse on the periodic square cannot avoid the band; it
  carries the background eigenvector everywhere, after the local
  eigenvector proved to inject a sub-grid traction jump that every
  scheme then measured instead of the edge. The reference is option (i),
  Fourier in x with one complex pseudo-spectral system per mode
  (`wave2d/spectral.py`), cheaper than the 4× finer seed run the issue
  suggested first and reusable for demo#42. Result: the seeds beat naive by
  1.4–1.9× and the x'-dependent seeds are essential through a sharp
  edge (without them the seeds are worse than naive), but the order is
  2.5, not 4, and that is the resolution floor of the mode-converted S
  waves (1.73× finer than the incident pulse, pre-asymptotic on these
  node sets), which the jump-aware stencils of Part 2 share (2.4 at
  oblique incidence, never measured before) and which the P train's own
  floor does not see (§5.5). Six diagnostics in the notes; the S pulse
  at normal incidence is fourth order at half the floor, so the
  u-component seeds are fine. Forecast for demo#42: measure the jump-aware
  operator's order at curved interfaces against a floor that contains
  the converted waves before judging the curved seeds.
- 2026-09-22: Part 3 and the manuscript move out of the talk repository
  into this one (#1), before `manuscript-v1` is tagged, so that each
  repository regenerates its own deliverables and the tag the paper links
  to sits next to the code. The history comes along through
  `git filter-repo` with the talk-only paths dropped, because the
  manuscript's disclosure checks its statements against the repository's
  history; `docs/split-commit-map.txt` maps the old SHAs to the new and the
  demo tag `part3-pre-split` holds the old ones. The package is renamed
  `pdes_demo` → `rbf_hyperbolic_interfaces`; the results cache's `schema`
  string keeps its `pdes-demo` prefix because it names a format. The demo's
  issue references are rewritten once as `demo#N` (both rewrites in
  `.git-blame-ignore-revs`), so a bare `#N` means this repository. The
  Part 2 core is forked, not shared: frozen at the talk state there,
  evolving here. The manuscript's disclosure text stays as it is; its
  pull requests and their reviews stay in the demo (35–77), and the README
  points at them.
