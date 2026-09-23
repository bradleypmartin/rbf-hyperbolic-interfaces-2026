# CLAUDE.md — rbf-hyperbolic-interfaces-2026

## What this is

Brad's research repo for the interface-aware wave solvers of his 2016 CU
Boulder dissertation and what grew out of them. Public, MIT (code) and
CC BY 4.0 (`paper/`).

- **Part 2, the replication.** Brad and Claude re-derived and ported to Python
  the dissertation's solvers: 1-D fourth-order finite differences through a
  layer (`wave1d/`), and 2-D elastic RBF-FD on scattered nodes with curved
  interfaces (`wave2d/`), each verified against a reference solution. Built
  2026-09-17. Problem statements and results tables:
  `docs/decisions-log.md`.
- **Part 3, stiff-but-smooth edges.** Brad's never-published idea, built from
  2026-09-19: a stencil crossing a material edge too steep for the grid is
  built from ODE-continued "seeds" instead of monomials. Canonical notes:
  `docs/stiff-features.md` (derivation §1 and §4, results §2 and §5).
- **The manuscript.** `paper/`, the arXiv write-up of Part 3; `LITERATURE.md`
  is its novelty ledger. Packaged, unsubmitted (endorsement pending, #13).

"Part 2" and "Part 3" are the numbering of the talk this work grew up in
(Part 1, Navier–Stokes, stayed there; see *Provenance*). Epic #1 moved it
here on 2026-09-22 and this side of it is done; what remains of the epic is
the demo's own pruning (demo#78, demo#79). The manuscript's submission is
tracked in #13.

Papers: `papers/README.md`. **Before reading a PDF, check
`docs/paper-index.md`** for the page ranges that matter and read only those
(`pdftotext -f A -l B -layout <pdf> -`).

### Part 3 in brief

The seeds are the t = 0 profiles of solutions polynomial in time, marched
through the edge as an ODE in the normal coordinate; the dissertation's jump
construction is the δ → 0 limit of the same chain, so `mode="aware"` covers
both (algebra for a jump, an ODE march for a smooth edge). What a session
needs to know before touching it:

- **1-D** (`wave1d/stiff.py`, `scripts/wave1d_stiff.py`): naive FD4 has a
  knee at h = δ; the seed stencils are fourth order at every resolution, on
  the same equispaced grid.
- **2-D seeds** (`wave2d/seeds.py`): all 30 seeds of a stencil march as one
  600-state linear ODE system, anchored at the evaluation node and restarted
  at every node and at the edge flanks. `build_operators(mode="aware")`
  dispatches smooth edges to `seed_weights`: 19-node, degree-3 seed rows for
  the elastic operator, and Δ³ rows on the naive 30-node footprint with the
  seeds annihilated. That split is what keeps the operator stable at the
  standard γ at every δ; 30-node elastic rows or 19-node Δ³ rows are not
  (`scripts/wave2d_stiff_eigenvalues.py`).
- **The rule is seed when δ ≤ h**, naive otherwise, per edge: the crossover
  sits at h ≈ δ, and a resolved edge is 1.4–2× worse seeded.
- **Measure edge error in the spurious u** (exactly 0 in the true solution at
  normal incidence); in v the resolution floor hides most of it.
- **Oblique incidence** (`--direction 1 2`, 26.6°): the x′-dependent seeds are
  essential through a sharp edge (the ablation, `seed_tangential=False`, is
  worse than naive), and every scheme, jump stencils included, converges at
  about 2.5 because the mode-converted S waves are 1.73× finer than the
  pulse. Compare oblique runs against a floor with the converted waves in it.
- **Curved edges** (`--amplitude 0.02`): the medium blends in the true signed
  normal distance (`SineInterface.signed_distance`) and the seeds march along
  the true normal through the stencil's foot point (route (a), zeroth order
  in curvature, the jump stencils' approximation). Route (a)'s geometry first
  shows at δ = 0.0025 and 19,600 nodes; route (b) is not needed on these
  node sets.
- **References**: the 1-D spectral solver mapped pointwise for normal
  incidence (`wave2d/exact.py`), Fourier in x for oblique incidence and a
  product-grid Fourier solver for curved δ > 0 (`wave2d/spectral.py`), a
  122,500-node jump-aware run for curved δ = 0. All cached under `outputs/`.
- **The standing alternative** (demo#69; `wave1d/treatments.py`,
  `wave2d/treatments.py`): cell means of compliance and density, band-limited
  coefficients and the widened edge, the same scheme on a changed medium. No
  coefficient treatment reaches the seeds' order; Schoenberg–Muir (T3) is not
  built. Results in notes §2.1 and §5.7.
- The seeds' fine-end rate (2.3–2.6) comes from the 19-node degree-3 rows
  covering the 19δ tails, not from the edge (`--seed-rtol` trims them).

### The manuscript

`paper/` is amsart built with tectonic (`references.bib`, `make_arxiv.py`;
the `paper/` pattern of Brad's weil-positivity-lab, bolza-bending and
dirichlet-bridge repos). Its rules:

- `docs/stiff-features.md` stays canonical: the manuscript quotes it, every
  number traces to a notes section or the results cache, and a `% TRACE`
  comment per section names the source. `scripts/paper_numbers.py` asserts
  the cache-backed numbers.
- Novelty wording comes only from `LITERATURE.md` §6; no unverified citation
  ships (the rule since the literature pass, demo#53).
- The results cache is `paper/data/` (one JSON per driver run, written with
  `--data-dir paper/data`). Figures and `tab_*.tex` fragments come from
  `scripts/paper_figures.py`, never hand-edited; `--check` gates byte
  identity. `\sci{m}{e}` and `\tablesetup` in `main.tex` are what the
  fragments assume.
- Notation is fixed in §2 (the `% NOTATION` comment): 2-D stresses are σ_xx,
  σ_xy, σ_yy (the code's f, g, h; h stays the spacing), K = λ + 2μ, local
  frame (x′, y′) with y′ normal, anchor x_e.
- The *Tool and computational resource disclosure* section follows the Leiden
  Declaration and is checked against the repository's history. Its text stays
  as it is after the move; the README points at the demo's pull requests.
- The pre-submission decisions (math.NA + physics.comp-ph, endorsement,
  CC BY 4.0, date, length) and the submission steps are in
  `paper/README.md`. Left to Brad: the submission and the `manuscript-v1`
  tag, here; §1 item 6 links it. #13 is the tracker (successor to demo#62).

## Provenance

- **`#N` is this repo; `demo#N` is
  `bradleypmartin/20260930-zd-ai-pdes-demo`**, where Parts 2 and 3 were built.
  Carried files and imported commit messages were rewritten once (#3), so a
  bare `#N` never means a demo issue. Write new demo references as `demo#N`.
- The history before 2026-09-22 was imported with `git filter-repo` (#2), so
  every SHA changed. `docs/split-commit-map.txt` maps demo SHA → SHA here
  (all zeros: a talk-only commit, dropped); the demo tag `part3-pre-split`
  holds the demo side. `.git-blame-ignore-revs` lists the package rename and
  the `demo#N` rewrite.
- **`git_sha` values in `paper/data/*.json` from before the split are demo
  SHAs** (7-character prefixes). Resolve them through
  `docs/split-commit-map.txt`, not through the demo tag: three of them
  (`db598c0`, `09dccf7`, `c3733ed`) are pre-rebase commits of the demo's
  demo#54 branch that never reached the demo's `main`, and the hand-added
  block at the end of the map points each at its rebased twin (identical
  `src/`, `scripts/` and `tests/`).
- The results cache's `schema` string keeps its `pdes-demo` prefix: it names
  a format, and changing it would rewrite the committed cache for nothing.
- The pull requests and their reviews for Parts 2 and 3 stay in the demo
  (Part 3 and the manuscript: PRs 35–77); the README links them.

## Repo layout

```
src/rbf_hyperbolic_interfaces/   library code
  fd_weights.py    Fornberg FD weights (shared)
  plotting.py      matplotlib style; blue = interface-aware, orange = naive;
                   aqua / violet single-hue maps for 2-D fields / errors;
                   use_print_style() for the manuscript (text width,
                   SOURCE_DATE_EPOCH pinned so PDFs are byte-identical)
  results_cache.py JSON results cache the stiff drivers write (demo#54): errors,
                   rates, truncation, snapshot and spectra records + provenance
  stiff_figures.py Part 3 figures drawn from cache records (1-D / 2-D
                   convergence) and the seed-basis figures (1-D, 2-D sections)
  stiff_tables.py  booktabs table fragments from the cache (\sci{m}{e})
  wave1d/          domain.py (periodic grid, piecewise-constant materials
                   with optional tanh edges, pulse) / operators.py (naive vs
                   interface-aware differentiation matrices, thin-layer
                   double-cross; smooth edges dispatch to stiff.py) /
                   simulate.py (RK4) / exact.py (ray-sum reference solution) /
                   spectral.py (Fourier pseudo-spectral reference for smooth
                   edges) / stiff.py (ODE-continued seed stencils, Part 3) /
                   treatments.py (the standing alternative, demo#69: cell-averaged
                   and band-limited coefficients, the widened edge, as a
                   Medium1D for the naive scheme)
  wave2d/          domain.py (materials, sine interfaces with optional tanh
                   edges for flat interfaces, interface-straddling node sets
                   by repulsion; signed normal distance for curved smooth
                   edges) / neighbors.py (periodic kNN via
                   cKDTree boxsize) / rbf.py (Gaussian RBF-FD weights with
                   polynomial augmentation, batched) / interface.py
                   (interface-aware stencils, dissertation §3.3) /
                   operators.py (sparse dx, dy, hyperviscosity, 5-field block
                   operator) / simulate.py (RK4) / exact.py (flat-interface
                   plane-wave references: ray sum for a jump, 1-D spectral
                   solver for smooth edges) / resample.py (one-sided
                   interpolation to pixel grids and other node sets) /
                   seeds.py (elastic seed bases marched in the normal
                   coordinate, along the true normal for a curved edge,
                   Part 3) / spectral.py (Fourier-in-x reference for
                   oblique incidence on flat media; product-grid reference
                   for curved smooth media, Part 3) / treatments.py (demo#69:
                   cell means and band-limited coefficients on scattered
                   nodes for the naive operator)
scripts/         drivers that write figures and clips to outputs/;
                 paper_figures.py (the manuscript's figures and tables from
                 paper/data/, demo#54); paper_numbers.py (the cache-backed
                 numbers in main.tex, demo#61, --data-dir for another cache);
                 compare_caches.py (two results caches, record by record, #6)
tests/           pytest, 261 tests; every numerical routine has one
docs/            stiff-features.md (Part 3) with its figures in figures/,
                 decisions-log.md (Part 2 record, what happened when,
                 decisions), paper-index.md, split-commit-map.txt
LITERATURE.md    the manuscript's novelty ledger (demo#53)
papers/          reference PDFs (gitignored), fetch_papers.sh, README.md with
                 sources and checksums
paper/           the Part 3 manuscript (demo#51): main.tex → main.pdf (committed),
                 references.bib, data/ (the results cache, one JSON per
                 driver run), figures/ (PDFs and tab_*.tex fragments from
                 scripts/paper_figures.py, never hand-edited), make_arxiv.py,
                 README.md, LICENSE (CC BY 4.0; code stays MIT)
outputs/         generated artifacts and the reference / operator caches
                 (gitignored)
```

## Commands

```
uv sync                                       # .venv, Python 3.13
uv run pytest                                 # tests
uv run ruff check . && uv run ruff format .
uv run python scripts/<driver>.py             # figures / clips into outputs/
uv run python scripts/wave1d_stiff.py         # Part 3 figures, ~50 s (references
                                              # cached in outputs/); --comparators adds
                                              # the coefficient treatments of demo#69 (50 s)
uv run python scripts/wave2d_stiff.py         # Part 3 2-D flat δ sweep, naive vs seeds,
                                              # and a still: ~11 min on 12 workers, ~4 min
                                              # once the seed operators are cached in outputs/
uv run python scripts/wave2d_stiff.py --direction 1 2 --widths 0.0025 0.01 \
    --modes naive aware ablate --seed-floor   # demo#41 oblique sweep, ~8 min from the cache
uv run python scripts/wave2d_stiff.py --amplitude 0.02 --widths 0 0.005 0.01 \
    --seed-floor --truncation --snapshot-width 0.005   # demo#42 curved sweep: ~12 min from
                                              # the caches, ~2 h to build them (references
                                              # 4–35 min each, seed operators 1–8 min each)
uv run python scripts/wave2d_stiff.py --modes naive widen1 widen2 cell cell2 bandlimit \
    --widths 0.0025 0.01 --no-snapshot        # demo#69 comparators, flat (~7 min from the
                                              # caches); --amplitude 0.02 --widths 0.0025
                                              # 0.005 0.01 for curved (~15 min)
uv run python scripts/wave2d_stiff_eigenvalues.py   # Part 3 2-D spectra, seed vs naive,
                                              # ~10 min at n = 900; --n 2500 --run ~25 min;
                                              # --amplitude 0.02 for the curved geometry
uv run python scripts/wave2d_demo.py --amplitude 0.02 --edge-width 0.005   # demo#42 curved
                                              # smooth-edge clip vs the product-grid reference
uv run python scripts/wave1d_stiff.py --data-dir paper/data   # any stiff driver: also write
                                              # its results JSON to paper/data (the committed
                                              # cache); --style print --format pdf for the
                                              # manuscript's look
uv run python scripts/paper_figures.py        # paper/figures/ from paper/data/: ~10 s;
                                              # --all adds the stills and spectra through the
                                              # drivers (~50 min from the outputs/ caches, the
                                              # two n = 2500 spectra most of it); --check
                                              # verifies byte identity of the cached set
uv run python scripts/paper_numbers.py        # the cache-backed numbers in main.tex;
                                              # --data-dir <dir> checks another cache
uv run python scripts/compare_caches.py paper/data <dir>   # a rebuilt cache against the
                                              # committed one, record by record (#6)
./papers/fetch_papers.sh                      # public papers, checksum-checked
(cd paper && tectonic main.tex)               # the manuscript → paper/main.pdf
(cd paper && tectonic --keep-intermediates main.tex && uv run python make_arxiv.py)
                                              # arXiv tarball: repo checks, comment
                                              # stripping, rebuild-and-compare gate, ~1 min
```

The Part 2 drivers are `wave1d_convergence`, `wave1d_demo`, `wave2d_nodes`,
`wave2d_eigenvalues`, `wave2d_hyperviscosity`, `wave2d_convergence` and
`wave2d_demo` (README lists what each draws); both demo drivers take
`--png-only` to refresh a still without rendering a clip.

## Conventions

- Python 3.13, `uv`, src layout. Deps: numpy, scipy, matplotlib (Pillow comes
  with matplotlib). Add others only with a reason; check what's used first.
- `ruff format`, line length 88. Comments explain *why*, not what.
- Tests for all numerical logic: convergence-order checks and
  analytic-solution comparisons, not just "it runs".
- Conventional Commits. Branch names `<issue>-<short-description>`. One PR per
  issue or pass; Brad reviews and merges, with a merge commit: the results
  cache records the commit a driver ran at, and a squash would orphan it.
- Default driver parameters run in seconds; bigger runs sit behind flags.
- Part 3 figures referenced from `docs/stiff-features.md` are committed
  under `docs/figures/` so the notes read on GitHub.
- Manuscript: after any edit under `paper/`, rebuild with tectonic, look at
  the changed pages with `pdftoppm`, and commit `main.pdf` with the source.
  `\date` is fixed by hand, never `\today`.

## Hard constraints

- **This repo is PUBLIC (MIT; `paper/` CC BY 4.0).** Never commit PDFs other
  than `paper/main.pdf` and the manuscript's figures, credentials, or
  anything from FullContact / Ziff Davis systems. The content is Brad's own
  academic work plus public papers; nothing employer-proprietary goes here.
- `papers/*.pdf` and `outputs/` are gitignored on purpose; don't un-ignore.
  `paper/main.pdf` is committed on every manuscript change.
- Everything the manuscript and the notes cite must trace to a source we've
  read (a PDF in `papers/`, a verified bib entry, or a URL cited inline).

## Reference implementation (MATLAB, read-only)

`~/MathGraduateResearchAndCourseWork/` (separate personal repo, not vendored
here) holds Brad's original MATLAB. The port reproduces the *method* and
restructures the code; module docstrings name the MATLAB functions and the
dissertation equations they follow.

- `waveEq1DMatlab/FD4wave1DAC.m` + `runDriver1DWE.m`: 1-D two-way wave
  equation in first-order form on periodic [-1, 1), equispaced 4th-order FD,
  RK4. Heterogeneous layer on [0, w) with (c2, rho2). Stencils that cross an
  interface are rebuilt from piecewise polynomials that satisfy the PDE's
  continuity conditions (Taylor terms "translated" across the interface via
  the operator); stencils that cross *both* sides of a thin layer get a
  second translation ("double-cross"). `weights.m` is Fornberg's algorithm.
  Ported in `wave1d/`, for piecewise-constant materials only (the exact
  ray-sum reference needs that); the dissertation's smoothly varying
  background speed is not ported.
- `waveEq2DMatlab/` (`EWE2DRbfPrep.m`, 3k lines of local functions;
  `runScript170107HO.m` still uses pre-rename names): 2-D elastic wave
  equation (u, v, f, g, h) on a doubly periodic unit square with two curved
  interfaces. Node set from a repulsion process, periodic kNN via tiling +
  `knnsearch`, Gaussian RBF-FD weights with polynomial augmentation, Δ³
  hyperviscosity, sparse block operators, RK4. Ported in `wave2d/` with these
  exceptions:
  - **Triple stencils** (`tripleVec`: one stencil that sees both interfaces of
    a thin band) are not ported; `operators.py` raises `NotImplementedError`
    if a band is thinner than a stencil. The test band is 0.25 wide.
  - **Variable Lamé parameters in the band** (λ = μ = 4 + sin 2πx sin 2πy,
    JCP 2017 test case 1) are not ported; the band is a constant material.
  - **FD/RBF hybrid** away from interfaces is not ported; the port is RBF-FD
    everywhere ("all RBF"), a runtime optimisation we never needed.
  - Point sources and the mini-Marmousi model (dissertation §3.4.3) are out
    of scope.

MATLAB → Python mapping as used: `knnsearch` on a tiled node set →
`scipy.spatial.cKDTree` with `boxsize` (no tiling); per-stencil `A\b` →
batched `numpy.linalg.solve`; `sparse(i,j,v)` → `scipy.sparse.csr_array`;
`polyval`/`polyder` tables → monomial exponent tables
(`wave2d/rbf.py: monomial_exponents`). No null-space computations: the port
follows the JCP 2017 square-matrix construction.

## Working style (from Brad's global preferences)

Concise, technical, sparring-partner mode. State intent before non-trivial
changes, then take the wheel. Verify APIs rather than guess. Assume a second
agent may review the work.
