# CLAUDE.md — 20260930-zd-ai-pdes-demo

## What this is

Material for Brad's 30-minute talk to Ziff Davis coworkers on **2026-09-30**,
*AI and Applied Math circa September 2026: excitement, ethics, and individual
exploration*. Thesis: one pattern at every scale, agents' power paired with a
human's intuition in a field. Everything is built; what remains is the Sep 29
rehearsal, a freeze, and the `talk-2026-09-30` tag (issues #18, #19).

1. **Navier–Stokes (~15 min).** OpenAI's 2026-09-08 claim of finite-time
   blowup for 3-D Navier–Stokes (166-page manuscript + Lean 4 certificates),
   the parallel Alpöge–Buckmaster result and the credit dispute, and what
   "machine-checked" does and does not mean. Sourced notes in
   `docs/navier-stokes-notes.md`.
2. **Working with Claude (~15 min).** Brad and Claude re-derived and ported to
   Python the interface-aware wave solvers from Brad's 2016 CU Boulder
   dissertation: 1-D finite differences through a layer, and 2-D elastic
   RBF-FD on scattered nodes with curved interfaces, each verified against a
   reference solution. All of it landed on 2026-09-17.

**Part 3 (not in the talk; #27, sub-issues #28–#34).** Started 2026-09-19
after the freeze: Brad's never-published idea that a stencil crossing a
material edge too steep for the grid can be built from ODE-continued
"seeds" (the t = 0 profiles of time-polynomial solutions) instead of
monomials, on the same equispaced grid. Built and confirmed in 1-D: naive
FD4 has a knee at h = δ, the seed stencils are fourth order at every
resolution, and the dissertation's jump construction is the δ → 0 limit.
The 2-D chain (#36–#42, flat first, curved last) has started: #36 added
smooth flat edges to `LayeredMedium2D` and a normal-incidence reference
from the 1-D spectral solver; #37 measured the naive RBF-FD baseline
(`scripts/wave2d_stiff.py`): the resolution floor hides most of the edge
error in v; the spurious u (exactly 0 in the true solution) separates
unresolved from resolved edges 10× more sharply. No 2-D seeds yet.
Notes and results are in `docs/stiff-features.md` §5.

Audience: bright tech workers with no assumed PDE background. **No live
coding.** The deliverables are `slides/talk.pdf` (19 pages), three clips in
`slides/videos/` played from `slides/clips.html`, and the speaker script
`slides/notes.md`. GitHub Pages serves `main` at
<https://bradleypmartin.github.io/20260930-zd-ai-pdes-demo/>.

Results tables, decisions log, and what happened when: `docs/demo-outline.md`.
Papers: `papers/README.md`. **Before reading a PDF, check
`docs/paper-index.md`** for the page ranges that matter and read only those
(`pdftotext -f A -l B -layout <pdf> -`).

## Repo layout

```
src/pdes_demo/   library code
  fd_weights.py    Fornberg FD weights (shared)
  plotting.py      matplotlib style; blue = interface-aware, orange = naive;
                   aqua / violet single-hue maps for 2-D fields / errors
  wave1d/          domain.py (periodic grid, piecewise-constant materials
                   with optional tanh edges, pulse) / operators.py (naive vs
                   interface-aware differentiation matrices, thin-layer
                   double-cross; smooth edges dispatch to stiff.py) /
                   simulate.py (RK4) / exact.py (ray-sum reference solution) /
                   spectral.py (Fourier pseudo-spectral reference for smooth
                   edges) / stiff.py (ODE-continued seed stencils, Part 3)
  wave2d/          domain.py (materials, sine interfaces with optional tanh
                   edges for flat interfaces, interface-straddling node sets
                   by repulsion) / neighbors.py (periodic kNN via
                   cKDTree boxsize) / rbf.py (Gaussian RBF-FD weights with
                   polynomial augmentation, batched) / interface.py
                   (interface-aware stencils, dissertation §3.3) /
                   operators.py (sparse dx, dy, hyperviscosity, 5-field block
                   operator) / simulate.py (RK4) / exact.py (flat-interface
                   plane-wave references: ray sum for a jump, 1-D spectral
                   solver for smooth edges) / resample.py (one-sided
                   interpolation to pixel grids and other node sets)
scripts/         drivers that write figures and clips to outputs/;
                 check_slide_quotes.py
tests/           pytest, 143 tests; every numerical routine has one
docs/            demo-outline.md, navier-stokes-notes.md, paper-index.md,
                 stiff-features.md (Part 3) with its figures in figures/
slides/          talk.tex → talk.pdf (committed), notes.md (speaker script with
                 clip cues), clips.html (keyboard clip player), figures/ and
                 videos/ (committed; build.sh refreshes them from outputs/),
                 build.sh, README.md
papers/          reference PDFs (gitignored), fetch_papers.sh, README.md with
                 sources and checksums
outputs/         generated artifacts (gitignored)
index.html       GitHub Pages landing page (.nojekyll at the root)
```

## Commands

```
uv sync                                       # .venv, Python 3.13
uv run pytest                                 # tests
uv run ruff check . && uv run ruff format .
uv run python scripts/<driver>.py             # figures / clips into outputs/
uv run python scripts/wave1d_stiff.py         # Part 3 figures, ~50 s (references
                                              # cached in outputs/)
uv run python scripts/wave2d_stiff.py         # Part 3 2-D naive baseline, ~2 min
                                              # per pulse (1-D references cached)
./slides/build.sh                             # copy figures and clips from outputs/,
                                              # crop, tectonic → slides/talk.pdf
uv run python scripts/check_slide_quotes.py   # every \q{} in talk.tex is in the notes
./papers/fetch_papers.sh                      # public papers, checksum-checked
```

Clip renders are listed in `slides/README.md`. Both demo drivers take
`--png-only` to refresh a still without touching a committed clip.

## Conventions

- Python 3.13, `uv`, src layout. Deps: numpy, scipy, matplotlib (Pillow comes
  with matplotlib). Add others only with a reason; check what's used first.
- `ruff format`, line length 88. Comments explain *why*, not what.
- Tests for all numerical logic: convergence-order checks and
  analytic-solution comparisons, not just "it runs".
- Conventional Commits. Branch names `<issue>-<short-description>`. One PR per
  issue or pass; Brad reviews and merges.
- Default driver parameters run in seconds; bigger runs sit behind flags.
- Slides: Brad refers to a slide by its footer number (n/16). The title and
  the two section frames are unnumbered, so PDF page = n + 3. After any deck
  edit: `./slides/build.sh`, render the changed pages with `pdftoppm` and look
  at them, run the quote checker, commit `talk.tex` and `talk.pdf` together.
  `\q{}` marks a sourced quotation (checked against the notes), `\sq{}` a
  scare quote.
- Clips: iterate in `outputs/`; copy into `slides/videos/` and commit only
  when the content changed (ffmpeg output is not byte-identical run to run).
- Part 3 figures referenced from `docs/stiff-features.md` are committed
  under `docs/figures/` so the notes read on GitHub; nothing of Part 3
  touches the deck or the clips.
- Deck and clip passes go one item at a time, one commit per item, so the PR
  history reads item by item.

## Hard constraints

- **This repo is PUBLIC (MIT).** Never commit PDFs, credentials, or anything
  from FullContact / Ziff Davis systems. Demo content is Brad's own academic
  work plus public papers; nothing employer-proprietary goes here.
- `papers/*.pdf` and `outputs/` are gitignored on purpose; don't un-ignore.
  `slides/figures/`, `slides/videos/` and `slides/talk.pdf` are committed on
  purpose so the talk is self-contained from a fresh clone, and `talk.pdf` is
  committed on every deck change.
- Don't fabricate details about the OpenAI paper or its reception. Everything
  stated in `docs/` and on the slides must trace to a source we've read (PDF
  in `papers/` or a URL cited inline).

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
    if a band is thinner than a stencil. The demo band is 0.25 wide.
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
