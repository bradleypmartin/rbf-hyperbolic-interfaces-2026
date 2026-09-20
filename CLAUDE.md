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
unresolved from resolved edges 10× more sharply; #38 built the elastic seeds
(`wave2d/seeds.py`): all 30 seeds of one stencil march as one 600-state
ODE system in the normal coordinate, anchored at the evaluation node, with
constant-material, jump-limit, residual and conditioning checks; #39 built
`seed_weights` and the dispatch in `build_operators(mode="aware")` for
smooth edges (19-node seed rows for the elastic operator, Δ³ rows on the
naive 30-node footprint with the seeds annihilated) and answered the
stability question: stable at the standard γ at every δ
(`scripts/wave2d_stiff_eigenvalues.py`); at n = 2500, δ = h/8 the seeds
cut the spurious u 5.6× and v below the naive floor. #40 ran the flat δ
sweep (`scripts/wave2d_stiff.py`, naive vs seeds at every (n, δ), seed
marches on a process pool, operators cached under `outputs/`): through
an edge the nodes never resolve (δ = 0.0025) the seeds are fourth order
at every n, 20× below naive in v and 67× in spurious u at 19,600 nodes,
at their own floor, half the naive one; the crossover is at h ≈ δ and a
resolved edge (δ = 0.04) is 1.4–2× worse seeded, so the rule is seed
when δ ≤ h. #41 sent a plane-wave train in at 26.6° (`oblique_p_wave`,
`--direction 1 2`) against a Fourier-in-x reference (`wave2d/spectral.py`,
one complex pseudo-spectral system per x-mode): seeds beat naive 1.4–1.9×
and the x'-dependent seeds are essential through a sharp edge (the
ablation, `seed_tangential=False`, is worse than naive), but every scheme
converges at about 2.5 there, the jump-aware stencils included, because
the mode-converted S waves are 1.73× finer than the pulse and sit at a
pre-asymptotic floor on these node sets; compare oblique runs against a
floor with the converted waves in it. #42 (curved, the last of the
chain): the medium blends in the true signed normal distance
(`SineInterface.signed_distance`, one band image per point), the seeds
march along the true normal through the stencil's foot point (route (a),
zeroth order in curvature, the same approximation as the jump stencils),
and a product-grid Fourier solver (`wave2d/spectral.py: run_fourier_2d`,
`GridState`) is the reference for δ > 0, a 122,500-node jump-aware run
for δ = 0 (`scripts/wave2d_stiff.py --amplitude 0.02`, both cached under
`outputs/`). Results: the curved jump stencils sit at the resolution
floor at every n (3.6th order to 19,600 nodes); through δ = 0.005 the
curved seeds sit at the seed operator's own floor at every n, 2.2–7×
below naive, and through δ = 0.01 the curved numbers equal the flat ones,
crossover at h = δ as before; the seed rows' truncation error on the
true curved solution converges at the bulk rate; through δ = 0.0025 the
seeds are 3–12× below naive and route (a)'s geometry first shows (1.5×
the flat seeds at 19,600 nodes, level with the seed floor), so route
(b) is not needed on these node sets and would start to be beyond
them at δ ≤ h/3; the spectra at 2500 nodes are the flat ones.
The fine-end rate of the seeds (2.3–2.6) is the 19-node degree-3 rows
covering the 19δ tails, not the edge (`--seed-rtol` trims them).
Derivation in `docs/stiff-features.md` §4, results in §5 (§5.6 curved).

**Manuscript (#51, sub-issues #52–#62).** `paper/` holds the arXiv-ready
write-up of Part 3 (amsart, tectonic, `references.bib`, `make_arxiv.py`;
the `paper/` pattern of Brad's weil-positivity-lab, bolza-bending and
dirichlet-bridge repos). `docs/stiff-features.md` stays canonical: the
manuscript quotes it, every number traces to a notes section or the
results cache (#54), and a `% TRACE` comment per section names the source.
Figures and tables come from committed scripts, never hand-edited (#54).
Bibliography entries enter flagged `TODO(verify)` and are cited only after
the literature pass (#53) verifies them; novelty is cited, not claimed,
until `LITERATURE.md` buckets it. One sub-issue per PR, in the dependency
order on #51.

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
                   aqua / violet single-hue maps for 2-D fields / errors;
                   use_print_style() for the manuscript (text width,
                   SOURCE_DATE_EPOCH pinned so PDFs are byte-identical)
  results_cache.py JSON results cache the stiff drivers write (#54): errors,
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
                   edges) / stiff.py (ODE-continued seed stencils, Part 3)
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
                   for curved smooth media, Part 3)
scripts/         drivers that write figures and clips to outputs/;
                 check_slide_quotes.py; paper_figures.py (the manuscript's
                 figures and tables from paper/data/, #54)
tests/           pytest, 208 tests; every numerical routine has one
docs/            demo-outline.md, navier-stokes-notes.md, paper-index.md,
                 stiff-features.md (Part 3) with its figures in figures/
slides/          talk.tex → talk.pdf (committed), notes.md (speaker script with
                 clip cues), clips.html (keyboard clip player), figures/ and
                 videos/ (committed; build.sh refreshes them from outputs/),
                 build.sh, README.md
papers/          reference PDFs (gitignored), fetch_papers.sh, README.md with
                 sources and checksums
paper/           the Part 3 manuscript (#51): main.tex → main.pdf (committed),
                 references.bib, data/ (the results cache, one JSON per
                 driver run), figures/ (PDFs and tab_*.tex fragments from
                 scripts/paper_figures.py, never hand-edited), make_arxiv.py,
                 README.md, LICENSE (CC BY 4.0; code stays MIT)
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
uv run python scripts/wave2d_stiff.py         # Part 3 2-D flat δ sweep, naive vs seeds,
                                              # and a still: ~11 min on 12 workers, ~4 min
                                              # once the seed operators are cached in outputs/
uv run python scripts/wave2d_stiff.py --direction 1 2 --widths 0.0025 0.01 \
    --modes naive aware ablate --seed-floor   # #41 oblique sweep, ~8 min from the cache
uv run python scripts/wave2d_stiff.py --amplitude 0.02 --widths 0 0.005 0.01 \
    --seed-floor --truncation --snapshot-width 0.005   # #42 curved sweep: ~12 min from
                                              # the caches, ~2 h to build them (references
                                              # 4–35 min each, seed operators 1–8 min each)
uv run python scripts/wave2d_stiff_eigenvalues.py   # Part 3 2-D spectra, seed vs naive,
                                              # ~10 min at n = 900; --n 2500 --run ~25 min;
                                              # --amplitude 0.02 for the curved geometry
uv run python scripts/wave2d_demo.py --amplitude 0.02 --edge-width 0.005   # #42 curved
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
./slides/build.sh                             # copy figures and clips from outputs/,
                                              # crop, tectonic → slides/talk.pdf
uv run python scripts/check_slide_quotes.py   # every \q{} in talk.tex is in the notes
./papers/fetch_papers.sh                      # public papers, checksum-checked
(cd paper && tectonic main.tex)               # the manuscript → paper/main.pdf
(cd paper && tectonic --keep-intermediates main.tex && uv run python make_arxiv.py)
                                              # arXiv tarball (needs main.bbl)
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
- Manuscript: after any edit under `paper/`, rebuild with tectonic, look at
  the changed pages with `pdftoppm`, and commit `main.pdf` with the source.
  `\date` is fixed by hand, never `\today`. Section stubs are `\stub{}`
  lines, visible in the PDF until the owning sub-issue replaces them.

## Hard constraints

- **This repo is PUBLIC (MIT).** Never commit PDFs, credentials, or anything
  from FullContact / Ziff Davis systems. Demo content is Brad's own academic
  work plus public papers; nothing employer-proprietary goes here.
- `papers/*.pdf` and `outputs/` are gitignored on purpose; don't un-ignore.
  `slides/figures/`, `slides/videos/` and `slides/talk.pdf` are committed on
  purpose so the talk is self-contained from a fresh clone, and `talk.pdf` is
  committed on every deck change. `paper/main.pdf` likewise, on every
  manuscript change.
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
