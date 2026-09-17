# CLAUDE.md — 20260930-zd-ai-pdes-demo

## What this is

Material for a 30-minute talk on **2026-09-30** to Ziff Davis coworkers about
frontier-AI capability in research mathematics. Two halves:

1. **Navier–Stokes (talk, ~15 min).** OpenAI's 2026-09-08 claim of finite-time
   blowup for 3-D Navier–Stokes (166-page manuscript + Lean 4 certificates),
   the surrounding context and controversy, and the ethics/verification
   questions. Notes live in `docs/`.
2. **Exposition (~15 min).** Brad + Claude re-derive and re-implement in
   Python the interface-aware wave solvers from Brad's 2016 CU Boulder
   dissertation. 1-D first; 2-D elastic RBF-FD if 1-D and the 2-D prep land.

Audience: bright tech workers with no assumed PDE background. **No live
coding.** Deliverables are prepared ahead: two two-panel videos (naive FD vs
interface-aware, same resolution; 1-D and 2-D) in `outputs/`, and a short
slide deck PDF in `slides/` covering both halves.

Plan and timeline: `docs/demo-outline.md`. Papers: `papers/README.md`.
**Before reading a PDF, check `docs/paper-index.md`** for the page ranges that
matter and read only those (`pdftotext -f A -l B -layout <pdf> -`).

## Repo layout

```
src/pdes_demo/   library code
  fd_weights.py    Fornberg FD weights (shared)
  plotting.py      matplotlib style + validated two-series palette
  wave1d/          domain.py (grid, materials, pulse) / operators.py (naive vs
                   interface-aware differentiation matrices) / simulate.py (RK4)
                   / exact.py (ray-sum reference solution)
  wave2d/          (planned) same split: domain, operators, simulate, + node
                   sets, periodic kNN, RBF-FD weights, hyperviscosity
scripts/         runnable drivers that produce figures/animations in outputs/
tests/           pytest; every new numerical routine gets a test
docs/            demo outline, Navier–Stokes notes, derivations
slides/          slide deck source + PDF
papers/          reference PDFs (gitignored) + fetch script + index
outputs/         generated artifacts (gitignored)
```

## Commands

```
uv sync                         # create .venv, install everything (Python 3.13)
uv run pytest                   # tests
uv run ruff check . && uv run ruff format .
uv run python scripts/<driver>.py
./papers/fetch_papers.sh        # download public papers
```

## Conventions

- Python 3.13, `uv`, src layout. Deps: numpy, scipy, matplotlib. Add others
  only with a reason (check what's already used first).
- `ruff format`, line length 88. Comments explain *why*, not what.
- Tests for all new numerical logic: convergence-order checks and
  analytic-solution comparisons, not just "it runs".
- Conventional Commits. Branch names `<issue>-<short-description>`.
- Keep default driver parameters fast enough to run live in the demo
  (seconds, not minutes). Expose bigger runs behind CLI flags.

## Hard constraints

- **This repo is PUBLIC (MIT).** Never commit PDFs, credentials, or anything
  from FullContact / Ziff Davis systems. Demo content is Brad's own academic
  work plus public papers; nothing employer-proprietary goes here.
- `papers/*.pdf` and `outputs/` are gitignored on purpose. Don't un-ignore.
- Don't fabricate details about the OpenAI paper or its reception. Everything
  stated in `docs/` must trace to a source we've read (PDF in `papers/` or a
  URL cited inline).

## Reference implementation (MATLAB, read-only)

`~/MathGraduateResearchAndCourseWork/` (separate personal repo, not vendored
here) holds Brad's original MATLAB. Port with understanding: reproduce the
*method*, restructure the code. Key files:

- `waveEq1DMatlab/FD4wave1DAC.m` + `runDriver1DWE.m`: 1-D two-way wave
  equation in first-order form on periodic [-1, 1), equispaced 4th-order FD,
  RK4. Heterogeneous layer on [0, w) with (c2, rho2). Stencils that cross an
  interface are rebuilt from piecewise polynomials that satisfy the PDE's
  continuity conditions (Taylor terms "translated" across the interface via
  the operator); stencils that cross *both* sides of a thin layer get a
  second translation ("double-cross"). `doubleNaiveFlag` disables that for
  comparison. `weights.m` is Fornberg's FD-weight algorithm.
- `waveEq2DMatlab/`: 2-D elastic wave equation (u, v, s1, s2, s3) on a
  doubly-periodic unit square with two curved interfaces. Node set from a
  repulsion process (`mos2dsqperiodic7`), periodic kNN via tiling +
  `knnsearch`, RBF-FD weights (IMQ/GA + polynomial augmentation, per-stencil
  dense solves), hyperviscosity, sparse block operators, RK4. Hybrid: plain
  FD away from interfaces. `EWE2DRbfPrep.m` (3k lines) contains most local
  functions; `runScript170107HO.m` still references pre-rename function
  names.

MATLAB → Python mapping we'll need: `knnsearch` → `scipy.spatial.cKDTree`;
`A\b` → `scipy.linalg.solve`/`lstsq`; `null()` → `scipy.linalg.null_space`;
`sparse(i,j,v)` → `scipy.sparse.csr_array`; `polyval/polyder` →
`numpy.polynomial`.

## Working style (from Brad's global preferences)

Concise, technical, sparring-partner mode. State intent before non-trivial
changes, then take the wheel. Verify APIs rather than guess. Assume a second
agent may review the work.
