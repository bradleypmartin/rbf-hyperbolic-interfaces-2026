# 20260930-zd-ai-pdes-demo

Material for a 30-minute demo on **2026-09-30** about what frontier AI models
can currently do in research mathematics, in one narrow slice of it: partial
differential equations.

Two halves:

1. **Navier–Stokes, the state of play.** OpenAI announced on 2026-09-08 a
   166-page manuscript, *Finite Time Blowup for Navier–Stokes*, plus Lean 4
   certificates, claiming alternatives (C) and (D) of the Clay Millennium
   problem: smooth, compactly supported forcing under which no global smooth
   finite-energy solution exists. What was claimed, how it was produced, how it
   is being checked, and the questions it raises. Notes in `docs/`.
2. **Working with Claude on my own research.** A live/recorded exposition of
   Claude Code and me re-deriving and re-implementing in Python the
   interface-aware wave solvers from my 2016 CU Boulder applied-math
   dissertation (radial basis function-generated finite differences, RBF-FD).
   1-D wave equation with a thin heterogeneous layer first; 2-D RBF-FD on a
   scattered node set with curved interfaces if time allows.

## Quickstart

```sh
uv sync                        # Python 3.13 venv with numpy / scipy / matplotlib
uv run pytest                  # tests
./papers/fetch_papers.sh       # download the public reference PDFs (gitignored)
```

Drivers in `scripts/` write figures and animations to `outputs/` (gitignored):

```sh
uv run python scripts/wave1d_convergence.py   # error vs resolution (dissertation Fig. 2-8)
uv run python scripts/wave1d_demo.py          # two-panel MP4 + snapshot PNG, naive vs aware
uv run python scripts/wave1d_demo.py --n 100 --sharpness 150 --markers --out outputs/wave1d_naive_vs_aware_coarse.mp4  # coarse grid, ringing visible
uv run python scripts/wave2d_nodes.py         # interface-fitted node set (dissertation Fig. 3-3)
uv run python scripts/wave2d_eigenvalues.py   # operator spectrum with/without hyperviscosity (Fig. 3-2)
uv run python scripts/wave2d_hyperviscosity.py  # error and stability vs hyperviscosity amplitude
```

## Layout

| Path | Contents |
| --- | --- |
| `src/pdes_demo/` | Library: FD / RBF-FD stencil generation, interface treatment, time stepping, node sets |
| `scripts/` | Runnable drivers for the demo |
| `tests/` | pytest suite (convergence and analytic checks) |
| `docs/` | `demo-outline.md` (plan + timeline), Navier–Stokes notes, derivations |
| `papers/` | Index of reference papers with links and checksums; PDFs are not committed |

## Reference papers

See [`papers/README.md`](papers/README.md) for the full table. Headline links:

- OpenAI, *Finite Time Blowup for Navier–Stokes* (2026):
  [PDF](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf) ·
  [announcement](https://openai.com/index/navier-stokes-solution/) ·
  [Lean certificates](https://github.com/openai/NavierStokesAndEuler)
- Clay Mathematics Institute, [official problem statement](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf) (Fefferman)
- B. Martin, *Application of RBF-FD to Wave and Heat Transport Problems in
  Domains with Interfaces*, PhD dissertation, CU Boulder, 2016
  ([ProQuest](https://www.proquest.com/openview/ae4d936114520c2d34e604adeda7d81a/1?pq-origsite=gscholar&cbl=18750))
- B. Martin, B. Fornberg, A. St-Cyr, *Seismic modeling with RBF-FD*,
  Geophysics 2015 ([preprint](https://www.colorado.edu/amath/sites/default/files/attached-files/2015_mfstc_rbf-fd_2d_geophys_submitted_0.pdf))
- B. Martin, B. Fornberg, *Seismic modeling with RBF-FD – a simplified
  treatment of interfaces*, J. Comput. Phys. 2017
  ([preprint](https://www.colorado.edu/amath/sites/default/files/attached-files/2016_mf_rbf-fd_seismic_jcp_submitted.pdf))

The original MATLAB implementations live in a separate repo,
[`bradleypmartin/MathGraduateResearchAndCourseWork`](https://github.com/bradleypmartin/MathGraduateResearchAndCourseWork),
and are used as a read-only reference for the Python ports here.

## Status

- [x] Environment, papers, plan (2026-09-17)
- [x] 1-D wave equation: Fornberg FD weights, interface-aware stencils,
      thin-layer "double-cross", RK4, exact reference solution, tests,
      convergence figure, two-panel video (2026-09-17)
- [ ] Navier–Stokes notes and slides
- [x] 2-D RBF-FD, naive: interface-fitted periodic node set, kNN stencils,
      Gaussian RBF-FD weights, hyperviscosity, sparse block operators, RK4,
      analytic plane-wave reference (2026-09-17)
- [x] 2-D interface-aware stencils: coupled piecewise-polynomial bases
      across interfaces, error down to the resolution floor on the analytic
      test problem (2026-09-17)
- [ ] 2-D curved-interface runs, two-panel video
- [ ] Rehearsal

## License

MIT. Reference papers are the property of their respective authors and
publishers and are not redistributed here.
