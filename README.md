# 20260930-zd-ai-pdes-demo

Material for a 30-minute talk on **2026-09-30** to Ziff Davis coworkers,
*AI and Applied Math circa September 2026: excitement, ethics, and individual
exploration*. What frontier AI currently does in research mathematics, seen at
two scales in one narrow slice of it: partial differential equations.

1. **Navier–Stokes, the state of play.** OpenAI announced on 2026-09-08 a
   166-page manuscript, *Finite Time Blowup for Navier–Stokes*, plus Lean 4
   certificates, claiming alternatives (C) and (D) of the Clay Millennium
   problem: smooth, compactly supported forcing under which no global smooth
   finite-energy solution exists. What was claimed, how it was produced, the
   parallel Alpöge–Buckmaster result and the credit dispute, how the claim is
   being checked, and the questions it raises. Every statement on the slides
   traces to [`docs/navier-stokes-notes.md`](docs/navier-stokes-notes.md),
   which cites a PDF in `papers/` or a URL.
2. **Working with Claude on my own research.** Claude Code and I re-derived
   and re-implemented in Python the interface-aware wave solvers from my 2016
   CU Boulder applied-math dissertation: the 1-D wave equation through a
   heterogeneous layer with interface-aware finite differences, and the 2-D
   elastic wave equation on a scattered node set with curved interfaces using
   radial basis function-generated finite differences (RBF-FD). Both are
   verified against reference solutions: standard stencils lose accuracy at a
   material boundary, the interface-aware ones do not. All of it was built on
   2026-09-17.

After the freeze, and not part of the talk: **Part 3**, an idea from my
research that never got written up. A stencil crossing a material edge too
steep for the grid to resolve is built from "seeds" continued through the
edge by ODEs instead of monomials, on the same equispaced grid. Built and
confirmed in 1-D on 2026-09-19, with a design note for 2-D:
[`docs/stiff-features.md`](docs/stiff-features.md) (issue #27). The
arXiv write-up of Part 3 is in [`paper/`](paper/README.md) (issue #51:
drafted, assembled and packaged for submission on 2026-09-20, with the
pre-submission decisions and the AI-assistance disclosure recorded in its
README); [`LITERATURE.md`](LITERATURE.md) is its novelty ledger, where
the claim is itemised and every "not found" is tied to the search that
produced it (#53).

## Talk materials

Hosted from `main` by GitHub Pages: [landing page](https://bradleypmartin.github.io/20260930-zd-ai-pdes-demo/),
[slides (PDF)](https://bradleypmartin.github.io/20260930-zd-ai-pdes-demo/slides/talk.pdf),
[the three clips](https://bradleypmartin.github.io/20260930-zd-ai-pdes-demo/slides/clips.html).
Sources in `slides/` (see its README for the build, the quotation checker, and
the clip player).

To play the clips in the talk: open `slides/clips.html` in Chrome, press `F`
for full screen, then `1`, `2` or `3` to play a clip from the start (`space`
pauses, `R` restarts). `slides/notes.md` is the speaker script and says when
to switch to which clip.

## Quickstart

```sh
uv sync                                       # Python 3.13 venv with numpy / scipy / matplotlib
uv run pytest                                 # 106 tests: convergence orders and analytic comparisons
./papers/fetch_papers.sh                      # public reference PDFs (gitignored), checksum-checked
./slides/build.sh                             # rebuild slides/talk.pdf with tectonic
uv run python scripts/check_slide_quotes.py   # every quotation on a slide is in the notes
open slides/clips.html                        # the clip player (keys 1, 2, 3; F for full screen)
```

Drivers in `scripts/` write figures and animations to `outputs/` (gitignored):

```sh
uv run python scripts/wave1d_convergence.py   # error vs resolution (dissertation Fig. 2-8)
uv run python scripts/wave1d_demo.py          # two-panel MP4 + snapshot PNG, naive vs aware
uv run python scripts/wave1d_demo.py --n 100 --sharpness 150 --out outputs/wave1d_naive_vs_aware_coarse.mp4  # clip 1: coarse grid, ringing visible
uv run python scripts/wave2d_nodes.py         # interface-fitted node set (dissertation Fig. 3-3)
uv run python scripts/wave2d_eigenvalues.py   # operator spectrum with/without hyperviscosity (Fig. 3-2)
uv run python scripts/wave2d_hyperviscosity.py  # error and stability vs hyperviscosity amplitude
uv run python scripts/wave2d_convergence.py   # 2-D error vs resolution, flat and curved interfaces (Fig. 3-5 / 3-8)
uv run python scripts/wave2d_demo.py          # clip 2: 2-D two-panel MP4 + snapshot PNG; --amplitude 0.02 for clip 3 (curved)
uv run python scripts/wave1d_stiff.py         # Part 3: 1-D knee plot, snapshot and seeds through a stiff smooth edge
uv run python scripts/wave2d_stiff.py         # Part 3: 2-D flat delta sweep, naive RBF-FD vs seed stencils through smooth edges, and a still (seed operators cached under outputs/)
```

Both demo drivers take `--png-only` to refresh a still without re-rendering a
clip.

## Layout

| Path | Contents |
| --- | --- |
| `src/pdes_demo/` | Library. `wave1d/`: FD stencils across interfaces, RK4, exact ray-sum solution; for Part 3, smooth tanh edges, a Fourier pseudo-spectral reference and ODE-continued seed stencils. `wave2d/`: node sets, periodic kNN, Gaussian RBF-FD weights, interface-aware stencils, hyperviscosity, sparse elastic operators, RK4, analytic plane-wave reference, one-sided resampling; for Part 3, smooth tanh edges on flat interfaces and a spectral normal-incidence reference. Shared Fornberg weights and plotting palette. |
| `scripts/` | Drivers for the figures and clips; `check_slide_quotes.py` |
| `tests/` | pytest suite (convergence and analytic checks) |
| `docs/` | `demo-outline.md` (results tables, decisions log, what happened when), `navier-stokes-notes.md` (sourced notes for Part 1), `paper-index.md` (page ranges per PDF), `stiff-features.md` and `figures/` (Part 3) |
| `slides/` | `talk.tex` → `talk.pdf`, `notes.md` speaker script, `figures/`, `videos/` (the three clips), `clips.html` clip player, `build.sh` |
| `papers/` | Index of reference papers with links and checksums, fetch script; PDFs are not committed |
| `paper/` | The Part 3 manuscript: `main.tex` → `main.pdf` (tectonic), `references.bib`, `make_arxiv.py`; see its README |
| `index.html` | GitHub Pages landing page |

## Reference papers

See [`papers/README.md`](papers/README.md) for the full table. Headline links:

- OpenAI, *Finite Time Blowup for Navier–Stokes* (2026):
  [PDF](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf) ·
  [announcement](https://openai.com/index/navier-stokes-solution/) ·
  [Lean certificates](https://github.com/openai/NavierStokesAndEuler)
- OpenAI, companion *Finite Time Blowup for the Euler Equation* (2026):
  [PDF](https://cdn.openai.com/pdf/315b36cd-ec98-4023-8342-93345194ece1/euler.pdf)
- L. Alpöge, T. Buckmaster, *Blowup for the Euler equations with smooth
  forcing* (2026, preprint): [PDF](https://cims.nyu.edu/~tristanb/euler.pdf) ·
  [Lean](https://github.com/tristanbuckmaster/fluid_lean)
- T. Buckmaster, [statement of 2026-09-07](https://cims.nyu.edu/~tristanb/statement.pdf)
  on the results, the tools used, and the contacts with OpenAI
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
and were used as a read-only reference for the Python ports here. `CLAUDE.md`
lists what was ported and what was not.

## Status

- [x] Environment, papers, plan (2026-09-17)
- [x] 1-D wave equation: Fornberg FD weights, interface-aware stencils,
      thin-layer "double-cross", RK4, exact reference solution, tests,
      convergence figure, two-panel video (2026-09-17)
- [x] Navier–Stokes notes and slides: sourced notes in `docs/`, Beamer deck
      and speaker script in `slides/` (2026-09-17)
- [x] 2-D RBF-FD, naive: interface-fitted periodic node set, kNN stencils,
      Gaussian RBF-FD weights, hyperviscosity, sparse block operators, RK4,
      analytic plane-wave reference (2026-09-17)
- [x] 2-D interface-aware stencils: coupled piecewise-polynomial bases
      across interfaces, error down to the resolution floor on the analytic
      test problem (2026-09-17)
- [x] 2-D curved-interface runs, convergence figure, two-panel videos with
      error maps (flat: vs the exact solution; curved: vs a 4x finer
      interface-aware run) (2026-09-17)
- [x] Deck culled to seven content slides per part, slide-by-slide passes,
      new title and thesis (2026-09-19)
- [x] Clip pass: re-rendered with the final palette, 1-D clip simplified,
      2-D still reduced to the reference wave and two error maps (2026-09-19)
- [x] Docs pass (2026-09-19)
- [x] Part 3, not in the talk: smooth-edged layer, spectral reference,
      ODE-continued seed stencils, knee experiment, notes with verified
      related work and a 2-D design note (2026-09-19)
- [x] Part 3 manuscript in `paper/`: drafted, assembled, packaged for arXiv
      with the acknowledgments and the AI-assistance disclosure (2026-09-20)
- [ ] Submit the manuscript to arXiv (math.NA) and tag `manuscript-v1`
- [ ] Rehearsal on Sep 29, then freeze and tag `talk-2026-09-30`

## License

Code: [MIT](LICENSE). Manuscript (`paper/`): [CC BY 4.0](paper/LICENSE).
Reference papers are the property of their respective authors and
publishers and are not redistributed here.
