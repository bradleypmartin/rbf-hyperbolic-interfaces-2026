# rbf-hyperbolic-interfaces-2026

Python ports of the interface-aware wave solvers from my 2016 CU Boulder
applied-math dissertation and the papers around it, and an extension of them
to material changes that are smooth but too steep for the grid, written up
for arXiv.

- **Part 2, the replication.** The 1-D wave equation through a heterogeneous
  layer with interface-aware fourth-order finite differences, and the 2-D
  elastic wave equation on a scattered node set with curved interfaces using
  radial basis function-generated finite differences (RBF-FD). Both are
  verified against reference solutions: standard stencils lose accuracy at a
  material boundary, the interface-aware ones do not. Built with Claude Code
  on 2026-09-17 from my MATLAB, which it restructures rather than
  transcribes.
- **Part 3, stiff but smooth.** An idea from my research that never got
  written up. A stencil crossing a material edge too steep for the grid to
  resolve is built from "seeds" continued through the edge by ODEs instead
  of monomials, on the same grid; the dissertation's jump construction is
  the limit of a vanishing edge width. Confirmed in 1-D, then in 2-D through
  flat and curved edges and at oblique incidence, and measured against the
  standing alternative of smoothing or averaging the medium (2026-09-19 to
  20). Derivation and results: [`docs/stiff-features.md`](docs/stiff-features.md).
- **The manuscript.** The write-up of Part 3 in [`paper/`](paper/README.md),
  with [`LITERATURE.md`](LITERATURE.md) as its novelty ledger: the claim is
  itemised there and every "not found" is tied to the search that produced
  it. Drafted, assembled and packaged for arXiv; not yet submitted (the
  endorsement is pending).

## Where it came from

Parts 2 and 3 were built inside
[`bradleypmartin/20260930-zd-ai-pdes-demo`](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo),
the material for a talk on AI and applied math given on 2026-09-30. Part 1
of that talk, on the Navier–Stokes blowup claims, stays there, as do the
slides and the clips. On 2026-09-22 Parts 2 and 3 moved here with their
history (#1):

- `git filter-repo` carried every commit that touched a path outside the
  talk. [`docs/split-commit-map.txt`](docs/split-commit-map.txt) maps each
  demo SHA to its commit here; an all-zero entry is a talk-only commit that
  was dropped.
- The demo tag
  [`part3-pre-split`](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo/tree/part3-pre-split)
  is the demo as it stood at the split.
- `demo#N` is an issue or pull request in the demo; a bare `#N` is this
  repository. [`docs/decisions-log.md`](docs/decisions-log.md) is the
  demo's outline minus the talk: the Part 2 record, what happened when and
  the decisions behind the code.

## Where the pull requests are

The pull requests stayed in the demo, with the adversarial Claude Sonnet
reviews recorded on them as comments or in their descriptions. The
manuscript's disclosure section calls them "pull requests against the
repository"; this is where to find them.

- **Part 3 and the manuscript:** the 20 merged pull requests numbered 35–77,
  [listed here](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-09-19T14%3A00%3A00-06%3A00..2026-09-21),
  under the epics
  [demo#27](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo/issues/27)
  (the seed stencils) and
  [demo#51](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo/issues/51)
  (the manuscript).
- **Part 2:** the 1-D port
  ([demo#4](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo/pull/4))
  and the 2-D chain
  ([demo#10](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo/pull/10)–[demo#16](https://github.com/bradleypmartin/20260930-zd-ai-pdes-demo/pull/16)).

## Quickstart

```sh
uv sync                                  # Python 3.13 venv with numpy / scipy / matplotlib
uv run pytest                            # 245 tests: convergence orders and analytic comparisons
./papers/fetch_papers.sh                 # public reference PDFs (gitignored), checksum-checked
(cd paper && tectonic main.tex)          # the manuscript → paper/main.pdf
```

Drivers in `scripts/` write figures and animations to `outputs/`
(gitignored). The Part 2 ones reproduce the dissertation's figures:

```sh
uv run python scripts/wave1d_convergence.py     # 1-D error vs resolution (dissertation Fig. 2-8)
uv run python scripts/wave1d_demo.py            # 1-D two-panel MP4 + still, naive vs aware
uv run python scripts/wave2d_nodes.py           # interface-fitted node set (Fig. 3-3)
uv run python scripts/wave2d_eigenvalues.py     # spectrum with and without hyperviscosity (Fig. 3-2)
uv run python scripts/wave2d_hyperviscosity.py  # error and stability vs hyperviscosity amplitude
uv run python scripts/wave2d_convergence.py     # 2-D error vs resolution, flat and curved (Fig. 3-5 / 3-8)
uv run python scripts/wave2d_demo.py            # 2-D two-panel MP4 + still; --amplitude 0.02 for curved
```

Part 3 and the manuscript:

```sh
uv run python scripts/wave1d_stiff.py            # 1-D knee plot, snapshot and seeds through a smooth edge
uv run python scripts/wave2d_stiff.py            # 2-D flat δ sweep, naive RBF-FD vs seeds, and a still
uv run python scripts/wave2d_stiff_eigenvalues.py  # 2-D spectra, seed vs naive stencils
uv run python scripts/paper_figures.py           # paper/figures/ from the results cache in paper/data/
uv run python scripts/paper_numbers.py           # every cache-backed number in main.tex, re-derived
```

The 2-D Part 3 sweeps take minutes to hours and cache their references and
seed operators under `outputs/`; [`CLAUDE.md`](CLAUDE.md) lists the flags
behind each experiment and its running time, and
[`paper/README.md`](paper/README.md) the build and the arXiv packaging.

## Layout

| Path | Contents |
| --- | --- |
| `src/rbf_hyperbolic_interfaces/` | Library. `wave1d/`: FD stencils across interfaces, RK4, exact ray-sum solution; for Part 3, smooth tanh edges, a Fourier pseudo-spectral reference, ODE-continued seed stencils and the coefficient treatments they are compared with. `wave2d/`: node sets, periodic kNN, Gaussian RBF-FD weights, interface-aware stencils, hyperviscosity, sparse elastic operators, RK4, plane-wave references, one-sided resampling; for Part 3, smooth flat and curved edges, elastic seeds marched along the normal, Fourier references and the coefficient treatments. Shared Fornberg weights, plotting palette, results cache, and the Part 3 figures and tables. |
| `scripts/` | Drivers for the figures, clips and results cache; `paper_figures.py` and `paper_numbers.py` for the manuscript |
| `tests/` | pytest suite (convergence and analytic checks) |
| `docs/` | `stiff-features.md` and `figures/` (Part 3), `decisions-log.md` (Part 2 record, what happened when, decisions), `paper-index.md` (page ranges per PDF), `split-commit-map.txt` (demo SHA → SHA here) |
| `paper/` | The Part 3 manuscript: `main.tex` → `main.pdf` (tectonic), `references.bib`, the results cache `data/`, `figures/`, `make_arxiv.py`; see its README |
| `LITERATURE.md` | The manuscript's novelty ledger |
| `papers/` | Index of reference papers with links and checksums, fetch script; PDFs are not committed |

## Reference papers

See [`papers/README.md`](papers/README.md) for the full table.

- B. Martin, *Application of RBF-FD to Wave and Heat Transport Problems in
  Domains with Interfaces*, PhD dissertation, CU Boulder, 2016
  ([ProQuest](https://www.proquest.com/openview/ae4d936114520c2d34e604adeda7d81a/1?pq-origsite=gscholar&cbl=18750))
- B. Martin, B. Fornberg, A. St-Cyr, *Seismic modeling with RBF-FD*,
  Geophysics 2015 ([preprint](https://www.colorado.edu/amath/sites/default/files/attached-files/2015_mfstc_rbf-fd_2d_geophys_submitted_0.pdf))
- B. Martin, B. Fornberg, *Seismic modeling with RBF-FD – a simplified
  treatment of interfaces*, J. Comput. Phys. 2017
  ([preprint](https://www.colorado.edu/amath/sites/default/files/attached-files/2016_mf_rbf-fd_seismic_jcp_submitted.pdf))
- A.-K. Tornberg, B. Engquist, *Regularization for accurate numerical wave
  propagation in discontinuous media*, Methods Appl. Anal. 2006
  ([PDF](https://www.intlpress.com/site/pub/files/_fulltext/journals/maa/2006/0013/0003/MAA-2006-0013-0003-a003.pdf)),
  and E. F. M. Koene, J. Wittsten, J. O. A. Robertsson, *Finite-difference
  modeling of 2-D wave propagation in the vicinity of dipping interfaces*,
  Geophys. J. Int. 2022 ([arXiv](https://arxiv.org/abs/2104.08206)): the
  coefficient treatments Part 3 is measured against

The original MATLAB implementations live in a separate repo,
[`bradleypmartin/MathGraduateResearchAndCourseWork`](https://github.com/bradleypmartin/MathGraduateResearchAndCourseWork),
and were used as a read-only reference for the Python ports here. `CLAUDE.md`
lists what was ported and what was not.

## Status

- [x] Part 2: 1-D and 2-D ports with reference solutions, convergence
      figures and clips (2026-09-17)
- [x] Part 3 in 1-D: smooth edges, spectral reference, seed stencils, the
      knee experiment (2026-09-19)
- [x] Part 3 in 2-D: flat edges, the δ sweep, oblique incidence, curved
      edges; the coefficient treatments measured in 1-D and 2-D
      (2026-09-19 to 20)
- [x] Manuscript drafted, assembled and packaged for arXiv (2026-09-20)
- [x] Moved here from the talk repository with its history (#1, 2026-09-22)
- [ ] Every manuscript figure, table and number regenerates from this repo
      alone (#6)
- [ ] Submit to arXiv (math.NA, cross-list physics.comp-ph) and tag
      `manuscript-v1` here

## License

Code: [MIT](LICENSE). Manuscript (`paper/`): [CC BY 4.0](paper/LICENSE).
Reference papers are the property of their respective authors and
publishers and are not redistributed here.
