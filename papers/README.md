# Reference papers

PDFs in this directory are **gitignored** (this repo is public, and the
manuscripts are not ours to redistribute). Run `./papers/fetch_papers.sh` from
the repo root to download the public ones; drop the dissertation in by hand.

| File | What it is | Source |
| --- | --- | --- |
| `openai-2026-finite-time-blowup-navier-stokes.pdf` | OpenAI, *Finite Time Blowup for Navier–Stokes*, 166 pp., PDF dated 2026-09-08. Claims alternatives (C) and (D) of the Clay problem statement. | [PDF](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf) · [blog post](https://openai.com/index/navier-stokes-solution/) · [Lean 4 certificates](https://github.com/openai/NavierStokesAndEuler) (Apache-2.0) |
| `openai-2026-finite-time-blowup-euler.pdf` | OpenAI, companion *Finite Time Blowup for the Euler Equation*, 57 pp., PDF dated 2026-09-08. Unforced, inviscid blowup from smooth compactly supported data. | [PDF](https://cdn.openai.com/pdf/315b36cd-ec98-4023-8342-93345194ece1/euler.pdf) |
| `alpoge-buckmaster-2026-euler-blowup-smooth-forcing.pdf` | L. Alpöge, T. Buckmaster, *Blowup for the Euler equations with smooth forcing*, 112 pp., PDF dated 2026-09-07. The parallel human–AI result (forced Euler). Companion IPM and Boussinesq papers at the same site. | [PDF](https://cims.nyu.edu/~tristanb/euler.pdf) · [Lean](https://github.com/tristanbuckmaster/fluid_lean) |
| `buckmaster-2026-statement.pdf` | T. Buckmaster, statement of 2026-09-07 on the results, the tools used, and the contacts with OpenAI (4 pp.). Primary source for the dispute slide. | [PDF](https://cims.nyu.edu/~tristanb/statement.pdf) |
| `clay-2000-fefferman-navier-stokes-problem-statement.pdf` | C. Fefferman, official Clay Millennium problem statement (6 pp.). Defines alternatives (A)–(D). | [PDF](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf) · [problem page](https://www.claymath.org/millennium/navier-stokes-equation/) |
| `martin-dissertation-2016-rbf-fd-interfaces.pdf` | B. Martin, *Application of RBF-FD to Wave and Heat Transport Problems in Domains with Interfaces*, PhD dissertation, CU Boulder Applied Mathematics, 2016 (ProQuest 10151046), 145 pp. **Not auto-fetched**: ProQuest serves it behind a JS viewer and CU Scholar blocks scripted access. Brad supplies his copy under this filename. | [ProQuest open view](https://www.proquest.com/openview/ae4d936114520c2d34e604adeda7d81a/1?pq-origsite=gscholar&cbl=18750) · [CU AMATH dissertation list](https://www.colorado.edu/amath/academics/doctoral-program/past-phd-dissertations) |
| `martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf` | B. Martin, B. Fornberg, A. St-Cyr, *Seismic modeling with radial basis function-generated finite differences*, Geophysics 80(4) T137–T146, 2015 (submitted preprint, 26 pp.). 2-D elastic RBF-FD background for the 2-D half of the demo. | [PDF (CU AMATH)](https://www.colorado.edu/amath/sites/default/files/attached-files/2015_mfstc_rbf-fd_2d_geophys_submitted_0.pdf) |
| `martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf` | B. Martin, B. Fornberg, *Seismic modeling with RBF-FD – a simplified treatment of interfaces*, J. Comput. Phys. 335, 828–845, 2017 (submitted preprint, 44 pp.). The interface treatment the dissertation builds on. | [PDF (CU AMATH)](https://www.colorado.edu/amath/sites/default/files/attached-files/2016_mf_rbf-fd_seismic_jcp_submitted.pdf) · [publisher](https://www.sciencedirect.com/science/article/abs/pii/S0021999117300815) |
| `tornberg-engquist-2006-regularization-wave-propagation-maa.pdf` | A.-K. Tornberg, B. Engquist, *Regularization for accurate numerical wave propagation in discontinuous media*, Methods Appl. Anal. 13(3) 247–274, 2006 (28 pp.). The regularised coefficients (1/a and 1/b linear across one cell) that the #69 cell-averaged comparator reproduces at δ = 0. | [PDF (open access)](https://www.intlpress.com/site/pub/files/_fulltext/journals/maa/2006/0013/0003/MAA-2006-0013-0003-a003.pdf) · [DOI](https://doi.org/10.4310/MAA.2006.v13.n3.a3) |
| `koene-wittsten-robertsson-2021-anti-aliasing-vs-equivalent-medium-arxiv.pdf` | E. F. M. Koene, J. Wittsten, J. O. A. Robertsson, *Finite-difference modeling of 2-D wave propagation in the vicinity of dipping interfaces: a comparison of anti-aliasing and equivalent medium approaches*, arXiv:2104.08206v2 (2021-11-04), 34 pp.; published as Geophys. J. Int. 229 (2022). The anti-aliased step and the low-pass filter (§3.2–3.3) that the #69 band-limited comparator follows. | [arXiv](https://arxiv.org/abs/2104.08206) |

Related, not downloaded: B. Martin, B. Fornberg, *Using RBF-FD to solve heat
transfer equilibrium problems in domains with interfaces*, Eng. Anal. Bound.
Elem. 79, 38–48, 2017.

## Checksums (as fetched 2026-09-17)

The OpenAI manuscript in particular may be revised. `fetch_papers.sh` warns
(does not fail) on a checksum mismatch so we notice a new version.
Re-checked against fresh downloads on 2026-09-19: all unchanged.

```
0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f  openai-2026-finite-time-blowup-navier-stokes.pdf
c1b5f27b1a64705cfaf1afceea513db5deedca8a18ca56ab32e7f86445a06d0c  clay-2000-fefferman-navier-stokes-problem-statement.pdf
4f4cee20bf0683e15c99c8776c6c8422b3840ac17f8ae517a46a73a7e66c8f51  martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf
dd862ff7e69830e5bdd8ee72abccfd0bbdb96c3a8374df8a3733af25c86b1a58  martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf
a0c234518e6c489e16996805023eb2e75c00b7c03455f7a3a5be2c124954bfdd  openai-2026-finite-time-blowup-euler.pdf
97ef408bff09b4f6ed9f3867734d1eb2245f3f34e6334b28136c84c02d0ae8d8  alpoge-buckmaster-2026-euler-blowup-smooth-forcing.pdf
8d7723941bcda2fa55c1e74faa6298e04c706d17ff8abd2ad01878039c621f9d  buckmaster-2026-statement.pdf
fc4d63910bd79bc506d5ac21382ccc707033cc3e4f27e31a784dbd39f4a1185f  tornberg-engquist-2006-regularization-wave-propagation-maa.pdf  (fetched 2026-09-20)
5a52ef3473ae16a1276712813179a11df9e6d163580ffa89430d50a1a66cf826  koene-wittsten-robertsson-2021-anti-aliasing-vs-equivalent-medium-arxiv.pdf  (fetched 2026-09-20)
a658c8b94547eb085eed68e1bd70cca00ed9c49eba8911d8412576d9945e96f2  martin-dissertation-2016-rbf-fd-interfaces.pdf  (hand-supplied 2026-09-17)
```
