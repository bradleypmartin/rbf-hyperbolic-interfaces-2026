# Reference papers

PDFs in this directory are **gitignored** (this repo is public, and the
manuscripts are not ours to redistribute). Run `./papers/fetch_papers.sh` from
the repo root to download the public ones; drop the dissertation in by hand.

| File | What it is | Source |
| --- | --- | --- |
| `openai-2026-finite-time-blowup-navier-stokes.pdf` | OpenAI, *Finite Time Blowup for Navier–Stokes*, 166 pp., PDF dated 2026-09-08. Claims alternatives (C) and (D) of the Clay problem statement. | [PDF](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf) · [blog post](https://openai.com/index/navier-stokes-solution/) · [Lean 4 certificates](https://github.com/openai/NavierStokesAndEuler) (Apache-2.0) |
| `openai-2026-finite-time-blowup-euler.pdf` | OpenAI, companion *Finite Time Blowup for the Euler Equation* (unforced, inviscid). Optional; not fetched by default. | [PDF](https://cdn.openai.com/pdf/315b36cd-ec98-4023-8342-93345194ece1/euler.pdf) |
| `clay-2000-fefferman-navier-stokes-problem-statement.pdf` | C. Fefferman, official Clay Millennium problem statement (6 pp.). Defines alternatives (A)–(D). | [PDF](https://www.claymath.org/wp-content/uploads/2022/06/navierstokes.pdf) · [problem page](https://www.claymath.org/millennium/navier-stokes-equation/) |
| `martin-dissertation-2016-rbf-fd-interfaces.pdf` | B. Martin, *Application of RBF-FD to Wave and Heat Transport Problems in Domains with Interfaces*, PhD dissertation, CU Boulder Applied Mathematics, 2016 (ProQuest 10151046), 145 pp. **Not auto-fetched**: ProQuest serves it behind a JS viewer and CU Scholar blocks scripted access. Brad supplies his copy under this filename. | [ProQuest open view](https://www.proquest.com/openview/ae4d936114520c2d34e604adeda7d81a/1?pq-origsite=gscholar&cbl=18750) · [CU AMATH dissertation list](https://www.colorado.edu/amath/academics/doctoral-program/past-phd-dissertations) |
| `martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf` | B. Martin, B. Fornberg, A. St-Cyr, *Seismic modeling with radial basis function-generated finite differences*, Geophysics 80(4) T137–T146, 2015 (submitted preprint, 26 pp.). 2-D elastic RBF-FD background for the 2-D half of the demo. | [PDF (CU AMATH)](https://www.colorado.edu/amath/sites/default/files/attached-files/2015_mfstc_rbf-fd_2d_geophys_submitted_0.pdf) |
| `martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf` | B. Martin, B. Fornberg, *Seismic modeling with RBF-FD – a simplified treatment of interfaces*, J. Comput. Phys. 335, 828–845, 2017 (submitted preprint, 44 pp.). The interface treatment the dissertation builds on. | [PDF (CU AMATH)](https://www.colorado.edu/amath/sites/default/files/attached-files/2016_mf_rbf-fd_seismic_jcp_submitted.pdf) · [publisher](https://www.sciencedirect.com/science/article/abs/pii/S0021999117300815) |

Related, not downloaded: B. Martin, B. Fornberg, *Using RBF-FD to solve heat
transfer equilibrium problems in domains with interfaces*, Eng. Anal. Bound.
Elem. 79, 38–48, 2017.

## Checksums (as fetched 2026-09-17)

The OpenAI manuscript in particular may be revised. `fetch_papers.sh` warns
(does not fail) on a checksum mismatch so we notice a new version.

```
0e779481c4da40bd28d1e642e1d8ca57447d129610df28dfa5a11e9af8ae228f  openai-2026-finite-time-blowup-navier-stokes.pdf
c1b5f27b1a64705cfaf1afceea513db5deedca8a18ca56ab32e7f86445a06d0c  clay-2000-fefferman-navier-stokes-problem-statement.pdf
4f4cee20bf0683e15c99c8776c6c8422b3840ac17f8ae517a46a73a7e66c8f51  martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf
dd862ff7e69830e5bdd8ee72abccfd0bbdb96c3a8374df8a3733af25c86b1a58  martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf
a658c8b94547eb085eed68e1bd70cca00ed9c49eba8911d8412576d9945e96f2  martin-dissertation-2016-rbf-fd-interfaces.pdf  (hand-supplied 2026-09-17)
```
