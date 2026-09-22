# Reference papers

PDFs in this directory are **gitignored** (this repo is public, and the
manuscripts are not ours to redistribute). Run `./papers/fetch_papers.sh` from
the repo root to download the public ones; drop the dissertation in by hand.

| File | What it is | Source |
| --- | --- | --- |
| `martin-dissertation-2016-rbf-fd-interfaces.pdf` | B. Martin, *Application of RBF-FD to Wave and Heat Transport Problems in Domains with Interfaces*, PhD dissertation, CU Boulder Applied Mathematics, 2016 (ProQuest 10151046), 145 pp. **Not auto-fetched**: ProQuest serves it behind a JS viewer and CU Scholar blocks scripted access. Brad supplies his copy under this filename. | [ProQuest open view](https://www.proquest.com/openview/ae4d936114520c2d34e604adeda7d81a/1?pq-origsite=gscholar&cbl=18750) · [CU AMATH dissertation list](https://www.colorado.edu/amath/academics/doctoral-program/past-phd-dissertations) |
| `martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf` | B. Martin, B. Fornberg, A. St-Cyr, *Seismic modeling with radial basis function-generated finite differences*, Geophysics 80(4) T137–T146, 2015 (submitted preprint, 26 pp.). 2-D elastic RBF-FD background for the `wave2d` port. | [PDF (CU AMATH)](https://www.colorado.edu/amath/sites/default/files/attached-files/2015_mfstc_rbf-fd_2d_geophys_submitted_0.pdf) |
| `martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf` | B. Martin, B. Fornberg, *Seismic modeling with RBF-FD – a simplified treatment of interfaces*, J. Comput. Phys. 335, 828–845, 2017 (submitted preprint, 44 pp.). The interface treatment the dissertation builds on. | [PDF (CU AMATH)](https://www.colorado.edu/amath/sites/default/files/attached-files/2016_mf_rbf-fd_seismic_jcp_submitted.pdf) · [publisher](https://www.sciencedirect.com/science/article/abs/pii/S0021999117300815) |
| `tornberg-engquist-2006-regularization-wave-propagation-maa.pdf` | A.-K. Tornberg, B. Engquist, *Regularization for accurate numerical wave propagation in discontinuous media*, Methods Appl. Anal. 13(3) 247–274, 2006 (28 pp.). The regularised coefficients (1/a and 1/b linear across one cell) that the demo#69 cell-averaged comparator reproduces at δ = 0. | [PDF (open access)](https://www.intlpress.com/site/pub/files/_fulltext/journals/maa/2006/0013/0003/MAA-2006-0013-0003-a003.pdf) · [DOI](https://doi.org/10.4310/MAA.2006.v13.n3.a3) |
| `koene-wittsten-robertsson-2021-anti-aliasing-vs-equivalent-medium-arxiv.pdf` | E. F. M. Koene, J. Wittsten, J. O. A. Robertsson, *Finite-difference modeling of 2-D wave propagation in the vicinity of dipping interfaces: a comparison of anti-aliasing and equivalent medium approaches*, arXiv:2104.08206v2 (2021-11-04), 34 pp.; published as Geophys. J. Int. 229 (2022). The anti-aliased step and the low-pass filter (§3.2–3.3) that the demo#69 band-limited comparator follows. | [arXiv](https://arxiv.org/abs/2104.08206) |

Related, not downloaded: B. Martin, B. Fornberg, *Using RBF-FD to solve heat
transfer equilibrium problems in domains with interfaces*, Eng. Anal. Bound.
Elem. 79, 38–48, 2017.

## Checksums (as fetched 2026-09-17)

`fetch_papers.sh` warns (does not fail) on a checksum mismatch so we notice
a revised upload.
Re-checked against fresh downloads on 2026-09-19: all unchanged.

```
4f4cee20bf0683e15c99c8776c6c8422b3840ac17f8ae517a46a73a7e66c8f51  martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf
dd862ff7e69830e5bdd8ee72abccfd0bbdb96c3a8374df8a3733af25c86b1a58  martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf
fc4d63910bd79bc506d5ac21382ccc707033cc3e4f27e31a784dbd39f4a1185f  tornberg-engquist-2006-regularization-wave-propagation-maa.pdf  (fetched 2026-09-20)
5a52ef3473ae16a1276712813179a11df9e6d163580ffa89430d50a1a66cf826  koene-wittsten-robertsson-2021-anti-aliasing-vs-equivalent-medium-arxiv.pdf  (fetched 2026-09-20)
a658c8b94547eb085eed68e1bd70cca00ed9c49eba8911d8412576d9945e96f2  martin-dissertation-2016-rbf-fd-interfaces.pdf  (hand-supplied 2026-09-17)
```
