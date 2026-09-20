# LITERATURE.md — the novelty ledger for the seed stencils

> **STATUS: LITERATURE AND NOVELTY PASS #53 RUN 2026-09-20 (P1–P5 below).**
> Upgrades `docs/stiff-features.md` §3, the 2026-09-19 afternoon scan, into
> the house ledger: the claim of Part 3 (issue #27, manuscript epic #51) is
> itemised in §1 and every item sits in the bucket the record supports;
> §2 logs what was searched, with what, and what came back; §3 is the
> verification rule; §4 the standing hazards; §6 the wording the manuscript
> may use. `paper/references.bib` carries no unverified entry (P5).
>
> **Headline.** Every *ingredient* of the construction is classical and is
> cited, not claimed (§1a). What survived the pass as ours (§1b) is the
> combination: a stencil basis for the time-domain wave equation continued
> through a **given** sub-grid smooth edge by ODEs, on an otherwise
> unchanged equispaced FD or scattered-node RBF-FD scheme, together with the
> identification of the jump stencils as its δ → 0 limit and the empirical
> rule *seed when δ ≤ h*. The standing alternative, smoothing or averaging
> the medium (§1a K6), **was run head to head** in #69 (notes §2.1 and
> §5.7: cell means over one and two cells, band-limited coefficients and
> the widened edge, in 1-D and on the scattered nodes), as our
> implementation of each treatment on one problem at one contrast; §6b
> words what that licenses, and the manuscript may not claim superiority
> over those methods in general.

**Maintenance rule** (house convention, mirrored from weil-positivity-lab,
bolza-bending and dirichlet-bridge): update this file on **any** new
novelty claim, any new comparison method, and any new regime measured
(order, contrast, dimension). Before any positive is called unrecorded, run
a directed refuter pass and log it in §2 (date, tool, query, result).
Re-run P3's Q1 (the quasi-Trefftz group posted four preprints in 2025–26)
and P4's seismology sweep before any external circulation, including the
arXiv upload (#62).

**Cite, don't claim.** The manuscript cites this ledger's §1a items as
prior work and words §1b items as "no prior instance found", never as
"novel" or "first". Every "not found" below is a keyword-search negative
tied to the queries that produced it (§4).

Verification statuses (rule in §3): **[V]** = verified against a fetched
primary source (Crossref API record of the DOI, arXiv API record,
publisher or repository page, archive.org or zbMATH Open record for pre-DOI
books, the PDF itself), named in §2 or in the entry's comment in
`paper/references.bib`. **[S]** = seen in search results or a page summary
only; may be discussed here with hedging, may **not** enter the bib. A
metadata [V] does not certify content: where a paper's *claims* were read
only at abstract level or not at all, §4 says so.

## 1. The honest claim, itemised

### 1a. Cite, don't claim (trodden ground)

**K1. The x-like seed: constant flux through the feature.** The first seed
after the constant, K φ₁′ = K(x_e), is the two-point harmonic mean of
Tikhonov & Samarskii's homogeneous schemes (USSR CMMP 1 (1962) 5–67 [V];
textbook form Samarskii, *The Theory of Difference Schemes*, Dekker 2001
[V]), the special shape functions of Babuška–Osborn (SINUM 20 (1983) [V])
and Babuška–Caloz–Osborn (SINUM 31 (1994) [V]), the cell-problem bases of
the multiscale FEM of Hou–Wu (JCP 134 (1997) [V]) and the harmonic
coordinates of Owhadi–Zhang (CPAM 60 (2007) [V]). The same physics is the
practice of seismic FD: volume harmonic averaging of moduli and arithmetic
averaging of density (Moczo, Kristek, Vavryčuk, Archuleta & Halada, BSSA 92
(2002) 3042–3066 [V, author list from the paper's first page]; the
orthorhombic representation of Kristek, Moczo, Chaljub & Kristekova, GJI
208 (2017) [V]; the textbook Moczo–Kristek–Gális, CUP 2014 [V]), of thin
layers (Backus, JGR 67 (1962) [V]), and of FDTD subpixel smoothing, which
uses the harmonic mean for the field component normal to the interface
(Farjadpour et al., Opt. Lett. 31 (2006) [V]). *Framing:* the manuscript
may say the constant-flux seed is the harmonic-mean idea in a five-point,
fourth-order stencil; it may not present it as new.

**K2. Stencil bases from local solutions of the operator.** Fitted operators
for singularly perturbed problems: Allen–Southwell (QJMAM 8 (1955) [V]),
Il'in (Math. Notes 6 (1969) [V]), Scharfetter–Gummel (IEEE TED 16 (1969)
[V]), surveyed as "operator-fitted methods" in Roos–Stynes–Tobiska (Springer
2008, §3.5.1 [V]; the section was read in the Charles University PDF on
2026-09-19). Exact and nonstandard schemes: Mickens (World Scientific,
Crossref-dated Dec 1993 [V]); the "exact three-point schemes" for
self-adjoint second-order ODEs, Vizvari, Sari, Klincsik & Odry, Adv. Differ.
Equ. 2020:497 [V] (the DOI-only entry of the notes, now identified).
L-splines (Schultz–Varga, Numer. Math. 10 (1967) [V]) and generalised
divided differences (Mühlbach, JAT 9 (1973) [V]; the weight recurrence in a
finite-difference setting, Gerolymos, AMC 219 (2012) [V]). *Framing:*
"stencils exact on the local solutions of the operator" is a 1950s–60s
idea; the seeds apply it to the time-polynomial chain of a wave equation.

**K3. Basis functions that are polynomial-in-time solutions of the wave
equation.** The space–time Trefftz DG methods: Kretzschmar, Moiola, Perugia
& Schnepp (IMA JNA 36 (2016) [V]); Banjai, Georgoulis & Lijoka (SINUM 55
(2017) [V]); for the first-order acoustic system, the form the 1-D chain
is in, Moiola & Perugia (Numer. Math. 138 (2018) [V]). Their extension to
variable coefficients, the polynomial quasi-Trefftz spaces:
Imbert-Gérard, Moiola & Stocker, Math. Comp. 92 (2023) [V] (wave equation,
piecewise-smooth coefficients); Imbert-Gérard, Moiola, Perinati & Stocker,
IMA JNA 45 (2025) [V] (elliptic; the seeded bib had this as a 2024
three-author preprint); Imbert-Gérard, "Local Taylor-based polynomial
quasi-Trefftz spaces for scalar linear equations", arXiv:2505.18480 (2025)
[V as preprint]; the generalised-plane-wave quasi-Trefftz spaces for
inhomogeneous media of Fontana & Imbert-Gérard (Forum Acusticum 2025,
Crossref 10.61782/fa.2025.0967 [V, time-harmonic, not entered in the bib]).
*Framing obligation:* the seeds are exactly the t = 0 traces of the
polynomial Trefftz spaces, and the manuscript must say so in §3. The
quasi-Trefftz construction is Taylor-based by name and design; notes §1.5
shows why that route stops at a radius of order δ, which is the one
technical difference the manuscript is entitled to state.

**K4. Chebyshev systems and disconjugacy.** Pólya (Trans. AMS 24 (1922)
[V, Crossref record; the AMS PDF was not fetched]); Karlin & Studden,
*Tchebycheff Systems* (Interscience 1966 [V, archive.org + zbMATH]); Coppel,
*Disconjugacy* (Springer LNM 1971 [V]). The ECT structure theorem the notes
use is stated as Theorem 2 of R. A. Zalik, "Another look at Chebyshev
systems" (PDF at webhome.auburn.edu/~zalikri/fv/t.pdf, fetched 2026-09-20,
6 pp., which points to Karlin–Studden pp. 376–379) — **[S]**: no journal or
zbMATH record was found, so the manuscript cites Karlin–Studden and Coppel
and not Zalik.

**K5. The interface error of standard schemes, and the jump methods.** That
standard FD loses order at a coefficient jump is documented from Brown
(Math. Comp. 42 (1984) [V]) and Fornberg (Geophysics 53 (1988) [V]) through
Muir et al. (Geophysics 57 (1992) [V]), Symes & Vdovina (Comput. Geosci. 13
(2009) [V]) and Vishnevsky et al. (Geophysics 79 (2014) [V]); for wave
speeds with discontinuous *derivatives*, Erickson, O'Reilly & Nordström
(JSC 81 (2019) [V metadata; content not read at source, see §4]). The
families that take the jump as given and build it into the scheme: the
immersed interface method (LeVeque & Li, SINUM 31 (1994) [V]; Zhang &
LeVeque for acoustic waves, Wave Motion 25 (1997) [V]; recent high-order
forms Jeong, Ha & You, JCP 426 (2021) [V] and Sabatini et al., Phys. Fluids
35 (2023) [V]); the explicit simplified interface method (Piraux & Lombard,
JCP 168 (2001) [V]; Lombard & Piraux, JCP 195 (2004) [V]);
summation-by-parts treatments with the material interface at a block
boundary (Mattsson & Nordström, JCP 220 (2006) [V]; Mattsson, Ham &
Iaccarino, JCP 227 (2008) [V]; Duru & Virta, JCP 279 (2014) [V]; Granath &
Wang, JCP 524 (2025) [V]; Gustafsson & Wahlund, SISC 26 (2004) [V]); and
the interface-aware RBF-FD stencils this work generalises (Martin, Fornberg
& St-Cyr, Geophysics 80 (2015) [V]; Martin & Fornberg, JCP 335 (2017) [V];
the dissertation [V, PDF]). RBF-FD background: Fornberg (Math. Comp. 51
(1988) [V]), Fornberg & Lehto (JCP 230 (2011) [V]), Fornberg & Flyer (SIAM
primer 2015 [V]; Acta Numerica 24 (2015) [V]), Flyer, Fornberg, Bayona &
Barnett (JCP 321 (2016) [V]), Bayona, Flyer, Fornberg & Barnett (JCP 332
(2017) [V]); RBF-FD for waves in heterogeneous media after 2017, Močnik
Berljavac et al. (Comput. Geosci. 153 (2021) [V]); the most recent RBF-FD
interface work found, elliptic, Cheng, Ju & Zhang (JSC 108 (2026) [V]).

**K6. The standing alternative: change the medium, not the stencil.** In
seismic FD and FDTD the interface is routinely replaced by an averaged
medium (Backus 1962; Muir et al. 1992; Moczo et al. 2002; Kristek et al.
2017; Jiang & Zhang, GJI 239 (2024) [V], the efficient implementation of
equivalent-medium parametrisation; Symes & Terentyev, SEG 2009 [V], subgrid
modelling by mass lumping), by a homogenised one (Capdeville, Guillot &
Marigo, GJI 181 (2010) [V] and 182 (2010) [V], non-periodic homogenization
for the seismic wave equation), or by a *smoothed* one: the regularisation
line of Tornberg & Engquist (JSC 19 (2003) [V]; *Regularization for
accurate numerical wave propagation in discontinuous media*, MAA 13 (2006)
247–274 [V, fetched and read in full for #69, `papers/`: the Yee scheme "is
improved from first to second order by modifying the material coefficients
close to the material interface", and the fourth-order scheme keeps "a
second order error component originating from the discontinuities"; §4.1
eq. 18–22 fix the modification as 1/a and 1/b linear across one cell, i.e.
cell means of compliance and density, and §5 reaches second order for
their staggered fourth-order scheme only with its temporal correction
terms masked within 3h/2 of the jump]; JCP 227 (2008) [V]),
FDTD subpixel smoothing (Farjadpour et al. 2006), and the anti-aliased
interfaces that Koene, Wittsten & Robertsson (GJI 229 (2022) [V;
arXiv:2104.08206v2 fetched and §3.2–3.4 read for #69, `papers/`: density
and compliance band-limited to the grid Nyquist, the anti-aliased step
½ + Si(πz/h)/π, the windowed low-pass filter]) compare head to head with
Schoenberg–Muir equivalent media:
anti-aliasing wins in acoustic media, the equivalent medium in elastic
media. *Framing the manuscript owes (the #53 issue's fourth sweep):* in all
of these the smoothing width or averaging cell is a **numerical** choice
tied to h, and the price is a second-order error component at the
interface. The seed problem is the inverse: δ is a property of the medium,
the user chooses h, and the aim is the scheme's full order through the
edge for every δ ≤ h. So (i) the "twilight zone" of notes §1.1 is
precisely the medium these methods manufacture and then accept
second-order error from; (ii) a seeded scheme applied to such a smoothed
medium would remove that error component, which is a hypothesis, not a
result; (iii) **the head-to-head was run in #69** (notes §2.1 and §5.7,
manuscript §4 and §6): Tornberg–Engquist's regularised coefficients as
cell means of compliance and density over one cell (their eq. 18 and 22;
on the 1-D cell-centred grid a jump sits on a cell boundary, so this
changes nothing at δ = 0) and over two, the Mittet/Koene band-limited
coefficients as a windowed sinc (cutoff 1.1π/h, half-width 2.5h), and the
widened tanh edge (the "regularise a itself" case they warn about), all
through the unchanged scheme against the true-δ reference; in 2-D the wave
moduli harmonically and the density arithmetically over a square, isotropy
kept, and the separable low-pass. *1-D:* at a jump only the two-cell mean
lifts the order, to two; through an unresolved edge the one-cell mean and
the band-limited medium gain 1.25–1.6× on sampling, the two-cell mean
5.7–11×; on a resolved edge every treatment at its prescribed width is
second order, 47–530× above the fourth-order sampled scheme at n = 1600;
the seeds are 3.2× below the best treatment on the coarsest grid, 37× at
h = 2δ and 146–730× at and past the knee, and no treatment reaches their
order. *2-D, flat and curved:* the widened edge is 5–120× worse than
sampling wherever it acts; through δ = 0.0025 the one-cell mean is level
with sampling at 2500 nodes and 2.1–3.6× below it at 10,000–19,600, with
the seeds 3.1–5.9× below the one-cell mean; the band-limited coefficients
gain at most 1.7×, the two-cell mean is above sampling at every n; where
the knee is inside the sweep every treatment crosses above sampling at
h ≈ 1.4–2δ, before the seeds. The Schoenberg–Muir anisotropic medium (T3)
was not built (no anisotropic operator in the port). Scope: one scheme per
dimension, one contrast, one pulse, our implementation of each treatment;
the sources analyse and run them on staggered Cartesian grids.

### 1b. Ours as scoped (survived the pass; framing obligations noted)

**O1. ODE-continued seeds through a given sub-grid smooth edge on an
unchanged scheme.** A five-point equispaced FD4 stencil (1-D) or a
19-node RBF-FD stencil on scattered nodes (2-D) whose basis is the t = 0
profiles of time-polynomial solutions of the *variable-coefficient* wave
equation, obtained by marching the chain L φ_k = k(k−1) c_e² φ_{k−2}
through the edge as a first-order ODE system in the normal coordinate,
with the rest of the scheme (nodes, time stepper, hyperviscosity) left
standard. **No prior instance found** (P2: Q2, Q3, Q5, Q13, Q15, Q16, Q19,
Q20; P3: Q1, Q7, Q8, Q13, Q14; web W3, W5, W7, W8, W10). Nearest
relatives, each different in mechanism: quasi-Trefftz spaces (K3; Taylor at
a point, DG), Owhadi–Zhang 2008 (harmonic coordinates precomputed globally,
implicit time stepping, FEM, no scale separation; the closest 2-D relative),
fitted and exact schemes (K2; steady or ODE), Tornberg–Engquist 2006
(smooths the medium instead). *Obligations:* name K1–K3 as the ingredients
in the same paragraph; word the claim as "no prior instance found", with
this ledger as the reference for the search.

**O2. The jump as the δ → 0 limit, and one construction from jump to
resolved edge.** The dissertation's translated Taylor basis (JCP 2017's
square-matrix construction) is the weak-sense seed chain
{1} ⊂ ker L ⊂ {Lg = const} ⊂ ker L² ⊂ …, and the seed weights converge to
`interface_weights` at first order in δ (notes §1.3; tested). Since the
translated basis exists only in own work (P1), the identification is
ours; the "one construction" observation (algebra for a jump, ODE march
for a sub-grid edge, Fornberg's weights recovered as δ/h → ∞ in the sense
that both are fourth order on solutions, notes §1.5) follows from it and
from K3. *Obligation:* state the δ/h → ∞ end as "indistinguishable end to
end, though the weights differ at first order in h/δ", as the notes do.

**O3. Well-posedness for every δ, including the kinked limit.** Each span in
the chain is the kernel of an operator in Pólya form, hence an ECT system
(K4), hence the stencil solve is nonsingular on any distinct nodes. This
is an *application* of textbook theorems and the manuscript states it as a
proposition proved by citation (Pólya 1922; Karlin–Studden 1966; Coppel
1971), as #56 asks. The Newton-form remainder O(h⁴ ∂ₓL² u) with a
δ-independent constant on solutions (notes §1.4) is likewise Mühlbach
1973 applied.

**O4. The knee at h = δ, and its removal, measured.** Order loss of standard
FD through an under-resolved coefficient is documented for jumps (K5) and
for non-smooth wave speeds (Erickson et al. 2019, metadata only). The
δ-parametrised knee of FD4 against a pseudo-spectral reference, with the
seed stencils flat across it at the jump case's errors to three digits
(notes §2), is ours as scoped: FD4, one contrast (c 1 → 2), δ ∈ {0.0025,
0.01, 0.04}, n ≤ 1600.

**O5. The two-dimensional construction and the rule.** The elastic seeds of
a straight edge as triangular systems of 1-D ODEs in the normal
coordinate with polynomial tangential dependence and tractions as flux
variables (notes §4); seeded 19-node rows with the Δ³ hyperviscosity on
the naive footprint annihilating the seeds; stability at the standard γ
at every δ; the flat δ sweep, the oblique train and the curved edge via
the true normal (route (a)); the rule **seed when δ ≤ h**. Ours as scoped:
degree 3, one contrast, ≤ 19,600 nodes, tanh edges, doubly periodic,
compared against naive coefficient sampling and against our implementation
of the cell-averaged and band-limited coefficients and the widened edge
(§1a K6 (iii), discharged as scoped by #69). The 2-D relative to name is
Owhadi–Zhang 2008.

**O6. Diagnostics.** The spurious u at normal incidence as a 10× sharper
indicator of edge error than v (notes §5.2), and the converted-S floor
that caps every scheme near order 2.5 in the oblique runs (notes §5.5), are
methodological remarks; no claim rests on them and none was searched.

### 1c. Unswept (no claim may cite this ledger for support)

- Three dimensions; contrasts other than c 1 → 2; orders above FD4 /
  degree 3; bands thinner than a stencil; variable Lamé parameters — own
  exclusions (#27), not literature gaps.
- Computational electromagnetics beyond the two FDTD entries (conformal and
  contour-path FDTD, dielectric-interface FDTD analyses): W11 only.
- Finite-volume and lattice-Boltzmann treatments of discontinuous
  coefficients; spectral-element and mass-lumped FEM interface literature
  beyond the dissertation's ch. 1.
- Russian-language stiff-scheme literature beyond Tikhonov–Samarskii and
  Il'in (mathnet.ru unreachable today); Chinese-language geophysics.
- The citation graph of Martin & Fornberg 2017: no Scholar-style
  "cited by" surface was available; W4 and W9 found only own work and the
  Takekawa mesh-free FD line [S].

## 2. Refuter-pass log

All fetches 2026-09-20. Tools: Crossref REST API
(`api.crossref.org/works/<doi>`, `query.bibliographic`), arXiv API
(`export.arxiv.org/api/query`), zbMATH Open API, archive.org metadata API,
`curl`, Claude Code's WebSearch (US) and WebFetch, `pdftotext` on the PDFs
in `papers/` and on two PDFs fetched during the pass (Moczo et al. 2002
from ig.cas.cz; Tornberg–Engquist 2006 from intlpress.com, open access).
Query strings are verbatim; arXiv counts are the API's result counts.

### P1 — own work

- **Read:** the dissertation's title page (B. P. Martin, Department of
  Applied Mathematics, University of Colorado, 2016; committee Fornberg,
  Flyer, Julien, Martinsson, Meyer) and its 47-entry bibliography (PDF pp.
  125–); JCP 2017 preprint pp. 2–4 (abstract, introduction, the six-point
  advantages list) and pp. 42–44 (17 references); JCP 2017 pp. 8–33
  grepped for the treatment of smoothly variable parameters.
- **Finding 1.** The 2017 paper handles "smoothly-variable model
  parameters on either side of an interface" by a fourth-degree Taylor
  expansion of the wave speed about the interface (its eq. 15) fed into the
  continuity matrices. That is the Taylor route of notes §1.5, and the only
  place own work touched smooth variation near an interface. The seeds do
  not appear in the 2015, 2016 or 2017 work.
- **Finding 2.** The dissertation's ch. 1 and the JCP introduction give the
  K5/K6 map from the author's own 2016 vantage: Fornberg 1988 [5]/[4],
  Symes–Vdovina [3]/[1], Vishnevsky [4]/[2], Zhang–LeVeque [8]/[7],
  Lombard–Piraux [11]/[9], Muir et al. [9]/[5] and Symes–Terentyev [10]/[6]
  ("we need not modify any material parameters near the interface, as done
  in [5] and [6]"). All entered [V].
- **Finding 3.** The RBF-FD lineage (Fornberg 1988 weights; Fornberg–Lehto
  2011; Fornberg–Flyer 2015 ×2; Flyer et al. 2016; Bayona et al. 2017) [V]
  by Crossref; the EABE 2017 heat paper's DOI recovered by Crossref query
  (10.1016/j.enganabound.2017.03.005).

### P2 — material interfaces in wave simulation, 2016–2026

arXiv API:
- Q2 `abs:"RBF-FD" AND abs:interface` — 1 (surfaces; irrelevant).
- Q3 `abs:"RBF-FD" AND abs:(wave OR seismic OR elastic)` — 9: Močnik
  Berljavac et al. 2021 (heterogeneous acoustic, entered), Mishra et al.
  2018 hybrid kernels [S], Londoño & Rodríguez-Cortés 2022 Helmholtz FWI
  [S]; nothing interface-aware.
- Q4 `abs:"immersed interface" AND abs:(wave OR acoustic OR elastic)` — 12:
  the Lombard/Chiavassa line (poroelastic, viscoelastic, Biot) [S]; Asghar
  et al. 2024 IIM convergence in elasticity [S].
- Q11 `abs:("summation by parts" OR "summation-by-parts") AND abs:interface
  AND abs:wave` — 17: grid-interface papers (Wang–Virta–Kreiss 2016,
  Wang–Petersson 2019, Zhang–Wang–Petersson 2021, Kozdon–Wilcox 2016,
  Gao et al. 2018, Almquist–Dunham 2020, Eriksson 2022) [V metadata for the
  first two via Crossref, not entered: non-conforming *grid* interfaces,
  not material ones].
- Q16 `abs:interface AND abs:"finite difference" AND abs:(elastic OR
  acoustic) AND abs:("high-order" OR "high order" OR "fourth-order") AND
  abs:(discontinuous OR jump)` — 5: Granath & Wang 2025 (entered).
- Q17 `abs:"discontinuous Galerkin" AND abs:wave AND abs:(interface OR
  "material discontinuity") AND abs:(elastic OR acoustic)` — 16:
  elasto-acoustic DG (Appelö–Wang 2018, Antonietti et al., Duru et al.,
  Guo–Acosta–Chan) [S]; coupled-media interfaces, not sub-grid coefficient
  variation.
- Q22 `abs:"RBF-FD" AND abs:(polynomial OR "polyharmonic") AND
  abs:(augmentation OR augmented OR "role of polynomials")` — 9: stencil
  size and stability studies (Kosec group; Shankar–Fogelson 2018
  hyperviscosity) [S].

WebSearch:
- W3 `immersed interface method high-order finite difference elastic wave
  equation material discontinuity 2020 … 2025 fourth order` → Sabatini et
  al. 2023 (arbitrary-order IIM, acoustic and elastic) and Jeong–Ha–You 2021
  (entered [V] via Crossref); Zhang & Wang 2021 SINUM nonconforming
  interfaces [S]; a search-summary statement that the IIM "is known to have
  numerical dispersion and instability even with moderate changes in
  material properties", motivating the ESIM — [S], not to be quoted without
  reading Lombard–Piraux.
- W4 `RBF-FD "interface" discontinuous coefficients elliptic OR wave
  "radial basis function-generated finite differences" 2019 … 2025` →
  Cheng–Ju–Zhang 2026 (entered); a 2019 weak-form RBF-FD / RBF-PUM elliptic
  interface paper [S, not opened]; otherwise own work.
- W9 `mesh-free finite difference OR RBF-FD seismic elastic wave modeling
  material interface 2018 … 2024 Takekawa OR "meshless" interface treatment`
  → the Takekawa–Mikada mesh-free FD line (frequency-domain elastic, free
  surface) [S]; own work.
- W5 (task vocabulary) `"unresolved" OR "under-resolved" material interface
  finite-difference wave propagation accuracy order reduction "interface
  error" 2018..2026` → Koene–Wittsten–Robertsson 2022 (entered),
  Mattsson–Nordström 2006 (entered), Almquist–Dunham 2020 [S], a 2025 Petroleum
  Science fluid–solid FD paper [S].

**Verdict.** After 2017 the interface-aware wave literature moved along
arbitrary-order IIM, SBP-SAT with the material interface at a block
boundary, and DG; RBF-FD for waves grew but no RBF-FD *material-interface*
wave paper other than own was found, and the RBF-FD interface work found is
elliptic. **No prior instance found** of a scheme that treats an interface
smooth on a sub-grid scale as an object distinct from a jump. Queries above.

### P3 — stiff and sub-grid features, any date

arXiv API:
- Q1 `all:"quasi-Trefftz"` — 16: the full line 2020–2026 (Imbert-Gérard
  2020/2021 GPW; IGMS 2023; IGMPS 2025; Imbert-Gérard 2025 local Taylor,
  entered as preprint; Fontana & Imbert-Gérard 2025 GPW inhomogeneous media
  [V, not entered]; Perinati, Imbert-Gérard, Moiola & Stocker 2026
  elliptic–hyperbolic DG, arXiv:2604.06910 [V metadata, not entered];
  Kapita 2026 Bernstein quasi-Trefftz Helmholtz [S]; Imbert-Gérard 2025 ×2
  Maxwell / first-order Helmholtz [S]).
- Q7 `abs:("nonstandard finite difference" OR "exact finite difference" OR
  "exact scheme") AND abs:(wave OR "second order")` — 12: NSFD for ODE
  dynamical systems and epidemic models; nothing for heterogeneous wave
  media.
- Q8 `abs:"exponentially fitted" AND abs:(compact OR stencil OR "finite
  difference")` — 6, all astrophysics noise.
- Q12 `abs:"harmonic coordinates" AND abs:(wave OR hyperbolic)` — 15:
  Owhadi–Zhang math/0604380 (the 2008 CMAME paper; abstract read: harmonic
  coordinates precomputed, implicit time stepping on coarse scales, no
  scale separation or ergodicity assumed); the rest general relativity.
- Q13 `abs:("local solutions" OR "solution-based" OR "operator-adapted") AND
  abs:(stencil OR "finite difference") AND abs:(coefficient OR
  heterogeneous)` — 4, noise.
- Q14 `abs:Trefftz AND abs:"finite difference"` — 2: Tsukerman's Trefftz
  analyses of FD schemes (electromagnetics) [S].
- Q15 `abs:("generalized finite difference" OR "finite difference") AND
  abs:("non-polynomial" OR "nonpolynomial") AND abs:basis` — 1 (RBF-WENO)
  [S].
- Q18 `abs:("numerical homogenization" OR "multiscale") AND abs:"wave
  equation" AND abs:("rough coefficients" OR "heterogeneous media" OR "no
  scale separation")` — 9: LOD and multiscale methods for waves
  (Abdulle–Henning 2014/2016; Maier–Verfürth 2021; CEM-GMsDG) [S] —
  coarse-scale bases from local problems, FEM/DG, the K1 idea at the
  multiscale-FEM end.
- Q19 `abs:("polynomial in time" OR "time-polynomial" OR "polynomial
  solutions") AND abs:"wave equation" AND abs:(basis OR Trefftz)` — 2:
  Moiola–Perugia 2018 (entered).
- Q21 `abs:("Chebyshev system" OR "Tchebycheff system" OR "extended complete
  Chebyshev") AND abs:(interpolation OR "finite difference" OR quadrature)`
  — 2: Gerolymos 2012 (entered).

WebSearch:
- W7 `Mickens "exact finite difference" OR "nonstandard finite difference"
  wave equation variable coefficients OR heterogeneous media scheme` →
  Mickens' books (1993 entered [V]; 2000, 2005, 2020 [S]); exact schemes for
  the unidirectional nonlinear wave equation [S]; nothing for a coefficient
  edge.
- W8 `quasi-Trefftz 2025 OR 2026 wave equation time-domain "smooth
  coefficients" OR "piecewise-smooth" space-time DG new paper Imbert-Gérard
  Moiola Stocker Perinati` → nothing beyond Q1.

Crossref: Vizvari et al. 2020 identified from the DOI the notes had
(entered).

**Verdict.** The quasi-Trefftz line is the closest published relative and
is Taylor-based by construction (Imbert-Gérard 2025's title says so);
exact and fitted schemes stay steady or ODE; harmonic coordinates (Owhadi–
Zhang 2008) are the closest 2-D relative and a different mechanism. **No
prior instance found** of an ODE-continued local basis for a time-domain
wave *stencil*. Queries above.

### P4 — the standing alternative: smooth or average the medium

arXiv API: Q5 `abs:(unresolved OR "under-resolved" OR subgrid OR "sub-grid"
OR "sub-cell") AND abs:interface AND abs:"finite difference" AND abs:wave`
— **0**; Q9 `abs:"effective medium" AND abs:"finite difference" AND
abs:(seismic OR elastic)` — **0**; Q10 `abs:("non-periodic homogenization"
OR "nonperiodic homogenization") AND abs:wave` — **0**; Q6 `abs:"thin
layer" AND abs:wave AND abs:(homogenization OR "finite difference" OR
"effective")` — 25, noise except two Backus-average papers (Bos et al. 2016;
Adamus 2020 [S]); Q20 `abs:(smoothed OR regularized OR "smoothing") AND
abs:interface AND abs:(coefficients OR parameters) AND abs:"finite
difference" AND abs:wave` — 1, noise. The seismology literature is not on
arXiv (§4).

WebSearch and fetches:
- W1 `Moczo Kristek Vavryčuk Archuleta Halada 2002 "3D heterogeneous
  staggered-grid finite-difference modeling" volume harmonic arithmetic
  averaging BSSA` → the paper's PDF at ig.cas.cz; first page read (author
  list, abstract: volume harmonic averaging of moduli at stress positions,
  arithmetic averaging of density at displacement positions; "the scheme
  allows for an arbitrary position of the material discontinuity in the
  spatial grid"). Also surfaced Koene et al. 2021 (arXiv:2104.08206).
- W2 `Capdeville Guillot Marigo "1-D non-periodic homogenization for the
  seismic wave equation" …` → OUP article page fetched (vol. 181(2)
  897–910); the 2-D P–SV paper [V] by Crossref; the 3-D (GJI 2018) and
  residual-homogenization (GJI 2015) papers [S].
- W10 `finite difference wave equation "smooth" interface "transition zone"
  OR "transition width" OR "smoothed interface" tanh sub-grid thinner than
  grid spacing accuracy order convergence knee` → Erickson–O'Reilly–
  Nordström 2019 (JSC, non-smooth wave speeds; [V] metadata by Crossref;
  the DiVA landing page returned no abstract, the Springer page was not
  fetched); otherwise SBP grid-interface papers. **No paper found on a
  given sub-grid smooth transition as the object of study.**
- W11 `sub-cell OR subcell material interface Yee FDTD staircase error
  smoothing permittivity averaging accuracy "second order" Tornberg Engquist
  Ditkowski Hesthaven effective medium` → MEEP subpixel smoothing
  (Farjadpour et al. 2006, entered [V]); volume-average effective
  permittivity and dispersive-media subcell averaging [S]; Ditkowski, Dridi
  & Hesthaven 2001 [V metadata by Crossref, not entered].
- Crossref queries by title → Tornberg–Engquist 2003 [V], Tornberg–Engquist
  2006 [V, open-access PDF's first page read, quoted in §1a K6],
  Tornberg–Engquist 2008 [V], Gustafsson–Wahlund 2004 [V], Symes–Terentyev
  2009 [V], Kristek et al. 2017 [V], Jiang–Zhang 2024 [V], Koene et al.
  2022 [V; arXiv abstract read].

**Verdict.** The alternative is real, current, and second-order-limited at
the interface by construction where analysed (Tornberg–Engquist 2006). It
was not compared against in #27–#42; **#69 (2026-09-20) ran the
comparison** in 1-D (notes §2.1) and on the scattered nodes (notes §5.7),
with both sources fetched and read (`papers/`, `docs/paper-index.md`). §1a
K6 (iii) records the result and its scope; §6 words it.

### P5 — seed-bibliography verification

- 27 seeded entries → 27 **[V]**, 0 dropped. Routes: Crossref records for
  the 19 DOIs and for the from-memory DOIs of Fornberg 1988 and Fornberg–
  Lehto 2011 (both resolved with matching metadata); the arXiv API for
  math/0505223, 2011.04617, 2408.00392; archive.org metadata plus zbMATH
  Open for Karlin–Studden; zbMATH Open plus the Crossref reference-book
  record for Samarskii 2001; the Crossref book record plus zbMATH for
  Roos–Stynes–Tobiska; the dissertation's own title page.
- Corrections made from the records: `ImbertGerardMoiolaStocker2024` is the
  IMA JNA 45(6) 2025 paper with Perinati as third author (key kept for the
  section drafts; #61 may rename); Owhadi–Zhang 2007 and 2008 given their
  DOIs (10.1002/cpa.20163, 10.1016/j.cma.2008.08.012); Allen–Southwell,
  Il'in and Scharfetter–Gummel titles confirmed as the records' own; Symes–
  Vdovina pages 363–371 per the record (the dissertation and JCP print
  363–370). Fields no fetch confirmed were **dropped and said so**: Coppel's
  LNM volume number, Roos et al.'s edition and series volume, Samarskii's
  place of publication.
- The three entries named in the notes but missing from the seed: Vizvari
  et al. 2020 and Jiang & Zhang 2024 identified and entered [V]; Zalik [S],
  not entered (K4).
- 39 entries added from P1–P4, each [V] with a dated comment; total 66;
  `tectonic` renders all 66 under `\nocite{*}` (main.pdf, 7 pp.).
- Unreachable today: mathnet.ru (connection reset, three attempts; the
  Russian original of Tikhonov–Samarskii and Il'in's Mat. Zametki record
  are therefore not entered — the English translations' DOIs are);
  intlpress.com article page (403; its open-access PDF was fetched);
  Springer article pages (Crossref used throughout); the AMS PDF of Pólya
  1922 (Crossref record only).

## 3. Verification-status rule

- **[V]** requires a fetched primary source: the Crossref API record of the
  DOI (publisher-deposited metadata), the arXiv API record, the
  publisher's or repository's page, the paper's own PDF, or, for pre-DOI
  books, the archive.org or zbMATH Open record. The fetch is named in §2 or
  in the entry's `VERIFIED` comment in `paper/references.bib`.
- **[S]** is search-result or page-summary evidence only: may be discussed
  here and in prose with hedging; may **not** enter `paper/references.bib`;
  no §1 verdict may rest on an [S] item alone.
- A metadata [V] is not a content [V]. Where a claim about what a paper
  *does* rests on its abstract or first page, §1–§2 say "abstract read" or
  "first page read"; where nothing was read, "metadata only". The
  manuscript quotes content only from papers read at source.
- Preprints ([V] via the arXiv API) are cited as preprints and weighted
  accordingly.

## 4. Standing hazards

- **Geophysics is not on arXiv.** P4's three arXiv queries returned zero;
  the seismology lines came from WebSearch plus Crossref. Before any
  circulation, re-sweep GJI, Geophysics and BSSA through Crossref
  (`query.bibliographic`) for "equivalent medium", "anti-aliased interface",
  "smooth transition" and "gradient zone".
- **Task vocabulary found what construction vocabulary missed.** Koene et
  al. 2022 surfaced through "unresolved interface", Erickson et al. 2019
  through "smooth interface transition zone"; neither says "seed", "basis
  function" or "Trefftz". The sub-grid smooth edge may appear in other
  words: "transition zone", "gradient layer", "smoothed velocity model",
  "anti-aliased interface", "tapered contrast". Sweep those first-class, as
  weil-positivity-lab's ledger learned the hard way.
- **Absence ≠ absence of practice.** Every "no prior instance found" above
  is tied to the listed queries. A paper that marches ODEs to build a
  stencil basis through a thin layer without calling it that may exist.
  Phrase every such statement as "no prior instance found", never
  "novel".
- **Content unread at source.** Pólya 1922 (theorem cited as the notes state
  it); Karlin–Studden page pointer via Zalik's note; Erickson et al. 2019
  (metadata only); Owhadi–Zhang 2008, Moczo et al. 2002 and Koene et al.
  2022 at abstract or first-page level; Tornberg–Engquist 2006 first page.
  Read before quoting specifics beyond what §1a records.
- **A fast-moving neighbour.** The quasi-Trefftz group posted four preprints
  in 2025–26 (Q1). Re-run Q1 before the arXiv upload; if a time-domain
  quasi-Trefftz paper for *under-resolved* coefficients appears, O1's
  bucket must be re-examined.
- **Blocked publishers.** Springer, SIAM, IEEE, AMS, Wiley and OUP article
  pages were not fetched directly (Crossref filled in metadata; one OUP page
  answered a WebFetch); intlpress 403'd; mathnet.ru reset the connection.
- **Bibliography labels.** With `amsalpha`, the unbraced accents that avoid
  tectonic's BibTeX hang (#52) render the single-author labels for
  Mühlbach and Pólya oddly (`M7̈3`). A #61 decision: another style, or
  `alpha`, or accept.

## 5. Cross-references

- `paper/references.bib` — verified entries only; each carries a dated
  `VERIFIED` comment naming its fetch.
- `docs/stiff-features.md` §3 — the 2026-09-19 scan this ledger upgrades
  (left unchanged; where this pass adds to it, the addition is here).
- `docs/paper-index.md` — page maps for the dissertation and the two
  preprints read in P1.
- Issues #27 (the exploration), #51 (the manuscript epic), #53 (this pass),
  #55 (places §6a), #61 (assembly; re-checks the wording against §6).

## 6. Manuscript wording

The paragraph and sentences below are the only novelty language the
manuscript may use without re-opening this ledger. #55 places (a) in §1
(`\subsection{Relation to prior work}`); #61 fixes (b) in the abstract and
conclusions. Every `\cite` key exists in `paper/references.bib` and is [V].

### 6a. Relation to prior work (§1 of the manuscript)

```latex
Three lines of prior work meet here, and the construction adds nothing to
any of them taken alone. Stencils and elements built from local solutions
of the operator are the fitted operators of the 1950s and 60s
\cite{AllenSouthwell1955,Ilin1969,ScharfetterGummel1969,RoosStynesTobiska2008}
and the exact schemes that followed \cite{Mickens1993,VizvariEtAl2020};
the constant-flux seed in particular is the harmonic mean of Tikhonov and
Samarskii \cite{TikhonovSamarskii1962,Samarskii2001}, the special elements
of Babu\v{s}ka and Osborn \cite{BabuskaOsborn1983,BabuskaCalozOsborn1994},
the multiscale elements of Hou and Wu \cite{HouWu1997} and the harmonic
coordinates of Owhadi and Zhang \cite{OwhadiZhang2007}, whose extension to
the acoustic wave equation \cite{OwhadiZhang2008} is the closest
two-dimensional relative of what follows. Basis functions that are
polynomial-in-time solutions of the wave equation are the Trefftz spaces
of space--time discontinuous Galerkin methods
\cite{KretzschmarMoiolaPerugiaSchnepp2016,BanjaiGeorgoulisLijoka2017,MoiolaPerugia2018};
their quasi-Trefftz extension to variable coefficients
\cite{ImbertGerardMoiolaStocker2023,ImbertGerardMoiolaStocker2024,ImbertGerard2025local}
builds them by Taylor expansion at a point, the route
Section~\ref{sec:seeds1d} shows cannot reach an edge thinner than the
stencil. The nonsingularity of the stencil solve rests on the theory of
Chebyshev systems \cite{Polya1922,KarlinStudden1966,Coppel1971,Muhlbach1973}.
For a jump, the interface-aware stencils this work generalises
\cite{Martin2016,MartinFornbergStCyr2015,MartinFornberg2017} stand beside
the immersed interface \cite{LeVequeLi1994,ZhangLeVeque1997,SabatiniEtAl2023},
explicit simplified interface \cite{PirauxLombard2001,LombardPiraux2004}
and summation-by-parts \cite{MattssonNordstrom2006,DuruVirta2014,GranathWang2025}
treatments, all of which take the jump as given; standard schemes are
first order there \cite{Brown1984,SymesVdovina2009,VishnevskyEtAl2014}.
The standing alternative in seismic and electromagnetic finite differences
changes the medium rather than the stencil: coefficients averaged over a
cell \cite{Backus1962,MuirEtAl1992,MoczoEtAl2002,KristekEtAl2017,JiangZhang2024,FarjadpourEtAl2006},
thin structure homogenised
\cite{CapdevilleGuillotMarigo2010a,CapdevilleGuillotMarigo2010b}, or the
jump smoothed over a few cells
\cite{TornbergEngquist2003,TornbergEngquist2006,KoeneWittstenRobertsson2022}.
Those methods choose the smoothing width from the grid and accept a
second-order error component at the interface \cite{TornbergEngquist2006};
here the width is a property of the medium, smaller than the grid, and
the aim is the scheme's full order through it. We found no prior instance
of stencils on an unchanged equispaced or scattered node set whose basis
is continued through a sub-grid smooth edge by ordinary differential
equations, nor of the observation that the construction reduces to the
jump stencils as $\delta \to 0$; the search is logged in the repository's
\texttt{LITERATURE.md}. We did not compare against the smoothing and
averaging methods, and claim nothing relative to them.
```

(About 330 words. If #55 needs the 150–250 the issue asked for, cut the
sentence on Chebyshev systems, which §3 of the manuscript cites anyway,
and merge the two jump-method sentences; keep the last three sentences
verbatim, they carry the obligations of §1a K6 and §1b O1.)

### 6b. Claim sentences (abstract and conclusions)

Bucket §1b, worded as "no prior instance found":

- *Abstract-safe.* "The seeds are the $t = 0$ profiles of polynomial-in-time
  solutions, continued through the edge by ordinary differential equations
  rather than by Taylor expansion, and they reduce to the interface-aware
  stencils of a jump as $\delta \to 0$: one construction covers the jump,
  the sub-grid edge and the resolved edge."
- *Conclusions.* "To our knowledge no earlier scheme builds a stencil basis
  for the time-domain wave equation through a sub-grid smooth edge by an
  ODE march while leaving the rest of the scheme standard; the ingredients
  (bases from local solutions, polynomial Trefftz spaces, the harmonic mean)
  are each classical. On the node sets tested the rule is: seed when
  $\delta \le h$."
- *Required caveat wherever the rule is stated as a headline.* "The
  comparison is against coefficient sampling on the same nodes and against
  our implementation of the coefficient treatments of seismic finite
  differences (cell means over one and two cells, band-limited
  coefficients, a widened edge) on the same schemes, one contrast and one
  test problem; their sources analyse and run them on staggered grids. On
  these grids and node sets no coefficient treatment reaches the seeds'
  order; the best of them is second order at a jump in one dimension, and
  the seeds lie 3 to 700 times below it there and 3 to 6 times below it on
  the scattered nodes." (Replaces the "not run" wording of 2026-09-20
  morning; #69.)

Not to be used: "novel", "new method", "first", "outperforms existing
interface methods", "beats smoothing / averaging methods" (only our
implementations, one scheme per dimension, one contrast, one problem;
#69), "arbitrary contrast", "any order" (only FD4 / degree 3 were run).
