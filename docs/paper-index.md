# Paper index

Where to look in each PDF so a session can jump straight to the relevant
pages with `pdftotext -f <first> -l <last> -layout <pdf> -` instead of
re-reading whole documents. Page numbers below are **PDF page numbers**
(what `pdftotext -f/-l` takes). All PDFs live in `papers/`.

Checked 2026-09-19: every public PDF was downloaded afresh and its sha256
matches the one recorded in `papers/README.md`, and the page counts below
match the local files, so no upstream revision has moved these ranges.

## Dissertation (`martin-dissertation-2016-rbf-fd-interfaces.pdf`, 145 pp.)

Printed page = PDF page − 9 (Chapter 1 begins on printed p. 1 = PDF p. 10).

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–9 | Front matter, abstract, TOC | — |
| 10–16 | Ch. 1 Introduction (history of interface treatments; RBF-FD overview §1.3 at p. 14) | Slide context for Part 2 |
| 17–29 | **§2.1** 1-D FD weights across two interfaces | The 1-D method. Eq. 1 PDE in first-order form; eq. 2–3 time-derivative continuity via powers of D; eq. 4–8 expansions and discrete operators; eq. 9 derivative matrix; eq. 10–12 variable-coefficient multiplication matrices; eq. 13–16 worked example medium; eq. 17–21 continuity rows k = 0..4; **eq. 22–24 continuity matrices and translated basis U**; eq. 25–28 f-basis via the operator (loses one order; we use the direct construction instead, as the MATLAB does); Fig. 2-2 basis functions (p. 26); Fig. 2-3 weights near an interface (p. 30) |
| 30–36 | **§2.2** 1-D numerical results | Test problem eq. 29–31 (c = 1.5 + 0.05 sin 2πx, ρ = 1.5 on [0, 0.5]); Fig. 2-5 snapshots; Fig. 2-6 method vs standard FD4 at t = 1 (p. 34); Fig. 2-7 errors; **Fig. 2-8 convergence: FD4 first order, method fourth order** (p. 36) |
| 37–39 | §3.1–3.2 2-D elastic wave equation, standard RBF-FD weights | 2-D operator construction |
| 39–48 | **§3.3** RBF-FD weights across interfaces (§3.3.1 continuous fields p. 40; §3.3.2 discontinuous fields p. 46) | 2-D interface treatment |
| 49–74 | §3.4 2-D EWE numerical examples (§3.4.1 analytic validation p. 52; §3.4.2 two curved interfaces p. 57; §3.4.3 mini-Marmousi p. 65; §3.4.4 sharp-corner limitation p. 71) | 2-D test cases and expected results |
| 75–89 | Ch. 4 Heat problems in 1-D | Not needed for the demo |
| 90–122 | Ch. 5 Heat problems in 2-D | Not needed for the demo |
| 123–124 | Ch. 6 Conclusions | Closing slide |
| 125– | Bibliography | Citations |

Code that implements §2.1: `src/pdes_demo/wave1d/operators.py` (docstrings
cite the equation numbers above).

## OpenAI, *Finite Time Blowup for Navier–Stokes* (`openai-2026-finite-time-blowup-navier-stokes.pdf`, 166 pp.)

PDF page = printed page.

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–3 | Abstract, contents, **§1 Introduction: Theorem 1.1**, §1.1 historical context and prior work | The claim, precisely; the prior-work slide |
| 3–6 | **§2 Physical description of the blowup** (§2.1 inner core) | Lay explanation: collapsing self-similar vortex, oscillatory pulses cancelling the singular residual |
| 6–24 | §3 Proof outline | Only if asked how the proof is organised |
| 24–125 | §4–§9 construction and corrections | Skip |
| 116–125 | §10 Compact forcing and whole-space breakdown; **Corollary 10.6** (torus case, alternative D) | Why both (C) and (D) follow |
| 126–164 | Appendices A–C | Skip |
| 165–166 | References | Citations |

Companion: `openai/NavierStokesAndEuler` README (Lean 4.34.0-rc2, Mathlib,
"Comparator" independent checking; `ComparatorChallenges/NavierStokes.lean`
is the formal target statement, adapted from Google DeepMind's
`formal-conjectures`). Clay statement
(`clay-2000-fefferman-navier-stokes-problem-statement.pdf`, 6 pp.): the four
alternatives (A)–(D) are on p. 2.

## OpenAI, *Finite Time Blowup for the Euler Equation* (`openai-2026-finite-time-blowup-euler.pdf`, 57 pp.)

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–3 | Abstract, §1 Theorem 1.1 (unforced Euler blowup from smooth compactly supported data), §1.1 historical context (Elgindi; Chen–Hou; Córdoba–Martínez-Zoroa) | The companion claim in one line |
| 3–9 | §2 amplification mechanism and proof order | Only if asked |
| 9–56 | Construction | Skip |

## Alpöge & Buckmaster, *Blowup for the Euler equations with smooth forcing* (`alpoge-buckmaster-2026-euler-blowup-smooth-forcing.pdf`, 112 pp.)

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–3 | Abstract, §1.1 Theorem 1.1 (forced, axisymmetric with swirl, supported in a torus; ‖∇Γ‖, ‖ω‖ → ∞), §1.2 mechanism, §1.3 related work | The parallel result, precisely; who they build on |
| 3–112 | Construction | Skip; the authors call the write-up preliminary |

## Buckmaster, statement of 2026-09-07 (`buckmaster-2026-statement.pdf`, 4 pp.)

Read in full (4 pp.): results, tools, the Sep 3 email verbatim, the Sep 6
calls, the two proposals, what he is and is not claiming. Quoted in
`docs/navier-stokes-notes.md`.

## Tornberg & Engquist 2006 (`tornberg-engquist-2006-regularization-wave-propagation-maa.pdf`, 28 pp.)

PDF page = printed page − 246. Fetched 2026-09-20 for #69 (the comparators).

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–2 | Abstract, §1 introduction, **§2 equations**: p_t = a u_x, u_t = b p_x with a = −ρc², b = −1/ρ; jump conditions eq. 6–7 | The variables their regularisation acts on |
| 3–4 | §3 the Yee scheme and the Q4 (fourth-order staggered) scheme, eq. 9–12 | Why their fourth-order result does not transfer: the Q4 has temporal correction terms |
| 5–7 | **§4.1 the regularised coefficients**: eq. 18–22 (1/a linear across [x̄ − h/2, x̄ + h/2], the same for b, i.e. 1/b = −ρ linear), eq. 23; the remark that regularising a itself stays first order; Theorem 1 (second order in L2) | The `cell` comparator (`wave1d/treatments.py`) is eq. 18 and 22 at δ = 0; T0 is the "regularise a itself" case |
| 8–14 | §4.2–4.4 numerical results and stability for the Yee scheme | — |
| 15–18 | **§5 higher order**: eq. 37–39, second order only with the correction terms masked within 3h/2 of the jump; §5.2 results | The caveat on "second order" in the notes §2 and the manuscript |
| 19–26 | §6 a problem with a discontinuous solution, §7 conclusions | — |
| 27–28 | References | Citations |

## Koene, Wittsten & Robertsson 2021 (`koene-wittsten-robertsson-2021-anti-aliasing-vs-equivalent-medium-arxiv.pdf`, 34 pp.)

arXiv v2 of the GJI 229 (2022) paper. Fetched 2026-09-20 for #69.

| PDF pages | Section | Use for |
| --- | --- | --- |
| 1–3 | Abstract, §1 the four classes of interface treatment | Framing: anti-aliasing wins in acoustic media, Schoenberg–Muir in elastic |
| 7–9 | **§3.2 the anti-aliased step-function** eq. 22 (½ + Si(πz/Δz)/π), §3.2.2 the Kaiser window (length and shape 3), which quantities: **density and compliance** (Mittet 2017), the three hazards (fluid–solid, negative properties, CFL) | The δ → 0 case of the `bandlimit` comparator |
| 10 | **§3.3 the 2-D low-pass filter**: oversample ×10, zero-phase FIR along each axis, Hanning window of 51 points centred at 1.1× the Nyquist wavenumber, subsample | The `bandlimit` kernel as implemented (half-width 2.5h, cutoff 1.1π/h) |
| 10–12 | §3.4 Schoenberg–Muir calculus, orthorhombic averaging | T3, named and excluded |
| 13–22 | §4 numerical tests (acoustic, elastic isotropic, anisotropic) | Their verdict, quoted in LITERATURE.md K6 |
| 23–28 | §5 discussion, conclusions | — |
| 30–34 | Appendices (Si rational approximation; compliance matrices) | — |

## Martin & Fornberg preprints

Both are indexed below; page numbers are PDF pages.

### Geophysics 2015 (`martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf`, 26 pp.)

PDF page = printed page. The 3rd-order predecessor of the dissertation's
2-D method; best source for the plain RBF-FD setup (Cases 1–3) and the
elastic test problems.

| PDF pages | Section | Use for |
| --- | --- | --- |
| 2–3 | Abstract, introduction (dispersive vs interface errors; FD is first order across interfaces) | Slide framing |
| 4–8 | RBF-FD methodology: Case 1 (Cartesian FD), Case 2 (RBF-FD, eq. 1–6: interpolant, weights `A w = L phi`, polynomial augmentation eq. 5–6, "rows in P about half the rows in A"), IMQ used here, hyperviscosity citation (Fornberg–Lehto 2011) | Naive operator construction (#6) |
| 8–13 | Case 3 in 1-D (eq. 7 EWE, eq. 8–17 continuity via powers of D; null space of `C^T`) | Cross-check of the 1-D method |
| 13–18 | Case 3 in 2-D (eq. 18 expansions, eq. 19 five-field D, eq. 20 continuity of u', v', g', h', eq. 21–22 rotations, eq. 23–24; curvature via local expansion of the interface in x') | Interface-aware stencils (#8) |
| 18–20 | Example 1: plane P-wave, two curved interfaces (Fig. 4 node set N = 1600, Fig. 5 errors at t = 0.3 vs pseudospectral) | Test case and expected error maps |
| 21–23 | Example 2: point source in a mini-Marmousi model | Out of scope |
| 23–24 | Conclusions, references | — |

### JCP 2017 (`martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf`, 44 pp.)

PDF page = printed page. The 4th-order "simplified" interface treatment the
dissertation ch. 3 is built on (square-matrix inversion instead of null
spaces of rectangular matrices).

| PDF pages | Section | Use for |
| --- | --- | --- |
| 2–4 | Abstract, introduction (advantages list; what changed vs the 2015 method) | Slide framing |
| 4–8 | Methodology: Type 1/2/3 stencils, Fig. 1, eq. 1 (RBF-FD with polynomials) | — (same as Geophysics) |
| 8–13 | Type 3 in 1-D (eq. 2–5), specific 1-D example (Fig. 2 wave-speed profile) | Cross-check of `wave1d` |
| 13–23 | 1-D basis construction in detail (Fig. 3 basis functions, Fig. 4 weights near the interface), why the f-basis is one order lower and how to recover it | Already ported in `wave1d/operators.py` |
| 23–31 | **2-D Type 3**: nearest interface point as origin, Fig. 5 (two rows straddle the interface; the layout "has been key in maintaining stability"; rest via static repulsion), eq. 33 continuity of u', v', g', h', eq. 34–35 rotations, u/v basis by inverting square continuity matrices, then f/g/h basis, elimination of dependent columns, curvature terms | Interface-aware stencils (#8) |
| 31–33 | **Numerical parameters**: GA with `eps = 0.4/d` (eq. 42), 30 nodes / degree 4 away from interfaces, 19 / degree 3 across, Δ³ hyperviscosity on the same piecewise basis, RK4 | Defaults for #6, #7 |
| 33–38 | Test case 1: two curved interfaces, λ = μ = 4 + sin(2πx) sin(2πy), ρ = 2 in the band (eq. 43; Fig. 6–10: fields, FD4 / PS / RBF / hybrid convergence, wall-clock, linear-vs-curved interface treatment) | Expected convergence (4th order) |
| 39–42 | Test case 2: mini-Marmousi point source (Fig. 11–13), conclusions | Out of scope |
| 42–43 | Acknowledgements, references | — |

MATLAB counterpart of both: `~/MathGraduateResearchAndCourseWork/waveEq2DMatlab/`
(see the breadcrumbs on issue #2). Note that the MATLAB uses GA for the
dx/dy and hyperviscosity stencils (`GAshp = 0.4`, distance normalised by the
4th column of `knnsearch`, i.e. the 3rd-nearest neighbour excluding self);
the IMQ shape 0.2 only appears in the post-hoc error interpolant.
