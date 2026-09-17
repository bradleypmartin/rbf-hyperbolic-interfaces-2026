# Paper index

Where to look in each PDF so a session can jump straight to the relevant
pages with `pdftotext -f <first> -l <last> -layout <pdf> -` instead of
re-reading whole documents. Page numbers below are **PDF page numbers**
(what `pdftotext -f/-l` takes). All PDFs live in `papers/`.

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
"Comparator" independent checking). Clay statement
(`clay-2000-fefferman-navier-stokes-problem-statement.pdf`, 6 pp.): the four
alternatives (A)–(D) are on p. 2.

## Martin & Fornberg preprints

- `martin-fornberg-2017-rbf-fd-seismic-interfaces-jcp-preprint.pdf` (44 pp.):
  the 2-D interface treatment the dissertation ch. 3 builds on. Index the
  relevant sections when 2-D work starts.
- `martin-fornberg-stcyr-2015-rbf-fd-2d-seismic-geophysics-preprint.pdf`
  (26 pp.): 2-D elastic RBF-FD without interfaces (node sets, hyperviscosity,
  stencil sizes). Index when 2-D work starts.
