# Stiff smooth edges: seed stencils (Part 3, issue #27)

Exploration started 2026-09-19, after the talk's material was frozen. Not
part of the presentation. Brad's question from a research-group session
twelve or thirteen years ago, explored once in 1-D and then lost: between
a smoothly varying material and a jump there is a "twilight zone" where the
material is technically smooth but changes over a distance far smaller than
the grid spacing. Can stencils on the *same equispaced grid* handle such an
edge at full order, the way the dissertation's interface-aware stencils
handle a jump, by replacing the monomials of the stencil's basis with
functions obtained from ODEs?

Short answer, from the experiment in section 2: yes, and the construction
is the dissertation's interface construction with the jump replaced by the
edge profile. Code: `src/pdes_demo/wave1d/stiff.py` (the seeds and the
weights), `spectral.py` (the reference solution), `domain.py`
(`LayeredMedium(edge_width=...)`), `scripts/wave1d_stiff.py` (the figures),
`tests/test_wave1d_stiff.py`. The 2-D proof of concept (#36–#42, flat
first) is under way: `LayeredMedium2D(edge_width=...)` and the
normal-incidence reference `wave2d/exact.py: spectral_plane_wave` from #36,
tested in `tests/test_wave2d_smooth_edges.py`; the elastic seeds of a
straight edge, `wave2d/seeds.py` from #38, derived in section 4.2 and
tested in `tests/test_wave2d_seeds.py`; results go into section 5 as they
land.

## 1. Formulation (#28)

### 1.1 Setting

The 1-D problem of Part 2 in the repo's variables, u = particle velocity,
f = stress, K = ρc²:

    ρ u_t = f_x,        f_t = K u_x.

The layer `[0, 0.5)` has (c, ρ) = (2, 1) inside and (1, 1) outside. With
`edge_width = δ > 0` the two edges are tanh transitions,

    s(x) = Σ_m ½ [tanh((x - x₁ + 2m)/δ) - tanh((x - x₂ + 2m)/δ)],
    c(x) = c_bg + (c_layer - c_bg) s(x),    ρ likewise,

summed over periodic images so the profile is smooth and periodic to
rounding; δ = 0 is the jump of Part 2, bit for bit. A standard FD4 stencil
samples c and ρ at its nodes. When δ ≪ h it sees a jump and treats it as if
the coefficients were smooth between nodes, which is the first-order
failure of Part 2; when δ ≫ h it sees a smooth function and is fine. In
between is the case of interest.

### 1.2 Seeds as the profiles of solutions polynomial in time

Eliminating one field gives a second-order operator per field:

    u_tt = L_u u,   L_u = (1/ρ) ∂ₓ K ∂ₓ;        f_tt = L_f f,   L_f = K ∂ₓ (1/ρ) ∂ₓ.

Look for solutions polynomial in time, u(x, t) = Σ_j tʲ g_j(x). Then
(j+2)(j+1) g_{j+2} = L_u g_j, so the two top coefficients are in ker L_u and
the t = 0 profile g₀ satisfies L_uᵐ g₀ = 0 for some m. The profiles of such
solutions are the functions that play the role of polynomials for this
operator: with constant coefficients L_u = c² ∂ₓ², and ker ∂ₓ²ᵐ is exactly
the polynomials of degree < 2m.

Brad's bullets in #27 (with the correction ∂ₜᵏu = C rather than 0) are this
construction, read one seed at a time. Anchored at the stencil's evaluation
point x_e:

- **φ₀ = 1.** The solution u ≡ C.
- **φ₁, the x-like seed.** A solution linear in time, u = g₀ + t g₁ with
  u_t = g₁ = C, forces L_u g₀ = 0, i.e. K g₀' = const. Normalised to slope 1
  at x_e: `K φ₁' = K(x_e)`, φ₁(x) = K(x_e) ∫ₓₑˣ dξ / K(ξ). Through the other
  field this says f_t = K u_x is constant in x: the flux is what stays
  smooth, not the slope.
- **φ₂, the x²-like seed.** u = g₀ + t g₁ + ½ C t² with u_tt = C forces
  L_u g₀ = C. With C = 2 c_e² (c_e = c(x_e)) and g₀(x_e) = g₀'(x_e) = 0 this
  is (x − x_e)² when the coefficients are constant. This is where C ≠ 0
  matters: C = 0 would give back ker L_u.
- **φ₃, the x³-like seed.** u_ttt = C fixes the profile of u_t (L_u g₁ = C),
  not of u; the cubic seed is the profile whose second time derivative is
  x-like, L_u φ₃ = 6 c_e² φ₁. Through the other field, f_ttt = K (u_tt)_x =
  6 c_e² K(x_e) is constant, so the odd seeds are "∂ₜᵏ f = C" and the even
  ones "∂ₜᵏ u = C". That is the same even/odd field swap as
  `operators.continuity_matrices` ("even k relate a field to itself, odd k
  route through the other field").

In general, for k ≥ 2,

    L φ_k = k (k−1) c_e² φ_{k−2},      φ_k(x_e) = φ_k'(x_e) = 0,

with L = L_u for the u stencil and L_f for the f stencil. Constant
coefficients give φ_k = (x − x_e)ᵏ by induction.

**Numerics.** The chain is integrated as a first-order system that never
differentiates the coefficients: for u, (φ, ψ = K φ') with φ' = ψ/K and
ψ' = ρ k(k−1) c_e² φ_{k−2}; for f, (φ, ψ = φ'/ρ) with φ' = ρ ψ and
ψ' = k(k−1) c_e² φ_{k−2} / K. In the stencil coordinate ξ = (x − x_e)/h_s,
h_s the stencil half-width, the same equations hold and φ_k ≈ ξᵏ, so the
interpolation matrix A[k, j] = φ_k(ξ_j) is a small perturbed Vandermonde
matrix (condition number 20 to 60 for δ from 10⁻⁵ to 1). One DOP853 march
per stencil from ξ = 0 outward to each side, with segment boundaries at the
edge centres and at ±10δ from them so the adaptive step never has to
discover the edge in the middle of a long step. The weights solve
A w̃ = e₁, since only φ₁ has a non-zero derivative at x_e, and w = w̃ / h_s.
`stiff_weights` returns (w_u, w_f) with the same contract as
`interface_weights`; `build_operators(mode="aware")` uses it in every
stencil whose nodes see different material values (for a tanh edge, out to
about 19δ + 2h, where the tails round away), and Fornberg weights
elsewhere. About 5 to 15 ms per stencil; 2.3 s for the n = 1600 operator at
δ = 0.01.

### 1.3 Nested kernels, and the jump as the δ → 0 limit

Only the span of the seeds matters for the weights, and the spans are
canonical: span{φ₀..φ_k} is the chain

    {1} ⊂ ker L ⊂ {f : L f = const} ⊂ ker L² ⊂ {f : L² f = const} ⊂ …

because L sends each seed two steps down the chain. Different anchors or
normalisations change the basis, not the space.

For a jump, read L in the weak sense: φ and the flux K φ' continuous
across it. Then ker L_u is spanned by 1 and a piecewise-linear function
whose slope ratio is K_left / K_right, which is the dissertation's
translated x (`test_translated_basis_keeps_rho_c2_u_x_continuous` checks
exactly that ρc² u_x stays continuous); L_u φ₂ = 2 c_e² gives φ₂'' = 2 c_e²/c²
on each side, which is u_tt continuous, and so on. The chain for a jump is
the translated Taylor basis of `piecewise_bases`. One construction with two
implementations: algebra for a jump, an ODE march for a smooth edge.
`test_jump_limit_recovers_the_interface_weights_at_first_order` checks
the limit numerically: for a stencil of half-width 0.02 straddling an edge,
the seed weights differ from `interface_weights` by 22%, 10%, 0.98% and
0.098% of the largest weight at δ = 10⁻², 10⁻³, 10⁻⁴, 10⁻⁵, i.e. first
order in δ, for both fields and also for a stencil straddling both edges of
a layer thinner than itself (the double-cross).

### 1.4 Well-posedness: the seeds form an extended complete Chebyshev system

Each space in the chain is the kernel of an operator in Pólya form, a
product of ∂ₓ and multiplications by positive functions:

    ρ L_uᵐ = ∂ₓ K ∂ₓ (1/ρ) ∂ₓ K ∂ₓ ⋯ ∂ₓ K ∂ₓ      (2m derivatives),
    ∂ₓ L_uᵐ  = ∂ₓ (1/ρ) ∂ₓ K ∂ₓ ⋯ ∂ₓ K ∂ₓ          (2m + 1 derivatives),

and likewise for L_f with K and 1/ρ swapped. Pólya (1922) showed that such
an operator is disconjugate on any interval where the weights are positive,
and its kernel is an *extended complete Chebyshev (ECT) system*: with
weights w₀, w₁, …, a nested basis u₀ = w₀, u₁ = w₀ ∫ w₁, u₂ = w₀ ∫ w₁ ∫ w₂, …
(the structure theorem as stated in Zalik's note, citing Karlin & Studden,
*Tchebycheff Systems*, 1966, pp. 376–380; Coppel, *Disconjugacy*, 1971, for
the disconjugacy side; references in section 3). An ECT system has the Haar
property: interpolation on any distinct nodes is uniquely solvable. So the
per-stencil solve A w̃ = e₁ is nonsingular for every edge width, including
δ → 0 where the seeds have kinks, and the condition numbers above are what
one expects from a Vandermonde matrix on five nodes in [−1, 1].

Interpolation in an ECT system also has a Newton-form remainder with the
operator M_k that annihilates the space in place of the (k+1)-th derivative
(generalised divided differences, Mühlbach 1973). For the five-point stencil
the annihilator of span{φ₀..φ₄} is ∂ₓ L², so the truncation error of the
seed stencil on a function g is O(h⁴ ∂ₓ L² g). For a *solution* of the wave
equation L u = u_tt, hence ∂ₓ L² u = ∂ₓ u_tttt: the x-derivative of another
solution, bounded independently of δ (u_x changes by O(1) across the edge
whatever its width; it is the flux K u_x that is smooth). Standard FD4 has
error O(h⁴ u⁽⁵⁾) with u⁽⁵⁾ ∼ δ⁻⁴ inside the edge. Hence the prediction:
**standard FD4 converges at fourth order only once h ≲ δ and looks
low-order above that, with a knee at h ≈ δ; the seed stencils converge at
fourth order with a constant that does not depend on δ.**

### 1.5 Why the Taylor route does not reach a stiff edge

`Material.taylor(p)` in `domain.py` was written so that the dissertation's
continuity-matrix construction (eq. 11–12, multiplication matrices from the
Taylor coefficients of 1/ρ and K about the interface) could take smoothly
varying coefficients. For a stiff edge that route cannot work: tanh(x/δ)
has poles at ±iπδ/2, so its Taylor series about the edge converges only
within πδ/2, far less than the stencil's reach of 2h when δ ≪ h. Polynomial
quasi-Trefftz spaces (section 3) are this Taylor route; the ODE march is
what a sub-grid edge needs. For a resolved edge (δ ≫ h) the seed weights
still differ from Fornberg's at first order in h/δ (17% at δ = 8h for the
default contrast), because the seeds are exact for a different
five-dimensional space; both stencils are fourth-order on solutions there
and the experiment finds them indistinguishable end to end.

## 2. Results (#32, `scripts/wave1d_stiff.py`, 2026-09-19)

Setup: the default layer `[0, 0.5)` with c: 1 → 2, ρ: 1 → 1, edges of
width δ; a right-going Gaussian stress pulse with sharpness 60 (about 7
nodes across at n = 100, so the pulse itself is resolved on every grid and
the edge is the only thing under test) started at x = −0.6 (far enough
that the ray sum's tail check passes); FD4 in space, RK4 at CFL 0.4, t = 1.
Errors are relative l2 errors in stress f at t = 1 against the exact ray
sum for δ = 0 and against a pseudo-spectral reference (1024 to 8192 nodes,
dt = 5·10⁻⁵, accurate to about 10⁻¹⁰) for δ > 0. "Seeds" is
`mode="aware"`, which for δ = 0 is the dissertation's interface-aware
stencil and for δ > 0 the ODE-continued seeds; "naive" is FD4 with the
coefficients sampled at the nodes. Runtime 50 s for everything, 30 s with
the cached references.

| δ | scheme | n = 100 | 200 | 400 | 800 | 1600 | rates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (jump) | naive | 7.1e-2 | 3.2e-2 | 1.6e-2 | 7.9e-3 | 4.0e-3 | 1.2, 1.0, 1.0, 1.0 |
| 0 (jump) | interface-aware | 3.9e-3 | 2.5e-4 | 1.6e-5 | 9.9e-7 | 6.2e-8 | 4.0, 4.0, 4.0, 4.0 |
| 0.0025 | naive | 7.2e-2 | 2.9e-2 | 6.3e-3 | 3.0e-5 | 7.4e-6 | 1.3, 2.2, **7.7**, 2.0 |
| 0.0025 | seeds | 3.9e-3 | 2.5e-4 | 1.6e-5 | 9.9e-7 | 6.2e-8 | 4.0, 4.0, 4.0, 4.0 |
| 0.01 | naive | 3.0e-2 | 5.6e-4 | 3.5e-5 | 1.1e-6 | 6.5e-8 | 5.7, 4.0, 5.0, 4.0 |
| 0.01 | seeds | 3.9e-3 | 2.5e-4 | 1.6e-5 | 9.7e-7 | 6.1e-8 | 4.0, 4.0, 4.0, 4.0 |
| 0.04 | naive | 3.5e-3 | 2.2e-4 | 1.4e-5 | 8.8e-7 | 5.5e-8 | 4.0, 4.0, 4.0, 4.0 |
| 0.04 | seeds | 3.5e-3 | 2.2e-4 | 1.4e-5 | 8.8e-7 | 5.5e-8 | 4.0, 4.0, 4.0, 4.0 |

h/δ runs from 8 to 0.5 for δ = 0.0025, from 2 to 0.125 for δ = 0.01, and
from 0.5 down for δ = 0.04.

![Error vs resolution for four edge widths](figures/wave1d_stiff_convergence.png)

**Verdict on the hypothesis of #27: confirmed on both counts.**

- *Standard FD4 looks low-order through an unresolved edge and recovers
  fourth order once the grid resolves it.* At δ = 0.0025 the naive error
  falls at rates 1.3 and 2.2 while h > δ, drops by a factor 200 between
  n = 400 and 800 (h = 2δ to h = δ, the knee), and is fourth-order-ish
  after. At δ = 0.01 the knee sits between n = 100 and 200, again at h ≈ δ.
  The pre-knee rates are not as cleanly first order as for the jump (1.0)
  because the nodes sample the tanh at grid-dependent positions, so the
  effective jump the naive stencil sees moves with n.
- *Seed stencils keep fourth order on the same equispaced grid at every
  resolution, with a constant independent of δ.* The seed errors at
  δ = 0.0025 and 0.01 agree with the jump case's interface-aware errors to
  three digits at every n (3.9e-3 down to 6.2e-8). At h = 8δ the seeds are
  18× more accurate than naive, at h = 2δ 400×. At δ = 0.04, where every
  grid resolves the edge, seeds and naive coincide to three digits although
  their weights differ (section 1.5): both are fourth-order on solutions.

Local truncation error tells the same story without the time stepping. On
the reference solution's profile at t = 1 (dissertation pulse, sharpness
600), the maximum error in u_x over the rebuilt rows, relative to
max |u_x|, at δ = 0.0025 for n = 100, 200, 400, 800, 1600:

| stencil | h = 8δ | 4δ | 2δ | δ | δ/2 |
| --- | --- | --- | --- | --- | --- |
| naive | 1.5e-2 | 2.8e-3 | 3.9e-3 | 1.9e-3 | 2.2e-4 |
| seeds | 9.9e-3 | 1.2e-3 | 9.1e-5 | 5.8e-6 | 3.6e-7 |

Seeds: ratios 8.5, 13, 16, 16, i.e. fourth order from h = 8δ on. The same
run at δ = 0.01 gives seed errors 2.2e-2, 1.8e-3, 8.0e-5, 4.5e-6, 2.3e-7:
the constant is the same. (`test_seed_stencils_are_fourth_order_on_the_true_solution`
keeps a small version of this as a regression test.)

![Snapshot at t = 1 on 100 nodes through edges of width h/8](figures/wave1d_stiff_snapshot.png)

The snapshot is the coarse-grid picture: 100 nodes, δ = 0.0025 = h/8, so
both layer edges fall between nodes. Naive FD4 trails a grid-scale
sawtooth of 4% of the pulse height everywhere (the highest FD4 modes have
negative group velocity, so the noise born at the edges outruns the
pulses); the seed stencils sit on the reference with a maximum error of
0.3%, which is the interior FD4 dispersion error of the pulse.

![Seeds for one stencil across an edge](figures/wave1d_stiff_seeds.png)

The seeds themselves, for the u field, on a stencil whose evaluation point
is h/2 left of an edge: the jump seeds (which are the dissertation's
translated Taylor basis: slope ratio K_left/K_right = 1/4 for φ₁, a
parabola of curvature ratio c_left²/c_right² = 1/4 for φ₂) and the δ = h/4
seeds are indistinguishable, the δ = 2h seeds bend over the whole stencil,
and all of them leave the monomials at the edge.

**Caveats.** One ODE march per stencil that sees the edge, about 5 to 15 ms
each, so the operator costs seconds rather than milliseconds to build;
fine for a study, not tuned. The aware region reaches 19δ + 2h from each
edge (where the tanh tails round to the far-field value), so at δ = 0.04
every row is rebuilt; a relative tolerance in `varies_over` would trim
that. The seeds converge to the jump construction at first order in δ, so
δ below about 10⁻⁵ h is better served by `interface_weights` directly. The
spectrum of the semi-discrete operator has max real part 8·10⁻⁸ at
δ = 0.005 (n = 400) and about 10⁻¹³ otherwise; RK4 at CFL 0.4 is stable
in every run here. Nothing was tried for sharper contrasts, layers thinner
than 19δ, or higher orders than FD4; the construction does not change for
any of them.

## 3. Related work (#33)

Brad's question: the construction came from intuition about these
problems around 2013; are similar approaches known? An afternoon's scan on
2026-09-19, every entry checked against a page that was actually opened
(publisher or repository page where reachable; SIAM, IEEE, AMS and
Springer block automated fetching, so for those the Crossref record of the
DOI was read instead, which is publisher-deposited metadata). Not a survey.

**Same mechanism for steady problems: basis functions from local
solutions of the operator.**

- A. N. Tikhonov and A. A. Samarskii, "Homogeneous difference schemes",
  Zh. Vychisl. Mat. Mat. Fiz. 1:1 (1961) 5–63; USSR Comput. Math. Math.
  Phys. 1:1 (1962) 5–67, [doi:10.1016/0041-5553(62)90005-8](https://doi.org/10.1016/0041-5553(62)90005-8),
  [mathnet.ru/eng/zvmmf7977](http://www.mathnet.ru/eng/zvmmf7977).
  Difference schemes for Sturm–Liouville operators with smooth and
  discontinuous coefficients. Their conservative scheme's cell coefficient
  is the harmonic mean h / ∫ dx/k over the cell, which is the two-point
  version of φ₁ (constant flux). Textbook form: Samarskii, *The Theory of
  Difference Schemes*, Marcel Dekker 2001.
- The "exact three-point schemes" line descending from it, e.g.
  [Adv. Cont. Discrete Models 2020](https://link.springer.com/article/10.1186/s13662-020-02957-7):
  stencils exact on the local solutions of a second-order ODE. Elliptic
  boundary-value problems, not waves.
- I. Babuška and J. E. Osborn, "Generalized finite element methods: their
  performance and their relation to mixed methods", SIAM J. Numer. Anal.
  20 (1983) 510–536, [doi:10.1137/0720034](https://doi.org/10.1137/0720034);
  I. Babuška, G. Caloz and J. E. Osborn, "Special finite element methods
  for a class of second order elliptic problems with rough coefficients",
  SIAM J. Numer. Anal. 31 (1994) 945–981,
  [doi:10.1137/0731051](https://doi.org/10.1137/0731051). Finite elements
  whose shape functions solve (a u')' = 0 locally: the FEM version of φ₁.
- T. Y. Hou and X.-H. Wu, "A multiscale finite element method for
  elliptic problems in composite materials and porous media", J. Comput.
  Phys. 134 (1997) 169–189,
  [doi:10.1006/jcph.1997.5682](https://doi.org/10.1006/jcph.1997.5682).
  Basis functions from local cell problems of the operator, in 2-D.
- H. Owhadi and L. Zhang, "Metric-based upscaling", Comm. Pure Appl. Math.
  60 (2007) 675–723, [arXiv:math/0505223](https://arxiv.org/abs/math/0505223):
  solutions of divergence-form elliptic equations are C^{1,α} as functions
  of the *harmonic coordinates* F, ∇·(a ∇F) = 0 with F ≈ x, which is the
  2-D x-like seed. Their follow-up, "Numerical homogenization of the
  acoustic wave equations with a continuum of scales", Comput. Methods
  Appl. Mech. Engrg. 198 (2008) 397–406,
  [Caltech repository](https://authors.library.caltech.edu/records/y53sh-pjq83),
  uses precomputed harmonic coordinates for the wave equation without
  scale separation. The closest 2-D relative of this construction.

**Fitted operators for singularly perturbed problems.** D. N. de G. Allen
and R. V. Southwell, Q. J. Mech. Appl. Math. 8 (1955) 129–145,
[doi:10.1093/qjmam/8.2.129](https://doi.org/10.1093/qjmam/8.2.129);
A. M. Il'in, Math. Notes 6 (1969) 596–602,
[mathnet.ru/eng/mzm6928](http://www.mathnet.ru/eng/mzm6928); D. L.
Scharfetter and H. K. Gummel, IEEE Trans. Electron Devices 16 (1969)
64–77, [doi:10.1109/T-ED.1969.16566](https://doi.org/10.1109/T-ED.1969.16566).
Stencil coefficients from the local exponential solutions of
convection–diffusion, so that the scheme is uniformly accurate in the
small parameter. Surveyed as "operator-fitted methods" in H.-G. Roos,
M. Stynes and L. Tobiska, *Robust Numerical Methods for Singularly
Perturbed Differential Equations*, 2nd ed., Springer 2008, §3.5.1
([PDF at Charles University](https://www.karlin.mff.cuni.cz/~knobloch/FILES/Roos-Stynes-Tobiska.pdf)).
Same mechanism, different regime: steady boundary layers, not a
propagating wave through an internal layer.

**Trefftz and quasi-Trefftz methods: basis functions that are (nearly)
local solutions of the time-dependent equation.**

- F. Kretzschmar, A. Moiola, I. Perugia and S. M. Schnepp, "A priori
  error analysis of space–time Trefftz discontinuous Galerkin methods for
  wave problems", IMA J. Numer. Anal. 36 (2016) 1599–1635,
  [doi:10.1093/imanum/drv064](https://doi.org/10.1093/imanum/drv064);
  L. Banjai, E. H. Georgoulis and O. Lijoka, "A Trefftz polynomial
  space-time discontinuous Galerkin method for the second order wave
  equation", SIAM J. Numer. Anal. 55 (2017) 63–86,
  [doi:10.1137/16M1065744](https://doi.org/10.1137/16M1065744): space-time
  polynomial solutions of the constant-coefficient wave equation as DG
  basis. The seeds of section 1.2 are the t = 0 traces of such solutions.
- L.-M. Imbert-Gérard, A. Moiola and P. Stocker, "A space–time
  quasi-Trefftz DG method for the wave equation with piecewise-smooth
  coefficients", Math. Comp. 92 (2023) 1211–1249,
  [arXiv:2011.04617](https://arxiv.org/abs/2011.04617), and "Polynomial
  quasi-Trefftz DG for PDEs with smooth coefficients: elliptic problems",
  [arXiv:2408.00392](https://arxiv.org/abs/2408.00392). Polynomial basis
  functions that satisfy the variable-coefficient wave equation to Taylor
  order at a point, with an algorithm to build them. The closest published
  relative of "seeds from time-polynomial solutions", and exactly the
  Taylor route of section 1.5: it needs the coefficients resolved by the
  polynomial degree, which a sub-grid edge is not.

**Chebyshev systems, the well-posedness side.** G. Pólya, "On the
mean-value theorem corresponding to a given linear homogeneous
differential equation", Trans. Amer. Math. Soc. 24 (1922) 312–324,
[doi:10.1090/S0002-9947-1922-1501228-5](https://doi.org/10.1090/S0002-9947-1922-1501228-5)
(the factorisation / disconjugacy result; the AMS PDF could not be
fetched, so its open-access status is unverified); S. Karlin and W. J.
Studden, *Tchebycheff Systems: With Applications in Analysis and
Statistics*, Interscience 1966 ([archive.org record](https://archive.org/details/tchebycheffsyste0000karl));
W. A. Coppel, *Disconjugacy*, Lecture Notes in Math. 220, Springer 1971,
[doi:10.1007/BFb0058618](https://doi.org/10.1007/BFb0058618); R. A. Zalik,
"Another look at Chebyshev systems",
[webhome.auburn.edu/~zalikri/fv/t.pdf](http://webhome.auburn.edu/~zalikri/fv/t.pdf),
which states the ECT structure theorem used in section 1.4 with the page
references into Karlin & Studden. G. Mühlbach, "A recurrence formula for
generalized divided differences and some applications", J. Approx. Theory
9 (1973) 165–172, [doi:10.1016/0021-9045(73)90104-4](https://doi.org/10.1016/0021-9045(73)90104-4):
a Neville–Aitken recurrence for Chebyshev systems, i.e. the analogue of
Fornberg's algorithm for the seeds, should one ever want to avoid the
dense solve. M. H. Schultz and R. S. Varga, "L-splines", Numer. Math. 10
(1967) 345–369, [doi:10.1007/BF02162033](https://doi.org/10.1007/BF02162033):
splines whose pieces lie in ker L.

**The jump case, for contrast.** R. J. LeVeque and Z. Li, "The immersed
interface method for elliptic equations with discontinuous coefficients
and singular sources", SIAM J. Numer. Anal. 31 (1994) 1019–1044,
[doi:10.1137/0731054](https://doi.org/10.1137/0731054) (erratum 32 (1995)
1704): jump conditions folded into modified stencils. B. Martin,
B. Fornberg and A. St-Cyr, "Seismic modeling with
radial-basis-function-generated finite differences", Geophysics 80 (2015)
T137–T146, [doi:10.1190/geo2014-0492.1](https://doi.org/10.1190/geo2014-0492.1);
B. Martin and B. Fornberg, "Seismic modeling with radial basis
function-generated finite differences (RBF-FD): a simplified treatment of
interfaces", J. Comput. Phys. 335 (2017) 828–845,
[doi:10.1016/j.jcp.2017.01.065](https://doi.org/10.1016/j.jcp.2017.01.065):
the dissertation's construction, which section 1.3 shows to be the δ → 0
limit of the seeds. The practical competitor in seismic FD codes is
equivalent-medium parametrisation (averaging the coefficients over the
cell rather than changing the basis), e.g. Geophys. J. Int. 239 (2024)
675, [academic.oup.com/gji/article/239/1/675/7733912](https://academic.oup.com/gji/article/239/1/675/7733912).

**What Brad's intuition coincides with, and what looks unwritten.** The
x-like seed, constant flux through the feature, is the harmonic mean of
Tikhonov–Samarskii, the special elements of Babuška–Osborn and the
harmonic coordinates of Owhadi–Zhang; basis functions from local
solutions are the fitted operators of the 1950s–60s; basis functions that
are time-polynomial solutions are the polynomial Trefftz spaces of the
2010s. Not found in this form: an ordinary equispaced finite-difference
stencil for the time-domain wave equation whose basis is the t = 0 profile
of time-polynomial solutions *integrated through a sub-grid smooth
feature*, with everything else in the scheme left standard, and the
observation that this is one construction from the jump (algebra) to the
resolved edge (Fornberg) with the ODE march in between. That is a
statement about a few hours of searching, not a claim of novelty.

## 4. Two dimensions: the seeds (#34, #38)

### 4.1 Design (#34)

A design note answering the three questions in #27, written before any 2-D
code; §4.2 is what #38 then built. The 2-D acoustic operator
L = (1/ρ) ∇·(K ∇) is the first target; the elastic system of `wave2d` adds
bookkeeping, not ideas.

**The chain survives, the choice of seeds does not come for free.** The
recursion L φ = (lower seeds) and the nested spaces {1} ⊂ ker L ⊂
{L f = const} ⊂ ker L² ⊂ … are dimension-free. What changes is that ker L
is infinite-dimensional in 2-D, so "polynomial-like of degree ≤ d", a
space of dimension (d+1)(d+2)/2 that reduces to the polynomials P_d for
constant coefficients, needs a rule for which L-harmonic functions play
the role of 1, x, y, x² − y², xy, …: the 2d + 1 harmonic polynomials of
degree ≤ d each need an L-harmonic continuation through the feature that
looks like them away from it. Given those, the chain generates the rest
(x² + y², for instance, is the φ with L φ = 4 c_e², anchored like φ₂).

**Straight feature: ODEs in the normal coordinate only.** If the material
depends on the normal coordinate n alone (locally true for any smooth
feature, exactly true for a flat edge), write the seed for the monomial
nᵃ sᵇ (s tangential) as φ = Σ_j g_j(n) sʲ, a polynomial in s. Then

    L (g(n) sʲ) = sʲ L_n g + j (j−1) (K/ρ) g s^{j−2},   L_n = (1/ρ) ∂ₙ K ∂ₙ,

so the coefficient functions solve a triangular chain of 1-D ODEs in n:
the top coefficient g_b satisfies the 1-D seed equation, g_{b−2} picks up
the source j(j−1)(K/ρ) g_b, and so on down. Every seed of a straight
feature is a short list of 1-D marches of exactly the kind `stiff.py` does,
with more right-hand sides. So the answer to "PDEs for each dimension of
each stencil's support" is no: for a straight feature, ODEs in one
coordinate.

**Curved feature: a local problem per stencil, three ways.** In the
feature's own coordinates (n, s) the operator picks up curvature terms
(κ ∂ₙ and the metric factors of the tangential derivative), and the
s-polynomial ansatz no longer closes. Three routes, in increasing
faithfulness and cost: (a) treat the curvature perturbatively, κh ≪ 1 for
a stencil, and add an O(κ) correction to the straight-feature seeds from
one more ODE chain (the same spirit as the "curvature terms are a possible
refinement" note in `wave2d/interface.py`, JCP 2017 Fig. 10); (b) solve a
small boundary-value problem L φ = (lower seed) on a patch around the
stencil on a fine local grid, with the harmonic polynomial as boundary
data, which is the multiscale-FEM / oversampling construction (Hou–Wu) and
the honest "PDE per stencil"; (c) compute the harmonic coordinates F once
globally (Owhadi–Zhang), use polynomials in F as the degree-one seeds and
the chain for the rest. For an edge whose curvature radius is many h,
(a) should be enough; (b) is the robust fallback. This is #42.

**Characteristics away from the interface?** No. Away from the feature
the material is constant, the seeds are the monomials, and nothing
changes; the seeds only differ from monomials on stencils whose nodes see
a varying material, and what "starts" them is the anchor at the
evaluation point, not characteristics. Characteristics would enter only if
one built seeds for the first-order system directly; here each field's
second-order operator defines its seeds, as in 1-D, and the odd seeds
carry the other field's information (f_ttt = C for the u-field's φ₃) the
way `continuity_matrices` routes odd time derivatives through the other
field.

**The elastic system and the "not full rank" subspace.** The existing
interface-aware 2-D stencils (`wave2d/interface.py`, dissertation §3.3)
work in a local tangent/normal frame, replace the polynomial augmentation
of the RBF-FD saddle-point system by piecewise polynomials translated
across the interface with the elastic continuity conditions, generate the
stress basis from the velocity basis through the PDE operator, truncate to
degree p, and solve coupled systems across fields (27 basis functions for
p = 3). The seeds slot into exactly that place: the piecewise-polynomial
values at the stencil nodes become ODE-marched values, the same coupled
saddle-point solve follows, and the Gaussian RBF part stays as it is. The
nested-kernel chain answers the subspace question: "degree ≤ d" means
{f : Lᵐ f ∈ seeds of degree ≤ d − 2m}, and the stress seeds follow from the
velocity seeds by applying the constitutive rows of the operator, as
`interface_basis` already does with polynomials. L is then the 2 × 2
matrix operator of linear elasticity acting on (u, v); the straight-feature
reduction still holds because the isotropic operator keeps its form under
the rotation into the feature's frame.

### 4.2 The elastic seeds of a straight feature (#38, `wave2d/seeds.py`)

**Frame and operator.** As in `wave2d/interface.py`: origin at the
closest edge-centre point, x' tangential, y' normal, and the rotated
fields (u', v', f', g', h') obey eq. 32 unchanged; primes are dropped
below. The material depends on y' only, exactly for a flat edge and
locally for a curved one. Eliminating the stresses from eq. 32 with
λ(y), μ(y), ρ(y) and K = λ + 2μ:

    ρ u_tt = ∂ₓ[K u_x + λ v_y] + ∂ᵧ[μ (u_y + v_x)]
    ρ v_tt = ∂ₓ[μ (u_y + v_x)] + ∂ᵧ[λ u_x + K v_y].

Call the right-hand side, divided by ρ, L(u, v).

**Ansatz.** u = Σ_j a_j(y) xʲ, v = Σ_j b_j(y) xʲ, degree ≤ q in x. The
brackets are the stress rates of eq. 32, each a polynomial in x with
coefficient functions of y:

    K u_x + λ v_y   = Σ_j [K (j+1) a_{j+1} + λ b_j']  xʲ   =: Σ_j F_j xʲ
    μ (u_y + v_x)   = Σ_j [μ (a_j' + (j+1) b_{j+1})]  xʲ   =: Σ_j τ_j xʲ
    λ u_x + K v_y   = Σ_j [λ (j+1) a_{j+1} + K b_j']  xʲ   =: Σ_j σ_j xʲ,

and ∂ₓ shifts the index while ∂ᵧ differentiates the coefficient, so the
xʲ coefficient of ρ L is

    ρ (L)₁,j = (j+1) F_{j+1} + τ_j' = (j+1)(j+2) K a_{j+2} + (j+1) λ b_{j+1}' + τ_j'
    ρ (L)₂,j = (j+1) τ_{j+1} + σ_j'.

Level j is driven by levels j+1 and j+2 only: the system is triangular
from the top x-degree down, and each level is a 2-component second-order
ODE in y for (a_j, b_j). The two tractions τ_j, σ_j are what stays
continuous across a jump (g and h in eq. 32), so they are the flux
variables of the first-order form, the 2-D version of ψ = K φ' in §1.2:

    a_j' = τ_j / μ − (j+1) b_{j+1}
    b_j' = (σ_j − λ (j+1) a_{j+1}) / K
    τ_j' = ρ R₁,j − K (j+2)(j+1) a_{j+2} − λ (j+1) b_{j+1}'
    σ_j' = ρ R₂,j − (j+1) τ_{j+1},

with b_{j+1}' substituted from the second line at level j+1, so λ, μ, ρ
are only ever evaluated, never differentiated. (R₁, R₂) is the chain's
right-hand side, next.

**The chain.** With constant coefficients L maps a monomial pair to
lower-degree monomial pairs:

    L (xᵃ yᵇ, 0) = (1/ρ) [ K a(a−1) x^{a−2} yᵇ + μ b(b−1) xᵃ y^{b−2},  (λ+μ) a b x^{a−1} y^{b−1} ]
    L (0, xᵃ yᵇ) = (1/ρ) [ (λ+μ) a b x^{a−1} y^{b−1},  μ a(a−1) x^{a−2} yᵇ + K b(b−1) xᵃ y^{b−2} ],

i.e. a matrix C over the 2m monomial pairs e (m monomials to degree q,
u-block then v-block), C = (D²)[uv, uv] with D the block operator of
eq. 35 as `pde_operator` builds it, degree-lowering by two and so strictly
triangular in degree. The seed S_e of monomial pair e is the solution of

    L S_e = Σ_e' C[e', e] S_e',      C frozen at the anchor y_e,

with the *seeds* of the lower monomials on the right, exactly
L φ_k = k(k−1) c_e² φ_{k−2} of §1.2 read in 2-D; in the ansatz,
R₁,j = Σ_e' C[e', e] a_j^{(e')} and R₂,j = Σ_e' C[e', e] b_j^{(e')}. All
2m seeds march as one linear system, and the triangularity in degree is
what makes it consistent: the right-hand side of a seed only ever needs
seeds two degrees down. Initial data at y_e are the monomial's jet: for
(xᵃ yᵇ, 0), a_a(y_e) = [b = 0] and a_a'(y_e) = [b = 1], everything else
zero (and the mirror for (0, xᵃ yᵇ)); τ and σ at y_e follow from the jet
and the anchor material. With constant coefficients the right-hand side
is L of the monomial itself (induction on degree) and the linear ODE with
that data has one solution, the monomial: the seeds *are* the monomials,
to 2.5 × 10⁻¹⁴ in `test_constant_material_seeds_are_the_monomial_basis`.

**The system for p = 3, written out.** The stress seeds need velocity
seeds to degree q = p + 1 = 4 (as `interface_basis` expands to degree
p + 1), so there are 15 monomials per component, 30 seeds, x-degrees
j = 0..4, and each seed carries five levels of (a_j, b_j, τ_j, σ_j): a
600-state system. Level by level (K = λ + 2μ, everything a function of
y, R the chain terms):

    j = 4:  a₄' = τ₄/μ            b₄' = σ₄/K              τ₄' = ρR₁,₄                       σ₄' = ρR₂,₄
    j = 3:  a₃' = τ₃/μ − 4 b₄     b₃' = (σ₃ − 4λ a₄)/K    τ₃' = ρR₁,₃ − 4λ b₄'              σ₃' = ρR₂,₃ − 4 τ₄
    j = 2:  a₂' = τ₂/μ − 3 b₃     b₂' = (σ₂ − 3λ a₃)/K    τ₂' = ρR₁,₂ − 12K a₄ − 3λ b₃'     σ₂' = ρR₂,₂ − 3 τ₃
    j = 1:  a₁' = τ₁/μ − 2 b₂     b₁' = (σ₁ − 2λ a₂)/K    τ₁' = ρR₁,₁ − 6K a₃ − 2λ b₂'      σ₁' = ρR₂,₁ − 2 τ₂
    j = 0:  a₀' = τ₀/μ − b₁       b₀' = (σ₀ − λ a₁)/K     τ₀' = ρR₁,₀ − 2K a₂ − λ b₁'       σ₀' = ρR₂,₀ − τ₁

For the seed of (xᵃ yᵇ, 0) only levels j ≤ a are ever non-zero (nothing
above is driven), so the seed's x-degree is that of its monomial, and the
y-structure is where the medium enters. Two seeds in closed form, through
any edge: (y, 0) becomes (∫ μ_e/μ dy, 0) and (0, x) becomes
(∫ (μ_e/μ − 1) dy, x), both with the constant shear stress g = μ_e and
no other stress. That is the 2-D twin of φ₁ = K_e ∫ dξ/K: the traction is
what stays smooth, not the slope.

**Stress seeds and rigid motions.** As `interface_basis` step 3: the
stress seed of a velocity seed is (f, g, h) = (Σ F_j xʲ, Σ τ_j xʲ,
Σ σ_j xʲ), read off the marched state and the flux variables (F_j uses
b_j' from the b-equation), so nothing is differentiated numerically. The
constants in u and v have no stress, and the shear pair (y, 0) and (0, x)
share theirs (g = μ_e, above), through any edge and not just for constant
coefficients; the same three columns are dropped as in the polynomial
case, and `test_rigid_motions_have_no_stress_and_the_shear_pair_shares_one`
checks it through a δ = 0.01 edge. That leaves 27 stress seeds for
p = 3, from the 30 velocity seeds, and the 20 velocity seeds of degree
≤ 3 for the velocity block: the same counts as the dissertation's.

**Anchor and scaling: which won.** The 1-D lesson (§1.2, §2) was that
anchoring at the evaluation point keeps the interpolation block
conditioned. The seeds are therefore anchored at the *evaluation node* in
both coordinates and scaled by r_max, the distance to the farthest
stencil node, so S ≈ Xᵃ Yᵇ in (X, Y) = (x − x_e, y − y_e)/r_max; the
material is sampled at y_e + r_max Y. The polynomial basis of
`interface.py` keeps its origin at the interface point (and the same
r_max). The two conventions are reconciled, not unified: moving the
origin is a change of basis within the same span (`shift_matrix` gives
it, and the tests use it to compare the two bases column by column), so
`interface_weights` is untouched and the jump path stays bit for bit.
What is not a change of basis is where the *normal* coordinate is
anchored, because that is where the jet is imposed and the coefficients
are frozen; there the seeds follow the 1-D lesson. A side benefit: at the
anchor the velocity seeds have the monomials' first derivatives (only X
and Y have one), and the stress seeds' derivatives (f_X, g_X, g_Y, h_Y)
come from the state and the ODE right-hand side at Y = 0; `seed_basis`
returns both jets, in stencil units, so the consumer multiplies by
1/r_max once, as `interface_weights` does with its `sc`. The identity
(f_X + g_Y)/ρ_e = C[const, e] (the rate of a stress seed at the anchor is
the chain's constant term) is checked in
`test_jets_at_the_anchor_are_the_monomial_jets_and_the_chain_constants`.

**Jump limit.** As δ → 0 the state (a, b, τ, σ) passes through the edge
unchanged, which is velocity and traction continuity, and on the far side
the chain holds with the anchor's frozen coefficients, which is what the
continuity matrices encode: matching D^k on both sides at every order
says L_other P_other is the translation of L_std (monomial). So the seeds
should tend to the translated basis of `interface_basis` with the
standard side at the anchor, re-expanded about the anchor. They do, at
first order in δ, for velocity and stress and with the anchor on either
side (`test_jump_limit_recovers_the_interface_basis_at_first_order`;
n = 400 node set, h = 0.05, 19-node stencil on the row nearest the lower
interface, relative to the largest basis value):

| anchor | block | δ = 10⁻³ | 10⁻⁴ | 10⁻⁵ |
|---|---|---|---|---|
| in the band (y' = +h/2) | velocity | 2.3e-2 | 2.4e-3 | 2.4e-4 |
| | stress | 1.25e-2 | 1.24e-3 | 1.24e-4 |
| in the background (y' = −h/2) | velocity | 9.9e-3 | 9.9e-4 | 9.9e-5 |
| | stress | 1.6e-2 | 1.6e-3 | 1.6e-4 |

**Residual.** `test_seeds_satisfy_the_2d_operator_applied_by_finite_differences`
uses none of the module's coefficient bookkeeping: it evaluates every seed
as a 2-D function on a 25 × 401 tensor grid through a δ = 0.01 edge,
applies the second-order elastic operator of `domain.py`'s docstring with
8th-order finite differences in both x and y (exact in x, where the seeds
are polynomials of degree ≤ 4), and compares with ρ Σ C S, the chain
matrix having been checked against its closed form separately. The
relative residual is 2 × 10⁻¹². An earlier version of this test rebuilt
the xʲ coefficient formulas the way `rhs` does and so could not see a
derivation error shared by the two; the adversarial review of PR #46
showed that by injecting one. The present test drops from 2 × 10⁻¹² to
0.23 when the (j+2)(j+1) factor is wrong and to 0.06 when the λ (j+1) b'
term is dropped.

**Conditioning.** On the same real stencil, the 2-norm condition numbers
of the velocity block (38 × 20) and the stress block (57 × 27):

| basis | velocity | stress |
|---|---|---|
| polynomial jump basis, re-expanded about the anchor | 34 | 30 |
| polynomial jump basis as `interface_weights` evaluates it (interface origin) | 32 | 51 |
| seeds, δ = h | 17 | 24 |
| seeds, δ = h/2.5 | 24 | 27 |
| seeds, δ = h/5 | 29 | 29 |
| seeds, δ = h/8 | 31 | 29 |
| seeds, δ = h/20 | 33 | 30 |
| seeds, δ = h/5000 | 34 | 30 |

The seed blocks are as well conditioned as the polynomial block at every
width and tend to it as δ → 0, the extended-Chebyshev-system behaviour of
§1.4 carried over (`test_seed_blocks_are_conditioned_like_the_polynomial_block`).

**Numerics and runtime.** One `SeedChain` per stencil: the 600-state
linear system, DOP853 at rtol 10⁻¹³, one march per side from Y = 0
outward, restarted at every node's Y (so node values are integrated, not
interpolated) and at the edge centres and their ±10δ flanks, as in 1-D.
The flanks matter: a single solve without them differs from the segmented
march by 4 × 10⁻¹⁰ at δ = 10⁻⁴, where the adaptive step has to discover
the edge. 22 to 40 ms per 19-node stencil for δ from h down to h/5000
(the ODE right-hand side is five array operations and one 30 × 30
product; `LayeredMedium2D.material_at` evaluates the blend weight once for
all three parameters). For the n = 900 to 19600 node sets of §5, the
stencils that see an edge number a few hundred to a few thousand, so an
operator costs seconds to a minute. The flat case has a translation
symmetry in x' (every stencil in one fixed row shares y_e and the
material profile, so one march with dense output could serve a whole
row); that is an optimisation for later, not now.

**What #39 gets.** `seed_basis(local, profile, degree)` takes the frame
`operators._local_frames` already builds (node 0 the evaluation node,
origin at the foot point) and a `NormalProfile` from
`normal_profile(medium, interface, x0)`, and returns the velocity seeds
`uv` (2, n, 20) and stress seeds `fgh` (3, n, 27) at the nodes plus both
jets at the anchor: the exact shapes `interface_weights` consumes through
`eval_side` and `deriv_e`, so the seed-augmented weights are the same
coupled saddle-point solve with the polynomial block swapped. One open
point for #39: the polynomial branch also imposes Δ³ p = 0 on its
augmentation (exact for degree ≤ 3). The seeds' sixth derivatives at the
anchor are not zero in a varying medium (they carry the material's
derivatives through the ODE), but hyperviscosity is a stabiliser, not
part of the discretised operator, and the natural choice is to let it
annihilate the seed space as it annihilates the polynomial one, i.e.
impose zero. Whether that is right is an eigenvalue question and belongs
with the spectrum study of #39.

## 5. Two dimensions: results (#36–)

The 2-D chain runs flat first (Brad, 2026-09-19): the principles and the
RBF-FD stability question get settled on the flat two-interface problem of
Part 2, which has an independent reference, before curvature enters (#42).

### 5.1 Smooth flat edges and the normal-incidence reference (#36, PR #43)

`LayeredMedium2D(edge_width=δ)` smooths both interfaces of the band
`[0.25, 0.5)` into tanh transitions of scale δ in the vertical offset,
summed over periodic images in y; λ, μ and ρ are blended linearly in the
tanh weight, so K = λ + 2μ is linear in it and c_p is not. That differs
from the 1-D `LayeredMedium`, which blends c and ρ; the two conventions
meet only in the jump limit, and it is why the reference below maps the
2-D profiles pointwise instead of building a 1-D layer of width 2δ. Two
edges a gap g apart only reach tanh(g / 2δ) of the contrast between them
(99.6% at δ = 0.03 for this band), so the class refuses 4δ > g.

The reference, `wave2d/exact.py: spectral_plane_wave`, is the 1-D
pseudo-spectral solver of section 2 on the image of the problem under
x = 1 − 2y (c′ = 2c_p, ρ′ = ρ/2, impedances unchanged), fed the exact
image of the 2-D initial state (u₁D = −v/Z_p, f₁D = h/Z_p with the
background Z_p), with f recovered exactly from f_t = λ v_y as
f = f₀ + λ/(λ+2μ) (h − h₀). Checks (`tests/test_wave2d_smooth_edges.py`):
the reference tends to the ray sum at first order in δ (max |Δv| = 0.167,
0.090, 0.046, 0.023, 0.012 for δ = 0.016 down to 0.001 at t = 0.3);
doubling the 1-D grid or halving the time step at δ = 0.005 changes v, f,
h by under 3·10⁻¹⁰ at t = 1; and a naive 2-D run at 10,000 nodes through a
δ = 0.02 = 2h edge agrees with it to 7.9·10⁻³ in v at t = 0.3, below the
run's own resolution floor of 1.07·10⁻², while differing from the jump
solution by 0.33. The shortcut f = λ/(λ+2μ) h of the jump case is off by
|h₀| |ratio − ratio_bg|, frozen in time: nothing for the default contrast
(λ = μ on both sides), and 2·10⁻¹⁴, 10⁻⁹, 7·10⁻⁶ at δ = 0.01, 0.02, 0.04
for a band with a different ratio.

### 5.2 Naive baseline: is there a knee in 2-D? (#37, `scripts/wave2d_stiff.py`)

Setup: the flat band with the default contrast, plain RBF-FD everywhere
(30-node stencils, degree-4 augmentation, Δ³ hyperviscosity at the MATLAB
γ), material coefficients sampled at the stencil centres, RK4 at CFL 0.5,
t = 1, the node sets of Part 2 (seed 0). Fixed δ per panel, as in 1-D, so
a knee would appear as n crosses h = δ: δ = 0 (ray-sum reference),
0.0025 (h/δ from 8 to 2.9, never resolved), 0.01 (h/δ from 2 to 0.71) and
0.04 (h/δ ≤ 0.5, resolved everywhere). Errors are relative l2 errors at
t = 1 against the ray sum or the cached 1-D spectral snapshots (4096 nodes
for δ = 0.0025, 1024 otherwise, dt = 5·10⁻⁵). "Floor" is the same pulse on
the same nodes in a uniform medium. Two pulses: the dissertation's
(sharpness 23, centre 0.75) and a wider one (sharpness 15, centre 0.875,
tails 2·10⁻¹⁴ at the edges), for the reason section 2 gave in 1-D.
Runtime 2 min 10 s per pulse, 19 s of it the references (cached after).

Dissertation pulse, error in v:

| δ | n = 2500 | 4900 | 10000 | 19600 | rates |
| --- | --- | --- | --- | --- | --- |
| 0 (jump) | 2.4e-1 | 1.4e-1 | 6.9e-2 | 3.5e-2 | 1.5, 2.1, 2.0 |
| 0.0025 | 2.0e-1 | 1.1e-1 | 5.2e-2 | 2.2e-2 | 1.8, 2.0, 2.5 |
| 0.01 | 1.5e-1 | 8.4e-2 | 3.0e-2 | 8.6e-3 | 1.8, 2.9, 3.7 |
| 0.04 | 1.9e-1 | 8.7e-2 | 3.0e-2 | 8.8e-3 | 2.3, 3.0, 3.6 |
| floor | 1.9e-1 | 9.2e-2 | 3.3e-2 | 1.0e-2 | 2.0, 3.0, 3.4 |

Wider pulse, error in v and the largest spurious |u| (the exact u is 0):

| δ | n = 2500 | 4900 | 10000 | 19600 | rates | max \|u\| at 2500 … 19600 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (jump) | 1.3e-1 | 7.2e-2 | 3.6e-2 | 2.2e-2 | 1.7, 2.0, 1.5 | 9.6e-3, 1.1e-2, 5.7e-3, 5.7e-3 |
| 0.0025 | 8.9e-2 | 4.8e-2 | 2.7e-2 | 1.3e-2 | 1.9, 1.6, 2.2 | 9.5e-3, 1.1e-2, 4.7e-3, 3.4e-3 |
| 0.01 | 6.2e-2 | 2.2e-2 | 5.7e-3 | 1.4e-3 | 3.1, 3.7, 4.2 | 2.7e-3, 1.5e-3, 7.6e-4, 8.5e-5 |
| 0.04 | 5.2e-2 | 1.9e-2 | 5.1e-3 | 1.4e-3 | 2.9, 3.7, 3.9 | 1.2e-3, 4.8e-4, 1.0e-4, 2.4e-5 |
| floor | 5.2e-2 | 1.8e-2 | 4.8e-3 | 1.3e-3 | 3.0, 3.8, 3.8 | 8.9e-4, 4.5e-4, 1.2e-4, 3.0e-5 |

![Naive RBF-FD through smooth edges: error in v and spurious u vs resolution](figures/wave2d_stiff_naive_s15.png)

**What the table says.**

- *The resolution floor is the first thing to see.* With the dissertation
  pulse, only the jump and the never-resolved δ = 0.0025 edge stand out
  from the floor (3.5× and 2.2× at 19,600 nodes); δ = 0.01 sits at the
  floor even at h = 2δ, where the 1-D naive scheme was 400× worse than the
  seeds. The 2-D floor is 10⁻² at our finest node set, against 10⁻⁷ in the
  1-D study, and the pulse can only be widened so far (its tails must
  clear both edges for the ray sum). The wider pulse lowers the floor by
  8× at 19,600 nodes and is the configuration to read.
- *A never-resolved edge behaves like the jump.* At δ = 0.0025 the naive
  error is second order and 10× the floor at 19,600 nodes, at 0.6–0.75 of
  the jump's level: the straddling rows at h/2 see tanh(h/2δ) of the
  contrast, 0.9993 at 2500 nodes and 0.89 at 19,600.
- *A resolved edge costs nothing.* At δ = 0.04 (h ≤ δ/2) the naive error is
  the floor to within 6% at every n, with the floor's rates.
- *The knee is there but small in v.* At δ = 0.01 the naive error exceeds
  the floor by 19%, 17%, 18% and 8% from h = 2δ to h = 0.71δ; taking the
  excess in quadrature, the edge's own contribution falls 3.3e-2, 1.1e-2,
  3.0e-3, 5.2e-4, faster than second order and accelerating, which is the
  knee shape of section 2 seen through a high floor.
- *The spurious u separates the cases far more sharply than v.* The
  naive stencils excite u where a field varies sharply in y, because d/dx
  on scattered nodes does not annihilate such a field exactly; u carries
  none of the pulse's own dispersion error, and the uniform-medium run
  gives its floor (8.9e-4 down to 3.0e-5). Against that floor the jump is
  11×, 24×, 48×, 190× from 2500 to 19,600 nodes and δ = 0.0025 is 11×,
  24×, 39×, 113×: an unresolved edge's error decays like the jump's,
  slower than the floor, so the ratio grows with n. δ = 0.04 equals the
  floor at every n. δ = 0.01 sits 3×, 3×, 6×, 3× above it: on the way to
  jump-like behaviour while h ≥ δ (the ratio rising to 6 at h = δ) and
  falling back once h < δ. That is the knee of section 2 in the one
  quantity the floor does not pollute, as a plateau rather than a drop,
  because at h = 0.71δ the edge is only just resolved.
- *Spectrum* (`scripts/wave2d_eigenvalues.py --edge-width`, 900 flat nodes,
  standard γ): max Re λ = +5.9·10⁻² for the jump and for δ = h/8, and
  +8.2·10⁻⁵ for δ = 2h; RK4 amplification 1.0004, 1.0004, 1.0000. As
  expected: only the centre-sampled coefficients change.

**Consequences for #38–#40.** The seed stencils can at most remove the
excess over the floor: 8–20% in v at δ = 0.01, a factor 10 at δ = 0.0025
(an edge 3–8× thinner than the node spacing), and in spurious u a factor
3–6 at δ = 0.01 and 11–190× for the jump-like cases. So the flat
convergence study of #40 should report max |u| and the δ = 0.0025 column
as its primary evidence, use the wider pulse, and treat the δ = 0.01
v-curve as a secondary check; a "knee plot" in v alone would show little. The 2-D naive scheme is not first order through
an unresolved edge the way 1-D FD4 was: the fixed rows straddling the edge
centre keep the coefficient sampling symmetric, and RBF-FD's error through
a jump is already the "second order, large constant" of Part 2.
