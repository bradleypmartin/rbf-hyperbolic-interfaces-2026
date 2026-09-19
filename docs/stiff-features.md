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
`tests/test_wave1d_stiff.py`.

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
