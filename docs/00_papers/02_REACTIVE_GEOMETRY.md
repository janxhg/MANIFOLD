# Reactive Geometry: Energy‑Modulated Curvature for Stable Neural Geodesic Flow

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
We introduce **Reactive Geometry**, a principled extension of Riemannian modeling in which the latent manifold stiffens in response to the instantaneous kinetic energy of the state. The metric acts as a self‑regulating substrate: high energy (semantic uncertainty) increases curvature and friction to brake chaotic motion; low energy (semantic certainty) allows inertial geodesic reasoning. Concretely, reactive curvature appears as a bounded multiplicative scaling of Christoffel interactions computed from a low‑rank geometric operator, combined with state‑ and input‑dependent dissipation gates and optional singularity potentials that stabilize discrete logical transitions. We present the mathematical formulation, derive the effective acceleration, detail structure‑preserving discretizations, and outline training objectives aligned with periodic targets and energy conservation.


## 1. Introduction

Riemannian manifolds provide structure for latent computation, yet fixed geometries cannot capture the subjective “effort” of reasoning. In Reactive Geometry, the manifold responds to the state co‑vector’s energy, producing a closed‑loop control: energy → curvature → braking → reduced energy. This establishes a physically grounded path to uncertainty handling and long‑horizon stability in neural geodesic flow.


## 2. Formalism

### 2.1 Low‑Rank Curvature Operator
Let $x\in\mathbb{R}^d$ be coordinates and $v=\dot{x}$ the velocity. The Christoffel interaction is parameterized by a low‑rank, symmetric operator:
$$
\Gamma^k_{\!\,ij}(x) \approx \sum_{r=1}^{R}\,\lambda_{kr}\,U_{ir}\,U_{jr},
$$
with diagonal saturation to bound extreme values. In practice, the projection $p=vU$ is used to scale curvature, ensuring numerical robustness by attenuating responses with a factor like $1/(1+\|p\|)$, and applying a smooth saturation (e.g., $\tanh$) to clip large forces.

### 2.2 Reactive Curvature (Plasticity)
The effective connection multiplies base curvature by a bounded function of kinetic energy:
$$
\Gamma_{\text{eff}}(x,v) \;=\; \Gamma_{\text{base}}(x)\,\big(1+\alpha\,\tanh K\big),
\qquad K=\tfrac12\,\|v\|^2,
$$
where $\alpha\!\ge\!0$ is a plasticity coefficient. As $K$ increases, curvature stiffens and turns motion more strongly, acting as a self‑braking mechanism that reduces oscillations and runaway drift.

### 2.3 Dissipation Gates (“The Clutch”)
Reactive Geometry uses a conformal symplectic dissipation term to regulate state rewriting:
$$
\mu(x,u) \;=\; \sigma\!\big(W_{\text{state}}\,[\sin x,\cos x] + W_{\text{input}}\,u\big)\cdot \mu_{max},
$$
with periodic features $[\sin x,\cos x]$ for toroidal topologies and an input‑dependent component coupled to the token force $u$. Small $\mu$ keeps memory conservative; large $\mu$ damps momentum to allow rapid overwriting.

### 2.4 Singularity Potentials (Optional)
Discrete logical “flips” can be stabilized by localized potentials:
$$
S(x) \;=\; \sigma\!\big(V(x)\big),\qquad
\Gamma_{\text{eff}} \;\leftarrow\; \Gamma_{\text{eff}}\cdot\!\big(1+(S(x)-S_0)\,(\beta-1)\big),
$$
where $V$ is a learned position‑gate, $S_0$ a threshold, and $\beta>1$ the singularity strength. The effect is a soft, differentiable intensification of curvature near high‑certainty regions, forming “event horizons” that trap the state after decisive transitions.

### 2.5 Effective Acceleration
The latent dynamics combine forcing, curvature, and dissipation:
$$
a(x,v,u) \;=\; F(x,u) \;-\; \Gamma_{\text{eff}}(x,v) \;-\; \mu(x,u)\,v,
$$
where $F$ is a token‑to‑force mapping. This form yields a conservative memory mode ($\mu\!\approx\!0$) and a dissipative rewrite mode ($\mu\!\gg\!0$), both modulated by reactive curvature from $K$.


## 3. Geometry–Topology Interplay

Reactive Geometry is compatible with non‑Euclidean topologies. In toroidal settings:
- Coordinates are periodic ($x\!\bmod 2\pi$), preventing edge discontinuities.
- Periodic features $[\sin x,\cos x]$ feed friction and potential gates.
- The diagonal torus metric blocks (inner/outer angles) produce centrifugal/Coriolis‑like terms that naturally regulate phase motion.


## 4. Discretization and Stability

### 4.1 Structure‑Preserving Integrators
Reactive dynamics are discretized via energy‑preserving schemes:
- Leapfrog/Verlet (kick‑drift‑kick) for robust qualitative stability.
- Higher‑order symplectic compositions (Yoshida, Forest‑Ruth, Omelyan) in smooth regimes.
- Heun/RK2 as a non‑symplectic but stable baseline.

A typical Leapfrog step with periodic wrapping is:
$$
v_{\tfrac12}=v+\tfrac12\,dt\,a(x,v,u),\quad
x'=\operatorname{wrap}\!\big(x+dt\,v_{\tfrac12}\big),\quad
v'=v_{\tfrac12}+\tfrac12\,dt\,a(x',v_{\tfrac12},u).
$$

### 4.2 Saturation and Clamps
To avoid numerical explosion under non‑smooth forces, curvature outputs are softly saturated (e.g., via $\tanh$) and clamped within bounded ranges. Dissipation coefficients use sigmoid activation and a fixed scale to keep updates within safe limits. These measures, together with reactive modulation, reduce oscillations and help maintain energy budgets.

### 4.3 Time‑Scale Adaptation
Reactive Geometry can be paired with learned per‑head time scales and optional dynamic time gating, reducing $dt$ in complex regions and increasing it in near‑flat geometry, improving stability without extra memory.


## 5. Implementation Overview

- **Curvature operator:** low‑rank symmetric parameterization with projection‑based scaling and smooth saturation.
- **Reactive curvature:** multiplicative plasticity using a bounded function of kinetic energy.
- **Friction gates:** state‑dependent via periodic features and input‑dependent via the force; both combined through sigmoid activation.
- **Singularity potentials:** differentiable intensification of curvature near high‑certainty regions using a learned position gate and soft thresholding.
- **Topology:** periodic wrapping for toroidal coordinates; periodic features in gates and potentials; diagonal torus metric blocks for stabilizing interactions.
- **Execution paths:** reliable Python loops for correctness; optional fused GPU kernels; optional scan‑based parallelization for training throughput, preserving functional behavior with bounded numerical differences.


## 6. Training Objectives

Reactive Geometry integrates task losses with physics‑informed regularization:
- **Task loss:** cross‑entropy for discrete prediction, or periodic targets using toroidal or bounded phase losses.
- **Hamiltonian penalty:** discourages spurious energy creation by penalizing $|E_{t+1}-E_t|$, with energy computed under the metric blocks $E_t=\tfrac12\,v_t^\top g(x_t)\,v_t$ when available.
- **Geodesic regularization:** tempers curvature excursions via a mean‑squared curvature term.
- **Symmetry regularization (optional):** encourages consistent geometric responses across isomeric heads.


## 7. Empirical Behavior

Reactive Geometry consistently reduces drift by braking high‑energy trajectories and stabilizing discrete transitions with optional singularity potentials. On cyclic algorithmic tasks, it supports long‑horizon phase tracking and robust extrapolation, especially in combination with toroidal topologies and structure‑preserving integrators. In non‑smooth regimes, lower‑order symplectic or RK2 steps often outperform high‑order explicit schemes.


## 8. Discussion and Limitations

Reactive manifolds relate to Finslerian extensions by allowing velocity‑dependent geometry, yet remain practical via simple bounded scaling of curvature and gates. Singularities provide useful stabilization for discrete logic but must be softly gated to preserve differentiability. CUDA fused paths and parallel scans yield throughput gains, though strict floating‑point parity with sequential updates is not guaranteed. Formal convergence proofs for learned curvature and reactive flows are an open direction.


## 9. Conclusion

Reactive Geometry equips neural geodesic flow with an internal feedback loop: energy modulates curvature and friction, which in turn regulates energy. This yields stable long‑horizon behavior, principled uncertainty handling, and improved alignment between geometry and computation. Combined with topology‑aware modeling and structure‑preserving discretization, it forms a foundation for constant‑memory, physically grounded sequence reasoning.


**References**  

[1]  Friston, K. (2010). The free‑energy principle: a unified brain theory?  
[2]  Amari, S. (2016). Information Geometry and Its Applications.  
[3]  Einstein, A. (1916). The Foundation of the General Theory of Relativity.  
[4]  Bao, D., Chern, S. S., & Shen, Z. (2000). An Introduction to Riemann‑Finsler Geometry.  
[5]  Sitzmann, V., et al. (2020). Implicit Neural Representations with Periodic Activation Functions.  
[6]  Nicolis, G., & Prigogine, I. (1977). Self‑Organization in Non‑Equilibrium Systems.  
[7]  Bejan, A. (2000). Shape and Structure, from Engineering to Nature.  
