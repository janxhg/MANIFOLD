# The Hyper‑Torus: Topology‑Aligned Geometry for Constant‑Memory Sequence Modeling

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
Cyclic reasoning tasks (parity, modular arithmetic, periodic phase tracking) are poorly expressed in Euclidean latent spaces. We present the **Hyper‑Torus**, a topology‑aligned manifold design that embeds the latent state into a product of circles $T^n = (S^1)^n$ with a physically meaningful Riemannian metric. Token processing becomes forced geodesic motion with symplectic discretization and periodic boundary conditions. The resulting architecture preserves information as phase and momentum, achieving constant‑memory inference and long‑horizon stability. We formalize the geometry, derive the geodesic equations (including Christoffel interactions and friction gates), describe the discretization via energy‑preserving integrators, and detail training losses for periodic targets. Empirical evaluations on algorithmic tasks demonstrate robust extrapolation and reduced drift relative to non‑geometric baselines.


## 1. Motivation and Overview

Modern sequence models often store an explicit history (e.g., KV caches), causing inference memory to grow with sequence length. Many algorithmic tasks, however, are naturally expressed as phase evolution (e.g., XOR parity as a half‑rotation; counters as winding numbers). In such settings:
- Euclidean embeddings require continuous “effort” to remain on cyclic trajectories.
- Periodicity is brittle near boundaries, leading to drift and aliasing.

The Hyper‑Torus addresses these issues by aligning the manifold with the task: states are represented as phases on $S^1$, coupled into $T^n$, and evolved by physically structured updates. Symplectic discretization preserves qualitative dynamics; periodic boundary conditions remove edge discontinuities; friction gates implement controlled forgetting for regime switching.


## 2. Geometric Preliminaries

### 2.1 The Torus
We define the latent configuration space as a product of circles:
$$
\mathcal{M} \cong T^n = \underbrace{S^1 \times \cdots \times S^1}_{n\ \text{times}} ,
$$
with local coordinates $x = (\theta_1,\phi_1,\theta_2,\phi_2,\dots)$ in paired blocks. Each pair $(\theta,\phi)$ encodes a two‑phase subsystem (inner/outer angles).

### 2.2 Riemannian Metric
Following a standard torus of revolution, we adopt a diagonal metric
$$
g(\theta,\phi) = \operatorname{diag}\big(r^2,\ (R + r\cos\theta)^2\big),
$$
and tile it across coordinates, yielding a block‑diagonal $g(x)$ for $T^n$. Here $R$ is the major radius (global scale) and $r$ the minor radius (local scale). This geometry produces curvature terms that act as stabilizing “geometric forces”.

### 2.3 Geodesic Equations with Forcing and Dissipation
Let $x \in \mathcal{M}$ and velocity $v = \dot{x}$. The forced geodesic dynamics are
$$
\ddot{x}^k + \Gamma^k_{ij}(x)\,\dot{x}^i\dot{x}^j + \mu^k(x,u)\,\dot{x}^k = F^k(x,u),
$$
where $\Gamma$ are Christoffel symbols induced by $g$, $F$ is a token‑driven force, and $\mu\!\ge\!0$ is a dissipation coefficient (“clutch”) that gates between conservative memory and rapid rewriting. For the diagonal torus metric, representative non‑zero symbol patterns for each $(\theta,\phi)$ block include
$$
\Gamma^{\theta}_{\phi\phi} \propto (R + r\cos\theta)\,\sin\theta / r,\quad
\Gamma^{\phi}_{\theta\phi} = \Gamma^{\phi}_{\phi\theta} \propto -\,\frac{r\sin\theta}{R + r\cos\theta},
$$
producing centrifugal and Coriolis‑like effects that naturally regulate phase motion.


## 3. Periodic Boundary Conditions and Distances

### 3.1 Boundary Wrapping
Periodic coordinates are wrapped modulo $2\pi$ component‑wise:
$$
x \leftarrow x \bmod 2\pi .
$$
This removes discontinuities at chart boundaries and prevents unbounded drift.

### 3.2 Toroidal Distance
Prediction targets that live on $T^n$ use the shortest angular distance
$$
d_{\text{torus}}(x_1,x_2) = \min\!\big(|\Delta|,\ 2\pi - |\Delta|\big),\quad \Delta = x_1-x_2,
$$
and losses such as $L_{\text{torus}} = \|d_{\text{torus}}(x_{\text{pred}},x_{\text{target}})\|_2^2$ or the bounded phase loss $L_{\text{phase}} = 1 - \cos(x_{\text{pred}} - x_{\text{target}})$ for smooth gradients near the wrap.


## 4. Token‑to‑Force Mapping and Multi‑Head Geometry

Tokens are embedded into per‑step forces $F_\theta(u_t)$ that act on $v$ and, via $\Gamma(x,v)$, deform trajectories. The latent state is split into $H$ heads. Each head owns a curvature module and integrator step; heads are mixed back linearly. This parallel decomposition yields:
- Independent sub‑geometries for specialized reasoning channels.
- Learned per‑head time scales to adapt resolution to local complexity.


## 5. Friction Gates and Reactive Curvature

### 5.1 Thermodynamic Gating (“The Clutch”)
We parameterize dissipation as $\mu(x,u) = \sigma(W_{\text{state}}\,[\sin x,\cos x] + W_{\text{input}}\,u)$, where $[\sin x,\cos x]$ provides periodic features. Small $\mu$ keeps memory conservative; large $\mu$ rapidly damps momentum to overwrite state.

### 5.2 Reactive Curvature
Curvature can be modulated by kinetic energy $K=\tfrac12\|v\|^2$ via
$$
\Gamma_{\text{eff}}(x,v) = \Gamma_{\text{base}}(x) \cdot \big(1 + \alpha\,\tanh K\big),
$$
increasing geometric resistance in high‑energy regimes and improving stability through difficult transitions.

### 5.3 Singularity Potentials (Optional)
Logical bottlenecks may be modeled as localized potentials that strengthen curvature near thresholds, imitating “event horizons” that stabilize discrete flips.


## 6. Discretization: Energy‑Preserving Integrators

We discretize the dynamics with structure‑preserving schemes:
- Symplectic/Verlet and Leapfrog (kick‑drift‑kick).
- Higher‑order symplectic compositions (Yoshida, Forest‑Ruth, Omelyan) for smoother regimes.
- Heun/RK2 as a robust non‑symplectic baseline.

A typical Leapfrog step for $(x,v)$ with torus boundary is
$$
v_{\tfrac12} = v + \tfrac12\,dt\,a(x,v,u),\quad
x' = \operatorname{wrap}\big(x + dt\,(v_{\tfrac12})\big),\quad
v' = v_{\tfrac12} + \tfrac12\,dt\,a(x',v_{\tfrac12},u),
$$
where $a(x,v,u) = F(x,u) - \Gamma(x,v) - \mu(x,u)\,v$ and $\operatorname{wrap}(\cdot)$ applies the $2\pi$ periodic boundary. This preserves qualitative phase‑space structure and prevents long‑horizon drift.


## 7. Parallelization (Optional)

For training throughput, first‑order affine approximations of the recurrence
$$
v_t = A_t\,v_{t-1} + b_t,\qquad
x_t = x_{t-1} + dt_t\,v_t,
$$
can be evaluated via associative parallel scans on GPUs. This reduces serial depth while maintaining functional agreement with sequential updates.


## 8. Training Objectives for Periodic Targets

We combine task losses with physics‑informed regularization:
- Cross‑entropy or toroidal/circular distance losses for outputs on $T^n$.
- Hamiltonian loss $L_H$ to penalize spurious energy creation:
$$
L_H = \lambda_H\,\mathbb{E}\big[\,|E_{t+1} - E_t|\,\big],\quad
E_t = \tfrac12\,v_t^\top g(x_t)\,v_t .
$$
- Geodesic curvature regularization to temper excessive curvature excursions.
- Optional soft‑symmetry (Noether) terms to align isomeric heads.


## 9. Complexity and Memory

Hyper‑Torus maintains a fixed‑size state $(x_t,v_t)$ independent of sequence length $L$:
$$
\text{Memory} = O(1),\qquad
\text{Compute} \approx O(L\cdot d^2)\ \text{(dense mixing)} .
$$
Symplectic discretization reduces gradient pathologies; periodic wrapping eliminates boundary artifacts, supporting stable infinite‑horizon reasoning.


## 10. Empirical Behavior

On cyclic algorithmic tasks (e.g., cumulative parity), Hyper‑Torus exhibits:
- Stable phase tracking visualized as limit cycles on periodic coordinates.
- Strong extrapolation far beyond training sequence lengths.
- Reduced drift versus non‑geometric baselines through boundary‑aware updates and energy‑preserving integration.


## 11. Discussion and Limitations

While torus topology aligns well with cyclic reasoning, non‑smooth dynamics (sharp logical transitions) can challenge high‑order explicit schemes; lower‑order symplectic or robust RK2 steps often perform best. Learned friction and reactive curvature provide practical stabilization but introduce task‑dependent hyperparameters. Formal convergence proofs for learned curvature remain open research.


## 12. Conclusion

Hyper‑Torus reframes sequence modeling as phase‑space evolution on $T^n$. By matching topology to task and respecting geometry in discretization, the architecture preserves information as momentum and winding rather than as explicit memory buffers. This yields constant‑memory inference, robust long‑horizon stability, and a principled path to physically grounded neural reasoning.


**References**  

[1]  Riemann, B. (1854). Über die Hypothesen, welche der Geometrie zu Grunde liegen.  
[2]  Arnold, V. I. (1989). Mathematical Methods of Classical Mechanics.  
[3]  Hairer, E., Lubich, C., & Wanner, G. (2006). Geometric Numerical Integration.  
[4]  Chen, R. T. Q., et al. (2018). Neural Ordinary Differential Equations.  
[5]  Bronstein, M. M., et al. (2021). Geometric Deep Learning.  
[6]  MacKay, D. J. C. (2003). Information Theory, Inference, and Learning Algorithms.  
[7]  Yoshida, H. (1990). Construction of higher order symplectic integrators.  
[8]  Omelyan, I. P., et al. (2002). Symplectic algorithms for molecular dynamics equations.  
[9]  Dinh, L., et al. (2014). NICE: Non‑linear Independent Components Estimation.  
[10]  Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows.  
