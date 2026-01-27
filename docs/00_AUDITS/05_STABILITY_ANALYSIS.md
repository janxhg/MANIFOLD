# System Stability Analysis: Convergence & Oscillations

**Date:** January 24, 2026
**Comparison:** Backup Revision vs. Current Manifold Implementation

---

## 1. Executive Summary
The current system implementation exhibits high-frequency oscillations and gradient instability (NaNs) during training. This audit identifies four concurrent factors contributing to this divergence:
1.  **Curvature Scale:** Parameter initialization magnitude increased by 150x.
2.  **Normalization:** Removal of `LayerNorm` on position coordinates ($x$).
3.  **Dynamics:** Transition from Damped (Dissipative) to Undamped (Oscillatory) forcing.
4.  **Differentiation:** Adoption of custom CUDA Adjoint methods over Autograd.

---

## 2. Component Analysis

### 2.1 Geodesic Dynamics (Acceleration Equation)
| Feature | Backup (Stable) | Current (Unstable) | Impact Analysis |
| :--- | :--- | :--- | :--- |
| **Equation** | $a = F - \Gamma - \mu v$ | $a = F - \Gamma - kx$ | **Critical:** Current version lacks velocity-dependent damping ($\mu v$). Spring force ($kx$) introduces potential energy but does not dissipate kinetic energy. |
| **Curvature Init** | $U, W \sim \mathcal{N}(0, 10^{-3})$ | $U, W \sim \mathcal{N}(0, 0.15)$ | **Severe:** Curvature scales as $\Gamma \sim U^2 W$. A 150x weight increase results in a $\approx 3.3 \times 10^6$x increase in effective curvature forces. |
| **Constraint** | Soft-clamp $|v| \le 5.0$ | Strict $|v| = 1.0$ | Current implementation enforces unit injection rigidly. |

### 2.2 Normalization Scheme
*   **Backup:** Applied `LayerNorm(x)` pre-integration.
*   **Current:** Identity mapping on $x$.
*   **Impact:** Without normalization, coordinate values ($x$) grow unbounded. In reactive manifolds, curvature $\Gamma(x)$ often scales with state magnitude, leading to exponential feedback loops.

---

## 3. Implementation Verification

### 3.1 Backpropagation Through Time (BPTT)
*   **Method:** Custom CUDA Kernel `recurrent_manifold_backward`.
*   **Verification:** GradCheck analysis confirms the C++ BPTT logic is correct for short sequences ($L=5$).
*   **Issue:** The "Adjoint" method requires re-computing forward states during the backward pass. Floating-point drift between $x_{fwd}$ and $x_{recomputed}$ amplifies gradient errors in high-curvature regimes ($10^{-7} \to 10^{-3}$).

### 3.2 Recurrence Context
*   **Current:** Propagates `context` (Gate Latents) across timesteps.
*   **Risk:** This creates an implicit RNN structure within the gating mechanism. Saturation of gate values leads to vanishing gradients for the entire sequence.

---

## 4. Root Cause Determination
The instability is primarily driven by **Curvature Explosion**. The aggressive initialization ($0.15$) combined with the non-linear term $(U^T v)^2$ generates acceleration vectors $|a| > 100$. The discrete integrator (Leapfrog) fails to resolve such high-frequency dynamics, violating the Courant-Friedrichs-Lewy (CFL) condition equivalent for manifolds.

## 5. Remediation Roadmap
1.  **Re-Normalization:** Restore `LayerNorm` within the `MLayer` block.
2.  **Geometry Deflation:** Reset $U, W$ initialization to $\sigma=0.05$.
3.  **Damping Restoration:** Re-introduce the `forget_gate` (Dynamic Friction) to dissipate excess kinetic energy.
4.  **Optimization:** Reduce Learning Rate to $10^{-4}$ for the initial warmup phase.
