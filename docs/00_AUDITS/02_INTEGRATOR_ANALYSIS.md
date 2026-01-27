# Technical Audit: Numerical Integrators

**Date:** January 24, 2026
**Component:** `gfn.integrators`
**Subject:** Comparative Analysis of Symplectic and Runge-Kutta Schemes

---

## 1. Overview
The Integrators Module implements the core physics engine responsible for evolving the latent state trajectory $\mathcal{T}(x, v, t)$. It provides a suite of algorithms optimized for different regimes of the speed-precision trade-off, including strictly Hamiltonian (energy-conserving) and high-order Runge-Kutta solvers.

---

## 2. Methodology Analysis

### 2.1 Symplectic Integrators (Energy Preservation)
These solvers are designed for reversible Hamiltonian dynamics, ensuring $\Delta H \approx 0$ over long horizons.
*   **Leapfrog / Verlet (2nd Order):** Implements time-reversible area-preserving flow. Used as the baseline for infinite-context memory.
*   **Yoshida (4th Order):** Utilizes sub-step composition to cancel 2nd-order error terms. Provides high-fidelity conservation for complex curvature.
*   **Consistency Check:** The CUDA kernel `recurrent_manifold_fused.cu` implements a fused Euler-Symplectic step, mathematically equivalent to the Leapfrog drift-kick operator in the limit of small $dt$.

### 2.2 Runge-Kutta Integrators (High Precision)
These solvers prioritize local truncation error minimization over global energy conservation.
*   **Heun (2nd Order):** Default predictor-corrector scheme used in stable training regimes.
*   **RK4 (4th Order):** Standard high-precision solver.
*   **Dormand-Prince 5 (DP5):** Adaptive step-size solver used for "Golden Validation" benchmarks to detect drift in lower-order models.

### 2.3 Neural & Flow Integrators
*   **Neural Controller:** An MLP-based step-size controller predicting optimal $dt(state)$, enabling adaptive temporal resolution ("Bullet Time") in regions of high curvature.
*   **Coupling Flow:** Based on NICE/RealNVP architectures, guaranteeing a Jacobian Determinant of exactly 1.0, ensuring perfect volume preservation in phase space.

---

## 3. Mathematical Consistency Verification

### 3.1 Force Evaluation
All integrators correctly interface with the `Christoffel` operator to retrieve geometric acceleration inputs:
$$ a = F_{ext} - \Gamma(v, v) - \mu v $$

### 3.2 State Update Logic
Update steps adhere strictly to Hamilton's equations of motion:
$$ \dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q} $$

### 3.3 Hardware Acceleration
High-order schemes (Yoshida, RK4) have been successfully ported to fused CUDA kernels (`src/integrators/*.cu`), ensuring numerical parity with the Python reference implementations within floating-point tolerance.

---

## 4. Conclusion
The integrator suite is robust, mathematically consistent, and optimized for both research (Python) and production (CUDA) environments. The implementation correctly supports the distinct requirements of energy conservation (Memory) and precise trajectory tracking (Logic).
