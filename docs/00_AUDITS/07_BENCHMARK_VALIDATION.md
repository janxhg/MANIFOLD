# Benchmark Methodology Validation

**Date:** January 24, 2026
**Script:** `vis_gfn_superiority.py`
**Subject:** Validation of Telemetry and Comparison Metrics

---

## 1. Overview
This report validates the methodology used in the comparative analysis between the Manifold GFN and standard Transformer architectures. The benchmark focuses on the **Parity Task** (Sequential Modulo-2 Sum) as a proxy for infinite-horizon state tracking.

---

## 2. Data Collection Methodology

### 2.1 Training Configuration (Manifold)
*   **Optimizer:** `RiemannianAdam` ($LR=10^{-4}$).
*   **Loss Composition:**
    *   **Semantic ($L_{MSE}$):** Primary prediction error.
    *   **Geodesic ($L_g$):** Curvature regularization ($\lambda=0.001$).
    *   **Hamiltonian ($L_h$):** Conservation constraint ($\lambda=0.01$).
*   **Stability:** Gradient clipping at $0.05$ is validated as necessary for high-curvature manifold training.

### 2.2 Implicit Readout Alignment
The benchmark utilizes a geometric mapping for binary targets:
*   **Mapping:** $\{0, 1\} \to \{-1, 1\}$ (or $\{-\pi/2, \pi/2\}$).
*   **Validation:** This zero-centered mapping is critical for symmetric potential well formation, preventing bias towards the coordinate origin.

### 2.3 Scaling Metrics
*   **Inference:** Validates $O(1)$ memory complexity via `PerformanceStats.measure_peak_memory` during streaming evaluation ($L=20 \to 2000$).
*   **Comparison:** Contrasts against Transformer $O(N^2)$ attention scaling and $O(N)$ KV-cache scaling.

---

## 3. Observations
*   **Hamiltonian Feedback:** The model correctly utilizes `return_velocities=True` to compute energy conservation metrics.
*   **Dimensionality:** The parity signal is verified to be encoded in the primary dimension ($d_0$) of the 128-dimensional latent space, confirming the efficacy of the holographic channel.

---

## 4. Conclusion
The benchmarking suite provides a rigorous and fair comparison of architectural capabilities. The metrics collected accurately reflect the theoretical advantages of the Recurrent Manifold regarding memory efficiency and state persistence.
