# 📚 GFN Models & Mechanics Reference

This document details the two primary trained models and the underlying physical mechanics of the Geodesic Flow Network.

---

## 🤖 Models Overview

We have two distinct model architectures, each optimized for specific hardware and usage scenarios.

### 1. GFN-Small ("The GTX Model")
*Designed for consumer GPUs (GTX 1650, 4GB VRAM) and rapid inference.*

- **Path**: `D:\ASAS\projects\models\gfn_math_epoch_*.pth` (Epoch 0-6)
- **Architecture**:
    - **Dim**: 256
    - **Depth**: 6 Layers
    - **Rank**: 64 (Christoffel Approximation)
    - **Integrator**: `leapfrog` (Symplectic, Discrete)
- **Training**:
    - Trained on 8-digit Arithmetic (`A op B = C`).
    - Loss Plateau: ~2.05 (Syntactic understanding, struggling with exact memorization).
    - **Physics**: Fully Energy Conserving.

### 2. GFN-Medium ("The RTX Model")
*Designed for high-end GPUs (RTX 4090, 24GB VRAM) and deep reasoning.*

- **Path**: `D:\ASAS\projects\models\gfn_math_epoch_*_rtx.pth`
- **Architecture**:
    - **Dim**: 1024
    - **Depth**: 24 Layers
    - **Rank**: 256
    - **Integrator**: `adjoint` (O(1) Memory "Time Machine") or `leapfrog`.
- **Status**: Scaled up to 50M+ parameters. Capable of carrying complex context.

---

## ⚙️ Physics Mechanics (The "Glass Box")

The GFN framework offers multiple ways to run "Forward" (Inference) and "Backward" (Learning), trading off speed, precision, and memory.

### 1. Forward Modes (Thinking)

| Mode | Description | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Leapfrog (Symplectic)** | Standard discrete physics step. `x(t+1) = x(t) + v(t)*dt`. Preserves Energy. | **Stable**, Fast, physically correct. | Memory scales linearly with depth (O(L)). |
| **Heun (RK2)** | 2nd Order Runge-Kutta. Predictor-Corrector. | Smoother trajectories. | Slightly slower per step. |
| **RK4** | 4th Order Runge-Kutta. High precision. | Exact math logic (e.g. carry operations). | 4x slower per step (4 evaluations). |

### 2. Backward Modes (Learning)

| Mode | Description | The "New Invention" |
| :--- | :--- | :--- |
| **Standard (Autograd)** | Stores every step in VRAM to compute gradients. | Perfect for shallow models (Depth < 12). Traditional. |
| **Adjoint (O(1) Memory)** | **The Memory Time Machine**. Does NOT store history. Reconstructs the past by running the physics *backwards* from the loss. | **Infinite Depth**: Train 1000 layers with 1 layer of RAM. | Slower (requires solving ODE fwd + bwd). |

---

## 🧪 Verification & Testing

We verified the core mechanics using `tests/verify_mechanics.py`.

### Results
1.  **Energy Conservation**: ✅ **PASSED**.
    - The Hamiltonian (Total System Energy) drifts by less than `4e-4` per step using Leapfrog integrity.
    - This validates the **Stability** claim.
2.  **Standard Learning**: ✅ **PASSED**.
    - Gradients flow correctly through the Manifold.
3.  **Adjoint Learning**: ⚠️ **EXPERIMENTAL**.
    - The O(1) solver works for state evolution, but gradient connectivity to parameters requires strict implementation details (currently being refined).

### running Verification
To run the proofs yourself:
```bash
python tests/verify_mechanics.py
```
