# Scientific Technical Audit: Toroidal Synchronization Failure

**Date:** January 24, 2026
**Subject:** Alignment Discrepancies between Theory and Implementation

---

## 1. The Conservation Conflict (Inertia vs State Machine)
**Problem:** A fundamental conflict exists between the physics engine (Inertia) and the required logic (Discrete State Machine). In the Parity Task ($x \to x \oplus u$), the state must transition $0 \to \pi$ and *stop*.
**Analysis:** The model possesses inertia. An impulse (Token 1) imparts velocity. If the subsequent token is 0 (Null Force), the particle conserves momentum and overshoots the target $\pi$, violating the logical requirement of a stable state.
**Conclusion:** The system lacks a native "Proximity Braking" mechanism in the absence of input.

## 2. Structural Defect: Force Multiplication
**Defect:** The implementation in `model.py` applies the embedding Force Impulses at *every* layer depth.
**Impact:** For a model of depth $N=6$, the total impulse is $J = 6 \times F$. This destroys the calibration of the impulse-timestep relation ($F \cdot dt \approx \pi$), causing systematic overshoot.

## 3. Holographic Readout Noise
**Defect:** The `Holographic` readout mode assumes Identity mapping.
**Analysis:** The Parity signal is encoded in Channel 0. However, Channels 1-127 are initialized with random weights. Due to metric coupling (Centrifugal terms), the motion in these unrelated dimensions injects noise forces into Channel 0.
**Impact:** Early training is dominated by geometric noise rather than the signal.

## 4. Christoffel Coupling Analysis
**Observation:** In a curved Toroidal manifold ($R, r$), motion in azimuth ($\phi$) generates centrifugal force in polar angle ($\theta$).
**Conclusion:** We are attempting to measure a linear clock (Parity) within a system behaving like a complex rotating body. This requires either decoupling the dimensions or supervising the full state space.

## 5. Singularity Resolution Failure
**Observation:** The "Clutch" mechanism relies on a potential threshold to increase effective mass (friction).
**Defect:** If the Leapfrog integrator uses a large timestep ($dt=0.3$), the particle may "jump over" the high-friction region in a single step, failing to engage the brake.

---

## 6. Corrective Action Plan
1.  **State-Dependent Friction:** Implement a potential well $U(x)$ at $0$ and $\pi$ to act as a passive attractor.
2.  **Coherent Readout:** Align the Embedding layer to inject force only into the dimensions actively monitored by the Readout layer.
3.  **Conditional Hamiltonian:** Enforce energy conservation only during padding/silence tokens, allowing dissipation during active computation.
