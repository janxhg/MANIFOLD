# Thermodynamic Gating: Dissipative Coupled Dynamics for Controllable Memory in Hamiltonian Neural Networks

**Author:** Joaquin Stürtz  
**Date:** January 24, 2026

**Abstract**  
Hamiltonian Neural Networks (HNNs) are characterized by the preservation of phase-space volume, a property that theoretically enables infinite-horizon information persistence. However, strict energy conservation poses a significant challenge for state updating and contextual switching: a purely conservative system cannot "forget" or relax into new semantic attractors without perpetual oscillation. We introduce **Thermodynamic Gating**, an architecture that dynamically couples a Hamiltonian latent space to a dissipative heat bath. By parameterizing the friction coefficient as a non-linear co-vector field relative to the latent state and input force, the model can transition between a **Persistent Hamiltonian Regime** (energy-conservative storage) and a **Relaxed Lagrangian Regime** (dissipative state update). We prove that this "Thermodynamic Clutch" allows for the resolution of the sequential "stopping problem" in continuous neural manifolds, enabling the synthesis of stable symbolic states in a differentiable flow.

---

## 1. The Paradox of Infinite Memory

### 1.1 Conservation vs. Updating
Standard Recurrent Neural Networks (RNNs) utilize multiplicative gates to discard historical information. While effective for forgetting, this mechanism inevitably leads to the "vanishing gradient" problem by contractively mapping the state space. Conversely, Hamiltonian systems satisfy Liouville's Theorem, ensuring that gradients flow without decay. This conservation, while ideal for memory, is fatal for **Context Resolution**: a system that never loses energy can never settle into a specific logical result; it will oscillate indefinitely around its semantic attractors.

### 1.2 The Principle of Selective Irreversibility
Intelligence requires the ability to switch between being an **Isolated System** (where information is conserved) and an **Open System** (where information is discarded and entropy is produced). We hypothesize that state-dependent dissipation is the physical foundation of the "Forget Gate."

---

## 2. Mathematical Formalism: The Dissipative Hamiltonian Update

We extend the standard Hamiltonian equations of motion by the addition of a non-conservative damping force $\mathcal{F}$ governed by the **Thermodynamic Gate coefficient** $\mu \ge 0$:

$$ \dot{q} = \frac{\partial \mathcal{H}}{\partial p} $$
$$ \dot{p} = -\frac{\partial \mathcal{H}}{\partial q} - \mu(q, u) \cdot p $$

The term $\mu(q, u)$ is a learned mapping from the current state and input force to a scalar dissipation factor.

### 2.1 The Two Operational Regimes
1.  **Memory Mode ($\mu \to 0$):** The system is physically isolated. Energy is conserved, and the information is carried through the manifold as persistent momentum.
2.  **Update Mode ($\mu \gg 0$):** In the presence of a significant new forcing term $u$, the system increases $\mu$, effectively "braking" the state co-vector. This dissipates the kinetic energy of previous states and allows the particle to relax into the new coordinate defined by the current input, satisfying the requirements for a **Thermodynamic State Update**.

---

## 3. Information Entropy and Landauer's Principle

The Thermodynamic Gating mechanism provides a physical realization of **Landauer's Principle**, which states that the erasure of one bit of information requires the dissipation of $kT \ln 2$ heat. In our framework, "forgetting" is not a numerical artifact, but a substantive physical process of energy transfer from the latent "thought" state into a dissipative reservoir.

### 3.1 Ballistic State Transitions
This mechanism enables a "Dash-and-Stop" mode of operation:
1.  **Impulse:** An external force injects an energy burst into the state.
2.  **Inertial Transfer:** The gate remains closed, allowing the state to traverse the manifold toward a distant semantic region with zero energy loss.
3.  **Terminal Damping:** Upon arrival at the target attractor, the thermodynamic gate opens, rapidly dissipating the excess energy and "latching" the state into the new symbolic configuration.

---

## 4. Empirical Evaluation of State Damping

Analysis of trained GFN models reveals a high degree of **Dissipative Sparsity**:
*   **Stationary States:** During long sequences of identical or irrelevant tokens, $\mu$ remains at a baseline near zero, confirming the efficiency of the conservative memory flow.
*   **Contextual Transitions:** At points of high semantic shift or logical bit-flips, $\mu$ exhibits narrow, high-intensity spikes. This confirms that the model has learned to expend entropy only at the specific points where information must be rewritten.

---

## 5. Conclusion

Thermodynamic Gating bridges the fundamental gap between **Conservation Laws** and **Information Theory**. By treating "forgetting" as a physical process of dissipative coupling, we enable artificial intelligence architectures that are both infinitely persistent and efficiently updatable. This alignment suggests that the path to robust machine reasoning lies in the active management of the system's own entropy production.

---
**References**  

[1]  Prigogine, I. (1955). *Introduction to Thermodynamics of Irreversible Processes*. Thomas.  
[2]  Landauer, R. (1961). *Irreversibility and Heat Generation in the Computing Process*. IBM Journal of Research and Development.  
[3]  Greydanus, S., et al. (2019). *Hamiltonian Neural Networks*. NeurIPS.  
[4]  Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.  
[5]  Ottinger, H. C. (2005). *Beyond Equilibrium Thermodynamics*. Wiley-Interscience.  
[6]  Cranmer, M., et al. (2020). *Lagrangian Neural Networks*. ICLR.  
[7]  Schlögl, F. (1971). *Thermodynamic stability of non-equilibrium states*. Zeitschrift für Physik.  
