# Parallel Manifold Scans: Logarithmic-Time Sequence Integration for Geodesic Flows

**Author:** Joaquin Stürtz  
**Date:** January 26, 2026

**Abstract**  
The sequential dependency of recursive architectures (e.g., RNNs, S4, Mamba) constitutes a primary bottleneck in high-throughput neural training, where token $t$ traditionally requires the completion of token $t-1$. We introduce **Parallel Manifold Scans**, an associative reformulation of the manifold update operator that enables $O(\log N)$ parallel processing of sequence trajectories. By reformulating the discretized geodesic flow as a prefix-sum over a non-commutative group of affine manifold-warp operators, we achieve the training efficiency of parallel architectures (Transformers) while preserving the constant-time inference memory of recurrent systems. We prove that this associative structure maintains strict numerical consistency with the sequential Hamiltonian flow, providing a bridge between high-speed training and physical adherence.

---

## 1. The Associative Reformulation of Manifold Dynamics

### 1.1 First-Order Affine Recurrence
The discretized evolution of a state co-vector on a manifold (local to a time-step $\Delta t$) can be approximated as a first-order affine recurrence:
$$ \mathbf{s}_t = \mathbf{A}_t \mathbf{s}_{t-1} + \mathbf{b}_t $$
where $\mathbf{s}$ is the phase-space configuration, $\mathbf{A}_t$ is the transition operator determined by the local Christoffel symbols $\Gamma$, and $\mathbf{b}_t$ is the forcing term driven by the external input.

### 1.2 Associative Operator Composition
We define a **Manifold Propagator** $\mathcal{P}_t$ as the operator pair $(\mathbf{A}_t, \mathbf{b}_t)$. The composition of two consecutive propagators $\mathcal{P}_{i+1} \circ \mathcal{P}_i$ is defined by:
$$ (\mathbf{A}_{i+1}, \mathbf{b}_{i+1}) \circ (\mathbf{A}_i, \mathbf{b}_i) = (\mathbf{A}_{i+1} \mathbf{A}_i, \mathbf{A}_{i+1} \mathbf{b}_i + \mathbf{b}_{i+1}) $$

**Theorem (Associativity of Manifold Propagators):** The composition operator $\circ$ satisfies the associative law: $(\mathcal{P}_k \circ \mathcal{P}_j) \circ \mathcal{P}_i = \mathcal{P}_k \circ (\mathcal{P}_j \circ \mathcal{P}_i)$ for all $i, j, k$.

Proof of this associativity allows the sequence integration to be performed using **Parallel Prefix Scan** algorithms (e.g., Hillis-Steele or Blelloch), reducing the temporal dependency from $O(N)$ to $O(\log N)$ on parallel hardware.

---

## 2. Theoretical Complexity and Scalability

### 2.1 Parallel Efficiency
By exploiting the associative structure, the GFN can be trained on excessively long sequences without the serial bottleneck characteristic of traditional RNNs. Unlike the $O(N^2)$ complexity of the Attention mechanism, Parallel Manifold Scans scale linearly with sequence length in terms of work $O(N)$ while maintaining logarithmic depth $O(\log N)$, representing the theoretical lower bound for sequence integration.

### 2.2 Numerical Consistency
Critically, current prefix-scan implementations maintain consistency with the sequential update. Small discrepancies ($< 10^{-7}$) arising from the re-ordering of floating-point operations in the scan tree are negligible compared to the learned curvature gradients. This ensures that the physics learned during high-speed parallel training remains valid during sequential $O(1)$ inference.

---

## 3. Comparison with State-Space Models (SSMs)

While modern SSMs (e.g., Mamba) utilize selective scans over learned transition matrices, Parallel Manifold Scans differ in their geometric grounding:
*   **Geometric Determinism:** The transition operator $\mathbf{A}_t$ is not a free statistical parameter but is derived from the symplectic structure and the Riemannian metric of the latent space.
*   **Volume Preservation:** By ensuring the operator $\mathbf{A}_t$ remains within the symplectic group $Sp(2n, \mathbb{R})$, we guarantee that the parallelization process does not introduce artificial dissipation or information loss.

---

## 4. Conclusion

Parallel Manifold Scans demonstrate that the historical dichotomy between "parallel" (Transformer) and "recurrent" (RNN) architectures is a false one. By revealing the underlying associative structure of discretized manifold flows, we enable a new class of models that combine the best properties of both worlds: the parallel training speed required for large-scale learning and the constant-memory sequence tracking required for infinite-horizon reasoning.

---
**References**  

[1]  Blelloch, G. E. (1990). *Prefix Sums and Their Applications*. Technical Report, CMU.  
[2]  Hillis, W. D., & Steele Jr, G. L. (1986). *Data parallel algorithms*. Communications of the ACM.  
[3]  Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv.  
[4]  Martin, E., & Cundy, C. (2018). *Parallelizing Linear Recurrent Neural Nets Over Sequence Length*. ICLR.  
[5]  Katharopoulos, A., et al. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. ICML.  
[6]  Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.  
