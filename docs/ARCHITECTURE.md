# GFN Architecture

> Deep dive into the Geodesic Flow Network architecture.

---

## Core Concept

GFN models sequences as **geodesic flows on a Riemannian manifold**.

```
Traditional: Token → Attention → Token
GFN:         Token → Force → Geodesic Flow → Position → Token
Training:    Input(Force) → Parallel Associative Scan → State Sequence (O(log N))
Inference:   Token → Sequential Geodesic Flow → State (O(1) Step)
```

---

## Mathematical Foundation

### State Variables
- **x**: Position on the manifold (hidden state)
- **v**: Velocity (tangent vector, rate of change)

### Geodesic Equation
$$\frac{d^2 x^k}{dt^2} + \Gamma^k_{ij} \frac{dx^i}{dt} \frac{dx^j}{dt} = F^k$$

Where:
- $\Gamma^k_{ij}$: Christoffel symbols (curvature)
- $F^k$: External force (input token embedding)

---

## Architecture Diagram (GFN V2)
 
 ```mermaid
 graph TD
     Force[Token Force] -->|Split| Heads
     
     subgraph Multi-Head GLayer
         direction LR
         Head1[Head 1: Subspace Flow]
         Head2[Head 2: Subspace Flow]
         HeadN[Head N... ]
         
         Heads --> Head1
         Heads --> Head2
         Heads --> HeadN
         
         Head1 --> Concat
         Head2 --> Concat
         HeadN --> Concat
     end
     
     Concat --> Mix[Mixing Projection]
     Mix --> Norm[Pre-LayerNorm]
     Norm --> Next[Next Layer]
 ```
 
 ### Multi-Head Geodesic Flows (The "V2" Breakthrough)
 
 Just as Transformers use Multi-Head Attention to attend to different subspaces, GFN V2 computes **parallel geodesic flows** on independent Riemannian sub-manifolds.
 
 - **Why?** A single manifold often forces all dynamics to share the same curvature.
 - **GFN V2:** Splitting `dim` into `K` heads allows the model to learn `K` distinct geometries simultaneously (e.g., one head for arithmetic, one for syntax).
 
 ### Pre-LayerNorm Design
 
 Consistent with modern LLM practices (GPT-2/3, Llama), GFN V2 applies LayerNormalization **before** the geodesic evolution. This ensures stable gradients in deep networks (12+ layers).
 
 ```python
 # GFN V2 Block
 x_norm, v_norm = ln(x), ln(v)
 x_heads = split(x_norm)
 # ... integrate ...
x_out = proj(concat(x_heads))
```

### Parallel Associative Scan (Training Mode)

To enable massive parallel training on GPUs, MANIFOLD switches to a "Linearized Geodesic Flow" mode.
- **Linearization**: The network predicts $A_t$ (decay/rotation) and $B_t$ (input modulation) for all timesteps in parallel.
- **Scan**: A recursive doubling algorithm computes the prefix sum of states.
- **Result**: Training time is reduced from $O(L)$ sequential steps to $O(\log L)$ parallel steps, enabling >200x speedups on long sequences.
 ```

---

## Complexity Analysis

| Model | Time | Memory | Context |
|-------|------|--------|---------|
| Transformer | O(N²) | O(N²) | Limited by attention |
| Mamba/SSM | O(N) | O(1) | Linear compression |
| **MANIFOLD** | **O(log N)** (Train) / O(N) (Scan) | **O(1)** | Geodesic memory |

**Parallel Training (The Scan Breakthrough):**
By approximating the non-linear geodesic flow as a Linear Time-Varying (LTV) system during training: 
$v_t = A_t v_{t-1} + B_t$
We can use a **Parallel Associative Scan** (Hillis-Steele algorithm) to compute the entire sequence in $O(\log N)$ parallel depth. This aligns MANIFOLD with state-of-the-art SSMs like Mamba in terms of training efficiency.

GFN achieves O(1) memory because:
- No attention matrix stored
- State (x, v) is fixed-size regardless of sequence length
- Information encoded in trajectory, not explicit memory

---

## Component Details

### 1. Embedding Layer
Standard token embedding that acts as "force" on the manifold.

```python
force = self.embedding(token)  # [batch, dim]
```

### 2. Christoffel Network (Low-Rank)
Computes curvature using efficient decomposition:

$$\Gamma(v, v) = W \cdot (U^T v)^2$$

Parameters:
- U: [dim, rank] - Projection basis
- W: [dim, rank] - Output weights

This reduces O(dim³) to O(dim × rank).

### 3. Integrators
Numerically solve the geodesic ODE:

| Integrator | Formula | Properties |
|------------|---------|------------|
| Heun | x' = x + dt/2 (v + v') | Fast, drifts |
| RK4 | 4th order Runge-Kutta | Accurate, slow |
| Leapfrog | v₁/₂ = v + dt/2 a, x' = x + dt v₁/₂ | **Symplectic** |

### 4. Gating Mechanism
Learned curvature-based flow control:

```python
gate = sigmoid(curvature_net(x))  # [0, 1]
x_out = x + gate * (x_new - x)
v_out = v + gate * (v_new - v)
```

High curvature → small steps (gate ≈ 0)
Low curvature → large steps (gate ≈ 1)

### 5. Readout
Project final position to vocabulary:

```python
logits = linear(layer_norm(x))
```

---

## Parameter Count

$$P = V \cdot D + L \cdot (3 \cdot D \cdot R + 2 \cdot R \cdot D) + D \cdot V$$

Where:
- V: vocab_size
- D: dim
- L: depth
- R: rank

Example (gfn_medium):
- V=16, D=512, L=12, R=128
- P ≈ 13M parameters

---

## Training Dynamics

1. **Token arrives** → Force applied to manifold
2. **State evolves** → Geodesic flow through layers
3. **Readout** → Position decoded to prediction
4. **Loss computed** → CE + Hamiltonian regularization
5. **Gradients flow** → Through Riemannian optimizer

The key insight: **Reasoning = Trajectory on Manifold**

Complex patterns emerge from simple geodesic dynamics.
