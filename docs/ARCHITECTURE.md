# MANIFOLD Architecture

**Version:** 2.6.2 "Symplectic Forgetting"
**Last Updated:** January 27, 2026

This document presents a system overview and design principles for the MANIFOLD framework, a Geodesic Flow Network (GFN) implementation for sequence modeling with O(1) memory complexity and infinite-horizon generalization capabilities.



## 1. System Overview

### 1.1 Core Principles

MANIFOLD is a sequence modeling architecture based on **Second-Order State Space Models (SSMs)**. It replaces discrete attention mechanisms with a continuous dynamical system that evolves according to geodesic equations on a learnable Riemannian manifold.

**Key Properties**:

- **O(1) Memory**: Constant-size state (x, v) regardless of sequence length
- **Symplectic Dynamics**: Volume-preserving integration improves information flow efficiency
- **Quadratic Interactions**: Learnable "curvature" layer mixes features multiplicatively
- **Compositionality**: Hierarchical stackable architecture

### 1.2 Physical Analogy

In MANIFOLD, information processing is conceptualized as particle motion on a Riemannian manifold:

| Neural Concept | Physical Analogy |
|----------------|------------------|
| Latent state x | Position on the manifold |
| Momentum v | Long-range memory |
| Token embedding F | External force |
| Christoffel Γ | Spacetime curvature |
| Geodesic flow | Reasoning trajectory |

Reasoning is therefore trajectory formation in a geometric space where "thought" is particle dynamics and "inference" is geodesic path computation.



## 2. System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MANIFOLD DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                            Token IDs [B, L]
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Functional Embedding        │
                    │   (SIREN Neural Field)        │
                    │   mode: 'linear' ✓            │
                    │   O(1) vs vocabulary          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                            Force F [B, L, D]
                                    │
                                    ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                    M-LAYER (replaces Attention)                        │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  Pre-LayerNorm                                                   │  │
    │  └─────────────────────────────┬───────────────────────────────────┘  │
    │                                ▼                                        │
    │                    Split into H heads                                  │
    │                                ▼                                        │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │  For each head h:                                               │  │
    │  │    1. Γ_h(v_h, x_h)  [Christoffel + Reactive Curvature]         │  │
    │  │    2. Adaptive Gate  [Thermodynamic Gating]                      │  │
    │  │    3. Integration: (x', v') = LeapFrog(x, v, F-Γ, dt)           │  │
    │  │    4. Velocity normalization v' ← v' / ||v'||                   │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                                ▼                                        │
    │                    Concatenate heads                                    │
    │                                ▼                                        │
    │                    Head Mixing (Linear projection)                     │
    │                                ▼                                        │
    │                    (x_L, v_L)  [Final state]                           │
    └───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Implicit Readout            │
                    │   (Inverse Neural Field)      │
                    │   Holographic Alignment       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                            Logits [B, L, V]
```

### 2.1 Training vs Inference Mode

| Phase | Time Complexity | Space Complexity |
|-------|-----------------|------------------|
| Training (parallel) | O(L) | O(L) for logits |
| Inference (autoregressive) | O(1) per token | O(1) state |



## 3. Core Components

### 3.1 Embedding Layer

**Purpose**: Convert discrete tokens into continuous force vectors

**FunctionalEmbedding** (recommended):

```python
from gfn.embeddings import FunctionalEmbedding

embedding = FunctionalEmbedding(
    vocab_size=50257,
    emb_dim=512,
    coord_dim=16,
    mode='linear'  # 'linear' is superior to 'binary'
)

# Forward: IDs → Coordinates → SIREN → Embedding
emb = embedding(input_ids)  # [batch, seq_len, emb_dim]
```

**Embedding Types**:

| Type | Description | Parameters | Use Case |
|------|-------------|------------|----------|
| `functional` | Neural field (SIREN) | O(1) vs vocabulary | **Recommended** - generalization |
| `implicit` | Learnable coordinate table | O(vocab · coord_dim) | Moderate |
| `standard` | Classic lookup table | O(vocab · emb_dim) | Baselines |

**Linear vs Binary Mode**:

The `linear` mode provides smooth interpolation between token representations, while `binary` uses binary coordinates. Benchmarks demonstrate that `linear` is superior for out-of-distribution generalization.

**Innovation**: Zero-force bias (E(0) = 0) enables inertial coasting, where the system maintains momentum without continuous input.

### 3.2 M-Layer (Manifold Layer)

**Replaces**: Transformer attention block

**Detailed Structure**:

```
Input: (x, v, F, context)
    │
    ▼
Pre-LayerNorm (x only)
    │
    ▼
Split into H parallel heads
    │
    ▼
For each head h:
    │
    ├───► Christoffel: Γ_h(v_h, x_h)
    │         │
    │         ├─── Base Christoffel (LowRank)
    │         ├─── Reactive Curvature: Γ · (1 + λ(K))
    │         └─── Singularities: logical attraction
    │
    ├───► Adaptive Gate: gate(x) ⊙ Γ
    │         │
    │         └─── Thermodynamic Gating: μ = sigmoid(gate) · 5.0
    │
    ├───► Force: F_h - Γ_h - Friction
    │
    ├───► Symplectic Integration: (x', v') = LeapFrog(x, v, F-Γ, dt)
    │         │
    │         └─── Velocity Verlet with normalization
    │
    └───► Velocity normalization: v' ← v' / (||v'|| + ε)
    │
    ▼
Concatenate all head outputs
    │
    ▼
Head Mixing: [x₁, x₂, ..., x_H] → Linear → x_out
                    [v₁, v₂, ..., v_H] → Linear → v_out
    │
    ▼
Output: (x_next, v_next, context_next)
```

**State Parameters**:

| Symbol | Name | Function |
|--------|------|----------|
| x | Position | Semantic location in manifold |
| v | Velocity | Semantic momentum (memory) |
| F | Force | Input token embedding |
| context | Context | Inter-layer communication |

### 3.3 Quadratic Interaction Layer (Christoffel)

**Mathematical Formulation**:

Instead of deriving Γ from a metric tensor g (which requires solving partial differential equations), we parameterize Γ directly as a low-rank quadratic operator:

$$\Gamma(v, x) \approx W \cdot \left[(U^T v)^2 \odot \sigma\left(\|U^T v\|\right)\right]$$

**Components**:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| $U \in \mathbb{R}^{d \times r}$ | Basis matrix | Velocity projection to rank space |
| $W \in \mathbb{R}^{d \times r}$ | Weight matrix | Curvature composition |
| $\sigma(\cdot)$ | Soft saturation | Numerical stability |
| $r$ (rank) | Integer 16-64 | Low-rank compression |

**Adaptive Features**:

```python
# Adaptive gating
gate = torch.sigmoid(W_gate · x)
Γ_gated = gate ⊙ Γ

# Position-dependent scaling
mod = 1 + σ(V^T x)
Γ_modulated = mod · Γ
```

**Initialization**: Near-zero (flat manifold) for stable training start.

### 3.4 Symplectic Integrator

**Algorithm**: Velocity Verlet (Leapfrog) with thermodynamics

```python
def leapfrog_step(x, v, F, christoffel, dt, gate_activ):
    # Learned friction coefficient (The Clutch)
    mu = torch.sigmoid(gate_activ) * 5.0  # Friction coefficient
    
    # Acceleration at t
    gamma = christoffel(v, x)
    friction = mu * v
    a_t = F - gamma - friction
    
    # Half-step velocity
    v_half = v + 0.5 * dt * a_t
    
    # Full-step position
    x_next = x + dt * v_half
    
    # Acceleration at t+1
    gamma_next = christoffel(v_half, x_next)
    friction_next = mu * v_half
    a_next = F - gamma_next - friction_next
    
    # Half-step velocity finalization
    v_next = v_half + 0.5 * dt * a_next
    
    # Velocity normalization (critical for stability)
    v_next = v_next / (torch.norm(v_next, dim=-1, keepdim=True) + 1e-6)
    
    return x_next, v_next
```

**Integrator Properties**:

| Property | Description |
|----------|-------------|
| Time-reversible | Symmetric under t → -t |
| Volume-preserving | det(∂(x', v')/∂(x, v)) = 1 |
| Local error | O(dt³) |
| Global error | O(dt²) |

**Alternative Integrators**:

| Integrator | Order | Symplectic | Use Case |
|------------|-------|------------|----------|
| Leapfrog | 2nd | Yes | **Recommended** - speed |
| Forest-Ruth | 4th | Yes | Maximum precision |
| Heun | 2nd | No | Balance |
| RK4 | 4th | No | High accuracy (may diverge) |

### 3.5 Multi-Head Architecture

**Parallelism**: H independent geodesic flows

For each head:
- Independent Christoffel symbols Γ_h
- Independent integration parameters dt_h
- Independent gating networks

**Head Mixing**:

```python
x_out = Linear([x₁, x₂, ..., x_H])
v_out = Linear([v₁, v₂, ..., v_H])
```

**Interpretation**: Different heads explore different geometric "hypotheses," and the learned mixing combines their perspectives.

### 3.6 Readout Layer

**Implicit Mode** (recommended):

```python
# Inverse coordinate-to-token mapping
logits = SIREN_inverse(x_final)  # [batch, seq, vocab_size]
```

**Standard Mode** (for language):

```python
logits = Linear(x_final)  # [batch, seq, vocab_size]
```

**Binary Mode** (for binary tasks):

```python
logits = MLP(x_final)  # → [batch, seq, coord_dim]
# Decoding via threshold or nearest-neighbor
```

**Holographic Alignment**: The latent state IS the answer. If the target is "1," the particle must physically be at θ = π.



## 4. Active Inference and Dynamics

### 4.1 Reactive Curvature

In standard physics, spacetime tells matter how to move, and matter tells spacetime how to curve. In Hyper-Torus, "matter" is the **Kinetic Energy of Thought**:

$$K = \frac{1}{2} \|v\|^2$$

**Plasticity Scalar**:

$$\lambda(K) = \alpha \cdot \tanh(K)$$

**Effective Connection**:

$$\Gamma_{\text{eff}} = \Gamma_{\text{base}} \cdot \left(1 + \lambda(K)\right)$$

**Interpretation**: When the model is "confused" (high oscillation/velocity), the space becomes viscous and highly curved, acting as an automatic braking system.

### 4.2 Logical Singularities

Singularities represent discrete logical decisions as topological attractors:

$$x_{\text{macro}} \xrightarrow{\mathcal{R} > \tau} x_{\text{micro}}$$

**Configuration**:

```python
singularities = {
    'enabled': True,
    'strength': 20.0,    # Attraction strength
    'threshold': 0.8     # Activation threshold
}
```

### 4.3 Fractal Portals

To address finite precision limitations of continuous integration, we implement **Fractal Manifolds**:

```python
fractal = {
    'enabled': True,
    'threshold': 0.5,    # Critical curvature
    'alpha': 0.2         # Recursive resolution parameter
}
```

When local curvature ℛ exceeds the threshold, the manifold recursively opens sub-manifolds with finer timestep (dt' << dt).



## 5. Thermodynamic Gating (The Clutch)

### 5.1 Stability-Plasticity Dilemma

A purely Hamiltonian system conserves energy indefinitely. This enables infinite memory but cannot "forget" or "update" cleanly (it would oscillate around the new target).

### 5.2 Solution: Variable Friction

We introduce a variable friction coefficient μ(x, u):

$$\frac{dp}{dt} = F_{\text{conservative}} - \mu(x, u) \cdot p$$

### 5.3 Thermodynamic Phases

| Phase | μ | Behavior |
|-------|---|----------|
| Superfluid | μ ≈ 0 | Information as persistent current (Memory) |
| Dissipative | μ >> 0 | Information overwritten; energy released as heat (Computation) |

**Implementation**:

```python
mu = torch.sigmoid(gate_activ) * 5.0  # Friction coefficient
friction = mu * v  # Friction force
```



## 6. Data Flow

### 6.1 Training (Parallel, O(N) memory)

```
Tokens [B, L]
    │
    ▼
Embeddings [B, L, D]
    │
    ▼
Process all L tokens → Logits [B, L, V]
    │
    ▼
Loss = CrossEntropy(Logits, Targets)
```

**Memory**: O(B·L·V) for output logits (unavoidable).

### 6.2 Inference (Recurrent, O(1) memory)

```
Token_t [B, 1]
    │
    ▼
Embedding_t [B, D]
    │
    ▼
State (x, v) [B, 2D] ← From previous step
    │
    ▼
M-Layers: (x', v') = f(x, v, Embedding_t)
    │
    ▼
Logit_t [B, V]
    │
    ▼
Sample next token
    │
    ▼
Repeat with State = (x', v')
```

**Memory**: O(B·D) (constant).



## 7. Optimization Strategy

### 7.1 Riemannian Manifold Constraints

**Problem**: Learned geometries define implicit constraints (e.g., bounded curvature).

**Solution**: RiemannianAdam with retraction:

```python
W_new = Retract_M(W_old - η · grad)
```

**Retraction Types**:

| Type | Description |
|------|-------------|
| `normalize` | W / max(1, ||W||/max_norm) - **recommended** |
| `cayley` | Orthogonal projection (square matrices only) |
| `euclidean` | No retraction (unstable) |

### 7.2 Stability Mechanisms

**Velocity Normalization** (critical):

```python
v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
```

Prevents explosion while preserving direction.

**Curvature Clamping**:

```python
Gamma = torch.clamp(Gamma, -5.0, 5.0)
```

Prevents numerical singularities.

**Gradient Clipping**:

```python
torch.nn.utils.clip_grad_norm_(params, 0.05)
```

Stricter than standard (0.1-1.0).



## 8. Computational Complexity

| **Operation** | **Training** | **Inference** |
|---------------|--------------|---------------|
| Embedding | O(L·d²) | O(d²) |
| Christoffel | O(L·d²·r) | O(d²·r) |
| Integration | O(L·d) | O(d) |
| Readout | O(L·d·V) | O(d·V) |
| **Total** | **O(L·d²·r)** | **O(d²·r)** |

### 8.1 Comparison with Other Architectures

| Architecture | Memory | Compute | Gradient Stability | Inductive Bias |
|--------------|--------|---------|--------------------|----------------|
| Transformer | O(N) | O(N²) | Good | Permutation invariance |
| LSTM/GRU | O(1) | O(N) | Poor | Sequential |
| Mamba (SSM) | O(1) | O(N) | Medium | State compression |
| **Manifold** | **O(1)** | **O(N·d²)** | **Excellent** | **Geometry** |

**Asymptotic Win**: For L > d (long sequences), Manifold is more efficient.



## 9. Design Principles

### 9.1 Physics-First

All operations grounded in differential geometry and Hamiltonian mechanics:
- Geodesic equation (GR)
- Symplectic integration (classical mechanics)
- Liouville's theorem (statistical mechanics)

### 9.2 Compositionality

Layers stack hierarchically:
- Layer 1: Processes raw tokens
- Layer L: Processes abstract semantic representations
- Each layer refines the manifold geometry

### 9.3 Memory Efficiency

State compression via geometric encoding:
- Traditional RNN: Stores history in fixed vector
- Manifold: Encodes history in curvature + momentum

### 9.4 Gradient Stability

Symplectic structure ensures:
- No vanishing (det(J) = 1)
- No exploding (energy conserved)
- Infinite horizon stability



## 10. Optimal Configuration

The following configuration has been validated in the superiority benchmark (`tests/benchmarks/viz/vis_gfn_superiority.py`):

```python
physics_config = {
    'embedding': {
        'type': 'functional',
        'mode': 'linear',       # Superior to 'binary'
        'coord_dim': 16
    },
    'readout': {
        'type': 'implicit',
        'coord_dim': 16
    },
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {
            'enabled': True,
            'plasticity': 0.2
        },
        'singularities': {
            'enabled': True,
            'strength': 20.0,
            'threshold': 0.8
        }
    },
    'fractal': {
        'enabled': True,
        'threshold': 0.5,
        'alpha': 0.2
    },
    'topology': {
        'type': 'torus'
    },
    'stability': {
        'base_dt': 0.4
    }
}

model = Manifold(
    vocab_size=2,
    dim=128,
    depth=6,
    heads=4,
    integrator_type='leapfrog',
    physics_config=physics_config,
    impulse_scale=80.0,
    holographic=True
)
```



## 11. Future Extensions

### 11.1 Mixture of Dynamics (MoM)

Multiple dynamics experts:
- Euclidean (Linear/ResNet behavior)
- Hyperbolic (Tree/Hierarchy specialized)
- Spherical (Cyclic/Rotation specialized)
- Router selects dynamics per token

### 11.2 CUDA Acceleration

Fused kernel implementation for the Leapfrog step:

```c
__global__ void symplectic_step_fused(
    float* x, float* v, float* F,
    float* U, float* W, float dt
);
```

Expected: 10-50× speedup.

### 11.3 Neural ODE Integration

Replace discrete layers with continuous depth solvers:

$$\frac{d(x,v)}{dt} = f_\theta(x, v, F, t)$$

Adaptive computation per token.



**Document Version**: 2.6.2  
**Last Updated**: January 27, 2026  
**Status**: Production Architecture

For theoretical references, see [SCIENTIFIC_PAPER.md](SCIENTIFIC_PAPER.md).  
For API reference, see [API.md](API.md).  
For benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).  
For mathematical foundations, see [PHYSICS.md](PHYSICS.md).
