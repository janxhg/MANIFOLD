# GFN Architecture

> Deep dive into the Geodesic Flow Network architecture.

---

## Core Concept

GFN models sequences as **geodesic flows on a Riemannian manifold**.

```
Traditional: Token → Attention → Token
GFN:         Token → Force → Geodesic Flow → Position → Token
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

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                         GFN                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Token ──► Embedding ──► Force                           │
│                            │                             │
│                            ▼                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │                   G-Layer × N                     │   │
│  │  ┌─────────────────────────────────────────────┐ │   │
│  │  │ Christoffel Network: Γ(v) = W(U^T v)²       │ │   │
│  │  │                      ↓                       │ │   │
│  │  │ Integrator: (x, v) → (x', v')               │ │   │
│  │  │              Leapfrog / Heun / RK4          │ │   │
│  │  │                      ↓                       │ │   │
│  │  │ Gating: x_out = x + g * (x' - x)            │ │   │
│  │  └─────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
│                            │                             │
│                            ▼                             │
│  Position x ──► LayerNorm ──► Linear ──► Logits         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Complexity Analysis

| Model | Time | Memory | Context |
|-------|------|--------|---------|
| Transformer | O(N²) | O(N²) | Limited by attention |
| Mamba/SSM | O(N) | O(1) | Linear compression |
| **GFN** | O(N) | **O(1)** | Geodesic memory |

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
