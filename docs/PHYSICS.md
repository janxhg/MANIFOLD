# GFN Physics Systems

> Complete guide to the physics-informed training systems in GFN.

---

## Overview

GFN uses **four advanced physics systems** to ensure stable, energy-conserving training:

| System | File | Purpose |
|--------|------|---------|
| Hamiltonian Loss | `src/losses.py` | Energy conservation |
| Integrators | `src/geometry.py` | Trajectory stability |
| Riemannian Adam | `src/optim.py` | Manifold-aware updates |
| Adjoint Method | `src/adjoint.py` | O(1) memory backprop |

---

## 1. Hamiltonian Loss (Energy Conservation)

### The Problem
In physics, a perfect geodesic conserves kinetic energy. If velocity explodes or vanishes, the model is behaving non-physically.

### The Solution
We add a regularization term that penalizes energy changes:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_{t} \left| \|v_t\|^2 - \|v_{t-1}\|^2 \right|$$

### Usage

```python
from src import GFNLoss

criterion = GFNLoss(
    lambda_h=0.01,        # Hamiltonian weight (higher = stricter energy)
    ignore_index=-100     # Padding token
)

# Training
logits, (x, v) = model(inputs)
loss, metrics = criterion(logits, targets, velocities=[v])

print(metrics)  # {'ce': 2.3, 'hamiltonian': 0.01, 'total': 2.31}
```

### When to Increase λ_h
- Training explodes with NaN → Increase to 0.05-0.1
- Loss not converging → Decrease to 0.001

---

## 2. Integrators (Trajectory Stability)

GFN evolves hidden states as geodesic flows. The integrator determines **how** we step through time.

### Available Integrators

| Name | Order | Energy Drift | Speed | Best For |
|------|-------|--------------|-------|----------|
| `heun` | 2nd | ⚠️ Accumulates | ⚡ Fast | Quick experiments |
| `rk4` | 4th | ⚠️ Accumulates | 🐢 Slow | Validation |
| `symplectic` | 2nd | ✅ None | ⚡ Fast | Research |
| **`leapfrog`** | 2nd | ✅ **Best** | ⚡ Fast | **Production** |

### What is "Energy Drift"?
Non-symplectic integrators (Heun, RK4) introduce small errors each step. Over thousands of steps, these accumulate and cause:
- Velocity explosion (NaN)
- Velocity collapse (dead gradients)

**Leapfrog** is symplectic: errors cancel out, energy stays constant.

### Usage

```python
from src import GFN, INTEGRATORS

# See all options
print(INTEGRATORS.keys())  # ['heun', 'rk4', 'symplectic', 'leapfrog']

# Create model with Leapfrog (recommended)
model = GFN(
    vocab_size=16,
    dim=512,
    depth=12,
    integrator_type='leapfrog'
)
```

### Leapfrog Algorithm (Kick-Drift-Kick)

```
1. v_{1/2} = v + (dt/2) * acceleration     [Half-Kick]
2. x_{new} = x + dt * v_{1/2}              [Full-Drift]
3. v_{new} = v_{1/2} + (dt/2) * acceleration  [Half-Kick]
```

This is **time-reversible** and **volume-preserving** in phase space.

---

## 3. Riemannian Adam (Manifold-Aware Optimization)

### The Problem
Standard Adam assumes flat Euclidean space:
```
W_new = W_old - lr * grad
```

But GFN weights live on a **curved manifold**. Linear updates can "jump off" the manifold → NaN.

### The Solution
Use **retraction** instead of subtraction:
```
W_new = Retract(W_old, -lr * grad)
```

### Available Retractions

| Type | Description | Best For |
|------|-------------|----------|
| `euclidean` | Standard Adam (fallback) | Debugging |
| `normalize` | Keeps weights bounded | **General use** |
| `cayley` | Preserves orthogonality | Structured matrices |

### Usage

```python
from src import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    retraction='normalize',  # Recommended
    max_norm=10.0            # Clip weights to this norm
)
```

### Why This Prevents NaN
The `normalize` retraction ensures:
```python
if weight.norm() > max_norm:
    weight *= max_norm / weight.norm()
```

Weights can never explode to infinity.

---

## 4. Adjoint State Method (O(1) Memory)

### The Problem
Standard backprop stores all intermediate states:
```
Memory = O(depth × batch × dim)
```

For deep GFN (depth=24+), this exhausts VRAM.

### The Solution
Instead of storing states, **recompute them backward** by solving an adjoint ODE:

```
Forward:  x(0) → x(T)     [Store only final state]
Backward: Solve ODE from T → 0 to recover gradients
```

### Usage

```python
from src.adjoint import AdjointGFN

# Drop-in replacement for GFN with O(1) memory
model = AdjointGFN(
    vocab_size=16,
    dim=512,
    depth=24,
    integration_time=1.0
)

# Requires torchdiffeq
# pip install torchdiffeq
```

### When to Use
- Depth > 20 layers
- VRAM < 8GB
- Training with very long sequences

---

## Complete Training Example

```python
from src import GFN, GFNLoss, RiemannianAdam

# Model with best physics settings
model = GFN(
    vocab_size=16,
    dim=512,
    depth=12,
    integrator_type='leapfrog'
).cuda()

# Physics-informed loss
criterion = GFNLoss(lambda_h=0.01)

# Manifold-aware optimizer
optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-3,
    retraction='normalize'
)

# Training loop
for batch in dataloader:
    inputs = batch.cuda()
    targets = torch.roll(inputs, -1, dims=1)
    
    logits, (x, v) = model(inputs)
    loss, metrics = criterion(logits, targets, velocities=[v])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"CE: {metrics['ce']:.4f}, Hamiltonian: {metrics.get('hamiltonian', 0):.4f}")
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| NaN loss | Energy explosion | Use `leapfrog`, increase `lambda_h` |
| Loss stuck | Vanishing gradients | Decrease `lambda_h`, check `max_norm` |
| OOM | Too deep | Use `AdjointGFN` |
| Slow training | Wrong integrator | Use `heun` for speed, `leapfrog` for stability |
