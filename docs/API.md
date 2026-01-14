# API Reference

> Complete API documentation for GFN modules.

---

## Models

### `GFN`
Main Geodesic Flow Network model.

```python
from src import GFN

model = GFN(
    vocab_size: int,           # Vocabulary size
    dim: int = 256,            # Hidden dimension
    depth: int = 4,            # Number of G-Layers
    rank: int = 32,            # Christoffel low-rank approximation
    integrator_type: str = 'heun'  # 'heun', 'rk4', 'symplectic', 'leapfrog'
)

# Forward pass
logits, (x_final, v_final) = model(
    input_ids,                 # [batch, seq_len]
    attention_mask=None,       # Optional [batch, seq_len]
    state=None                 # Optional (x, v) tuple for continuation
)
```

---

## Loss Functions

### `GFNLoss`
Combined loss with Hamiltonian regularization.

```python
from src import GFNLoss

criterion = GFNLoss(
    lambda_h: float = 0.01,    # Hamiltonian loss weight
    lambda_g: float = 0.001,   # Geodesic regularization weight
    ignore_index: int = -100   # Padding token index
)

loss, metrics = criterion(
    logits,                    # [batch, seq, vocab]
    targets,                   # [batch, seq]
    velocities=None,           # Optional list of velocity tensors
    christoffel_outputs=None   # Optional list of curvature tensors
)
```

### `hamiltonian_loss`
Standalone Hamiltonian energy conservation loss.

```python
from src import hamiltonian_loss

loss = hamiltonian_loss(
    velocities: list,          # List of [batch, dim] tensors
    lambda_h: float = 0.01
)
```

---

## Optimizers

### `RiemannianAdam`
Adam with manifold retraction.

```python
from src import RiemannianAdam

optimizer = RiemannianAdam(
    params,                    # Model parameters
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    retraction: str = 'normalize',  # 'euclidean', 'normalize', 'cayley'
    max_norm: float = 10.0
)
```

### `ManifoldSGD`
Simple SGD with retraction.

```python
from src import ManifoldSGD

optimizer = ManifoldSGD(
    params,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    max_norm: float = 10.0
)
```

---

## Integrators

All integrators share the same interface:

```python
integrator = Integrator(christoffel_net, dt=0.1)
x_new, v_new = integrator(x, v, force)
```

### `HeunIntegrator`
2nd order Runge-Kutta (fast, slight drift).

### `RK4Integrator`
4th order Runge-Kutta (accurate, slow).

### `SymplecticIntegrator`
Velocity Verlet (energy-preserving).

### `LeapfrogIntegrator`
Störmer-Verlet (best symplectic, recommended).

---

## Geometry

### `LowRankChristoffel`
Low-rank approximation of Christoffel symbols.

```python
from src import LowRankChristoffel

christoffel = LowRankChristoffel(
    dim: int,                  # Hidden dimension
    rank: int = 16             # Low-rank approximation
)

# Compute Γ(v, v)
gamma = christoffel(v)         # [batch, dim]
```

---

## Layers

### `GLayer`
Single geodesic evolution layer.

```python
from src import GLayer

layer = GLayer(
    dim: int,
    rank: int = 16,
    base_dt: float = 0.1,
    integrator_type: str = 'heun'
)

x_new, v_new = layer(x, v, force)
```

### `RiemannianGating`
Curvature-based flow control.

```python
from src import RiemannianGating

gating = RiemannianGating(dim)
gate = gating(x)               # [batch, 1] in range [0, 1]
```

---

## Constants

### `INTEGRATORS`
Registry of available integrators.

```python
from src import INTEGRATORS

print(INTEGRATORS.keys())
# ['heun', 'rk4', 'symplectic', 'leapfrog']
```
