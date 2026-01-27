# MANIFOLD API Reference

**Version:** 2.6.2 "Symplectic Forgetting"
**Last Updated:** January 27, 2026

Complete Python API documentation for the MANIFOLD framework, a Geodesic Flow Network (GFN) implementation for sequence modeling with O(1) memory complexity and infinite-horizon generalization capabilities.



## 1. Installation

### 1.1 Prerequisites

```bash
# System requirements
Python 3.10+
PyTorch 2.3+
CUDA 11.8+ (optional, for GPU acceleration)
```

### 1.2 Installation from Source

```bash
git clone https://github.com/Manifold-Laboratory/manifold
cd manifold
pip install -e .
```

### 1.3 Optional CUDA Acceleration

For production deployments requiring maximum throughput:

```bash
# For CUDA 12.9
./compile_cuda_12.9.bat

# For CUDA 11.8
./compile_cuda_11.8.bat
```

CUDA kernels provide 10-50× speedup for large-scale training. See `gfn/cuda/README.md` for detailed compilation instructions.



## 2. Core Model: Manifold

### 2.1 Constructor Signature

```python
from gfn.model import Manifold

model = Manifold(
    vocab_size: int = 50257,
    dim: int = 256,
    depth: int = 4,
    heads: int = 4,
    rank: int = 32,
    integrator_type: str = 'leapfrog',
    use_scan: bool = False,
    physics_config: Optional[Dict] = None,
    impulse_scale: float = 1.0,
    holographic: bool = False
)
```

### 2.2 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vocab_size` | `int` | 50257 | Size of token vocabulary |
| `dim` | `int` | 256 | Latent manifold dimension (hidden size) |
| `depth` | `int` | 4 | Number of stacked M-Layers |
| `heads` | `int` | 4 | Number of parallel geodesic flow heads |
| `rank` | `int` | 32 | Low-rank Christoffel approximation rank |
| `integrator_type` | `str` | `'leapfrog'` | Numerical integration scheme |
| `use_scan` | `bool` | `False` | Enable parallel scan for O(log N) training |
| `physics_config` | `Optional[Dict]` | `None` | Physics engine configuration dictionary |
| `impulse_scale` | `float` | `1.0` | Scaling factor for token embedding forces |
| `holographic` | `bool` | `False` | Enable holographic readout mode |

### 2.3 Integrator Types

The `integrator_type` parameter selects the numerical integration scheme for geodesic dynamics:

- **`'leapfrog'`** (default): Symplectic Velocity Verlet integrator. Recommended for production use due to excellent balance of speed and stability. O(dt³) local error with volume preservation.

- **`'heun'`**: Second-order Runge-Kutta method. Good balance between accuracy and computational cost. Suitable for debugging and initial model development.

- **`'rk4'`**: Fourth-order Runge-Kutta method. Highest accuracy but computationally expensive. Note: May diverge on tasks with non-smooth dynamics (e.g., parity with singularities).

- **`'forest_ruth'`**: Fourth-order symplectic integrator. Excellent energy conservation for high-precision reasoning tasks. Slower but most stable for complex logical operations.

- **`'yoshida'`**: Fourth-order symplectic integrator. Alternative high-order scheme with different coefficients.

- **`'omelyan'`**: Optimized fourth-order symplectic integrator. Good compromise between Forest-Ruth and Yoshida.

- **`'verlet'`**: Basic Velocity Verlet without symplectic optimization. Simple and fast.

### 2.4 Forward Pass

#### Standard Forward Pass

```python
# Input: token IDs [batch_size, seq_len]
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Forward pass
logits, state, christoffels = model(input_ids)

# Returns:
#   logits:      [batch_size, seq_len, vocab_size] - prediction logits
#   state:       Tuple (x_final, v_final) - final position and velocity
#   christoffels: List of curvature tensors per layer (for analysis)
```

#### Autoregressive Forward with State

```python
# For autoregressive generation, maintain state across tokens
state = None  # Initial state

for t in range(sequence_length):
    logits, state, christoffels = model(input_ids[:, t:t+1], state=state)
    # state is passed to next step, enabling O(1) memory inference

# Returns:
#   logits:      [batch_size, 1, vocab_size] - single token predictions
#   state:       Tuple (x, v) - updated dynamical state
#   christoffels: Curvature for current step
```

### 2.5 Generation

```python
model.eval()

prompt = torch.tensor([[1, 2, 3, 4, 5]])  # Token IDs
generated = model.generate(
    prompt_ids=prompt,
    max_new_tokens=100,
    temperature=1.0,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    do_sample: bool = True
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_ids` | `torch.Tensor` | Required | Input token IDs [batch_size, prompt_len] |
| `max_new_tokens` | `int` | Required | Number of tokens to generate |
| `temperature` | `float` | 1.0 | Softmax temperature (lower = more deterministic) |
| `top_k` | `Optional[int]` | 40 | Top-k sampling filter |
| `top_p` | `Optional[float]` | 0.9 | Nucleus sampling threshold |
| `do_sample` | `bool` | `True` | Use sampling vs greedy decoding |



## 3. Physics Configuration

### 3.1 Complete Reference Configuration

The following configuration reflects the optimal settings validated in the superiority benchmark (`tests/benchmarks/viz/vis_gfn_superiority.py`):

```python
physics_config = {
    'embedding': {
        'type': 'functional',     # Neural field embedding (O(1) vocabulary scaling)
        'mode': 'linear',          # 'linear' or 'binary' - linear is superior
        'coord_dim': 16            # Coordinate dimension for neural field
    },
    'readout': {
        'type': 'implicit',        # Implicit neural field readout
        'coord_dim': 16            # Coordinate dimension for inverse mapping
    },
    'active_inference': {
        'enabled': True,           # Enable active inference dynamics
        'dynamic_time': {
            'enabled': True        # Adaptive timestep based on uncertainty
        },
        'reactive_curvature': {
            'enabled': True,       # Curvature modulation by kinetic energy
            'plasticity': 0.2      # Plasticity coefficient α for λ(K) = α·tanh(K)
        },
        'singularities': {
            'enabled': True,       # Enable logical singularities
            'strength': 20.0,      # Singularity attraction strength
            'threshold': 0.8       # Activation threshold for singularity detection
        }
    },
    'fractal': {
        'enabled': True,           # Enable fractal manifold resolution
        'threshold': 0.5,          # Curvature threshold for sub-manifold opening
        'alpha': 0.2               # Recursive resolution parameter
    },
    'topology': {
        'type': 'torus'            # Toroidal topology for cyclic logic
    },
    'stability': {
        'base_dt': 0.4,            # Base integration timestep
        'curvature_clamp': 5.0     # Maximum curvature magnitude
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
).to(device)
```

### 3.2 Embedding Configuration

#### Functional Embedding (Recommended)

The functional embedding uses a neural field (SIREN MLP) to map token coordinates to embedding vectors, achieving O(1) memory scaling with vocabulary size:

```python
embedding_config = {
    'type': 'functional',
    'mode': 'linear',      # Superior performance vs 'binary'
    'coord_dim': 16
}
```

**Key Advantages**:

- O(1) parameters regardless of vocabulary size
- Smooth interpolation between token representations
- Enables generalization to unseen tokens

#### Embedding Modes Comparison

| Mode | Description | Use Case |
|------|-------------|----------|
| `'linear'` | Linear coordinate mapping with smooth interpolation | **Recommended** - superior generalization |
| `'binary'` | Binary coordinate representation | Legacy mode, inferior generalization |

### 3.3 Readout Configuration

#### Implicit Readout (Recommended)

The implicit readout uses an inverse neural field to map manifold coordinates directly to token logits:

```python
readout_config = {
    'type': 'implicit',
    'coord_dim': 16
}
```

**Advantages**:

- Holographic alignment: latent state IS the answer
- No additional classifier weights needed
- Perfect coordinate-to-token mapping

### 3.4 Active Inference Configuration

Active inference enables the manifold to adapt its dynamics based on uncertainty estimates:

```python
active_inference_config = {
    'enabled': True,
    'dynamic_time': {
        'enabled': True
    },
    'reactive_curvature': {
        'enabled': True,
        'plasticity': 0.2
    },
    'singularities': {
        'enabled': True,
        'strength': 20.0,
        'threshold': 0.8
    }
}
```

#### Reactive Curvature

The plasticity scalar modulates curvature based on kinetic energy:

$$\lambda(K) = \alpha \cdot \tanh(K)$$

$$\Gamma_{\text{eff}} = \Gamma_{\text{base}} \cdot (1 + \lambda(K))$$

**Interpretation**: High confusion (elevated velocity) increases curvature, creating a "viscous" manifold that slows processing for integration.

#### Logical Singularities

Singularities represent discrete logical decisions as topological attractors:

$$x_{\text{macro}} \xrightarrow{\mathcal{R} > \tau} x_{\text{micro}}$$

### 3.5 Fractal Configuration

Fractal manifold resolution enables adaptive precision for high-curvature regions:

```python
fractal_config = {
    'enabled': True,
    'threshold': 0.5,
    'alpha': 0.2
}
```

When local curvature $\mathcal{R}$ exceeds the threshold, the manifold recursively opens sub-manifolds with finer temporal resolution ($dt' \ll dt$).

### 3.6 Stability Configuration

```python
stability_config = {
    'base_dt': 0.4,        # Integration timestep (default: 0.3-0.5)
    'curvature_clamp': 5.0  # Maximum |Γ| for numerical stability
}
```

**Timestep Guidelines**:

| dt Range | Behavior |
|----------|----------|
| 0.1-0.2 | Over-cautious, slow learning |
| 0.3-0.4 | **Optimal** - balanced stability and speed |
| 0.5-0.6 | Borderline unstable |
| >0.7 | Complete divergence |



## 4. Geometry Module

### 4.1 LowRankChristoffel

Computes curvature (Christoffel symbols) with low-rank approximation for memory efficiency:

```python
from gfn.geometry import LowRankChristoffel

christoffel = LowRankChristoffel(
    dim=512,
    rank=32,
    physics_config=None
)

# Compute curvature: gamma = Γ(v, x)
# Input:  v [batch, dim] - velocity
#         x [batch, dim] - position
# Output: gamma [batch, dim] - Christoffel symbols
gamma = christoffel(v, x)
```

**Key Features**:

- **Adaptive Gating**: Learnable gate controls when curvature is applied
- **Dynamic Modulation**: Position-dependent scaling of curvature
- **Saturation**: Built-in clamping for numerical stability

### 4.2 ToroidalChristoffel

Toroidal-specific Christoffel symbols for cyclic logic tasks:

```python
from gfn.geometry import ToroidalChristoffel

christoffel = ToroidalChristoffel(
    dim=512,
    rank=32,
    major_radius=1.0,
    minor_radius=0.5,
    physics_config=None
)
```

**Advantages for Parity Tasks**:

- Natural encoding of modular arithmetic (mod 2 → mod $2\pi$)
- Phase space topology matches cyclic logical structure

### 4.3 ReactiveChristoffel

Curvature modulated by kinetic energy for active inference:

```python
from gfn.geometry import ReactiveChristoffel

christoffel = ReactiveChristoffel(
    dim=512,
    rank=32,
    plasticity=0.2
)
```

### 4.4 HyperChristoffel

Hyperbolic Christoffel symbols for hierarchical representations:

```python
from gfn.geometry import HyperChristoffel

christoffel = HyperChristoffel(
    dim=512,
    rank=32,
    curvature_scale=1.0
)
```



## 5. Symplectic Integrators

### 5.1 Leapfrog Integrator (Default)

```python
from gfn.integrators import LeapfrogIntegrator

integrator = LeapfrogIntegrator(
    christoffel=christoffel,
    dt=0.4
)

# One integration step
x_next, v_next = integrator(x, v, force=F, dt_scale=1.0)
```

**Algorithm** (Velocity Verlet with friction):

```python
# Friction term: μ(x, u) · v
# where μ = sigmoid(gate_activ) · 5.0

a_t = F - Γ(v_t, x_t) - friction(v_t, x_t)
v_half = v_t + 0.5 * dt * a_t
x_{t+1} = x_t + dt * v_half
a_{t+1} = F - Γ(v_half, x_{t+1}) - friction(v_half, x_{t+1})
v_{t+1} = v_half + 0.5 * dt * a_{t+1}
v_{t+1} = v_{t+1} / (||v_{t+1}|| + ε)  # Velocity normalization
```

### 5.2 Forest-Ruth Integrator (High Precision)

```python
from gfn.integrators import ForestRuthIntegrator

integrator = ForestRuthIntegrator(
    christoffel=christoffel,
    dt=0.4
)
```

Fourth-order symplectic integration with superior energy conservation for complex reasoning tasks.

### 5.3 Available Integrators

| Integrator | Order | Symplectic | Use Case |
|------------|-------|------------|----------|
| `LeapfrogIntegrator` | 2nd | Yes | **Default** - general purpose |
| `ForestRuthIntegrator` | 4th | Yes | High-precision reasoning |
| `YoshidaIntegrator` | 4th | Yes | Alternative high-precision |
| `OmelyanIntegrator` | 4th | Yes | Optimized 4th-order |
| `VerletIntegrator` | 2nd | Yes | Simple symplectic |
| `HeunIntegrator` | 2nd | No | Debugging, initial training |
| `RK4Integrator` | 4th | No | High accuracy (may diverge) |
| `EulerIntegrator` | 1st | No | Baselines only |



## 6. Layers

### 6.1 MLayer

Core Manifold layer replacing Transformer attention:

```python
from gfn.layers import MLayer

layer = MLayer(
    dim=512,
    heads=8,
    rank=32,
    integrator_type='leapfrog',
    physics_config=None
)

# Forward pass
x_next, v_next, context, christoffels = layer(x, v, force, context)
```

**Multi-Head Processing**:

- Splits (x, v) into `heads` independent geodesic flows
- Each head has independent Christoffel symbols and integrator
- Output heads mixed via learned projection

### 6.2 ParallelMLayer

Parallel scan variant for O(log N) training:

```python
from gfn.layers import ParallelMLayer

layer = ParallelMLayer(
    dim=512,
    heads=8,
    rank=32
)

# Process entire sequence in parallel
x_out, v_out, ctx, christoffels = layer(
    None, None, force=force_sequence
)
```

### 6.3 Gating Layer

Thermodynamic gating ("The Clutch") for switching between memory and computation:

```python
from gfn.layers import GatingLayer

gating = GatingLayer(
    dim=512,
    gate_type='thermodynamic'
)

# Learnable friction coefficient
mu = gating(x, v, temperature=1.0)
# mu → 0: Superfluid (memory preservation)
# mu → 1: Dissipative (computation/update)
```



## 7. Loss Functions

### 7.1 ToroidalDistanceLoss

Specialized loss for toroidal manifolds:

```python
from gfn.losses import ToroidalDistanceLoss

criterion = ToroidalDistanceLoss()

# Compute loss on circular coordinates
loss = criterion(predictions, targets)
# Handles π/2 offset for binary classification
```

### 7.2 Geodesic Regularization

Encourages particles to follow geodesic paths:

```python
from gfn.losses import geodesic_regularization

loss_geo = geodesic_regularization(
    christoffels,
    lambda_g=0.001
)
```

### 7.3 Hamiltonian Loss

Energy conservation regularization:

```python
from gfn.losses import hamiltonian_loss

loss_ham = hamiltonian_loss(
    v_sequence,
    states=x_sequence,
    metric_fn=metric_fn,
    lambda_h=0.0,
    forces=forces
)
```



## 8. Optimization

### 8.1 RiemannianAdam (Required)

Standard Adam causes Euclidean drift on manifolds. Use RiemannianAdam with retraction:

```python
from gfn.optim import RiemannianAdam

optimizer = RiemannianAdam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    retraction='normalize',  # 'euclidean', 'normalize', 'cayley'
    max_norm=10.0
)

# Standard PyTorch usage
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Retraction Types**:

| Type | Description |
|------|-------------|
| `'normalize'` | Clip to max_norm - **recommended** |
| `'cayley'` | Orthogonal projection (square matrices only) |
| `'euclidean'` | No retraction (unstable) |

### 8.2 Recommended Training Configuration

```python
# Optimizer groups (differential learning rates)
optimizer = RiemannianAdam([
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-2, 'weight_decay': 0}
])

# Learning rate schedule
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-3,
    total_steps=max_steps,
    pct_start=0.2
)

# Gradient clipping (critical)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```



## 9. Utilities

### 9.1 Memory Measurement

```python
from tests.benchmarks.bench_utils import PerformanceStats

def forward_fn():
    return model(x)

peak_mb = PerformanceStats.measure_peak_memory(model, forward_fn)
```

### 9.2 Configuration Loading

```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = Manifold(**config['model'])
```

### 9.3 Model Checkpointing

```python
# Save
torch.save({
    'model_state': model.state_dict(),
    'config': model.config,
    'physics_config': model.physics_config
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state'])
```



## 10. Complete Training Example

```python
import torch
import torch.nn as nn
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import ToroidalDistanceLoss, geodesic_regularization, hamiltonian_loss

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimal configuration (from superiority benchmark)
physics_config = {
    'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
    'readout': {'type': 'implicit', 'coord_dim': 16},
    'active_inference': {
        'enabled': True,
        'dynamic_time': {'enabled': True},
        'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
        'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
    },
    'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
    'topology': {'type': 'torus'},
    'stability': {'base_dt': 0.4}
}

# Model
model = Manifold(
    vocab_size=2,
    dim=128,
    depth=6,
    heads=4,
    integrator_type='leapfrog',
    physics_config=physics_config,
    impulse_scale=80.0,
    holographic=True
).to(device)

# Optimizer with differential learning rates
optimizer = RiemannianAdam([
    {'params': [p for n, p in model.named_parameters()
                if not any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-3, 'weight_decay': 1e-4},
    {'params': [p for n, p in model.named_parameters()
                if any(x in n for x in ['x0', 'v0', 'impulse_scale', 'gate'])],
     'lr': 1e-2, 'weight_decay': 0}
])

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-3, total_steps=1000, pct_start=0.2
)

criterion = ToroidalDistanceLoss()

# Training loop
for step in range(1000):
    optimizer.zero_grad()
    
    # Forward
    output = model(inputs, collect_christ=False)
    x_pred = output[0]
    
    # Compute loss
    loss_val = criterion(x_pred, targets.float().unsqueeze(-1).expand_as(x_pred))
    
    # Physics losses (if collecting Christoffel data)
    if isinstance(output, tuple) and len(output) >= 6:
        christoffels = output[2]
        if christoffels:
            loss_phy = geodesic_regularization(None, christoffels, lambda_g=0.001)
            loss_val = loss_val + loss_phy
    
    # Backward
    total_loss = loss_val
    if not torch.isnan(total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    # Logging
    if step % 50 == 0:
        accuracy = compute_accuracy(x_pred, targets)
        print(f"Step {step}: Loss = {loss_val.item():.4f}, Acc = {accuracy:.2%}")
```

### 10.1 Parity Task Example

```python
class ParityTask:
    """Cumulative Parity (Modulo-2) task for state tracking."""
    def __init__(self, vocab_size=2, length=20, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
    
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y_int = torch.cumsum(x, dim=1) % self.mod  # Cumulative parity
        PI = 3.14159265359
        y_angle = (y_int.float() * 2.0 - 1.0) * (PI * 0.5)  # Map to angles
        return x, y_int, y_angle
```



## 11. Troubleshooting

### 11.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Training loss oscillates chaotically | Using standard Adam | Use `RiemannianAdam` |
| Model diverges after ~100 steps | Missing velocity normalization | Enable `velocity_norm` in config |
| Out of memory | Excessive batch size | Reduce batch size or use gradient accumulation |
| Slow training | Sequential Christoffel | Enable CUDA kernels |
| Poor generalization | Binary embedding mode | Use `mode='linear'` |
| Low accuracy | Wrong readout type | Use `type='implicit'` |

### 11.2 Gradient Issues

```python
# If gradients are NaN:
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# And check for NaN in inputs:
assert not torch.isnan(inputs).any()
```

### 11.3 Numerical Instability

```python
# Enable stability checks
model.check_numerical_stability()
```



## 12. API Changelog

### Version 2.6.2 (Current)

- **Fixed**: CUDA kernel saturation terms
- **Added**: Forest-Ruth integrator support
- **Improved**: Active inference documentation
- **Updated**: Optimal configuration to use `linear` embedding mode

### Version 2.6.0

- Added Dynamic Forget Gate (Thermodynamic Friction)
- Updated M-Layer with friction term
- Verified 100K token generalization
- Added Parallel Scan to CUDA kernels

### Version 2.5.0

- Initial production release
- Functional embeddings
- Binary readout mode
- Multi-head architecture



**Documentation Version**: 2.6.2  
**API Stability**: Beta (breaking changes possible)  
**License**: Apache 2.0

For theoretical foundations, see [SCIENTIFIC_PAPER.md](SCIENTIFIC_PAPER.md).
For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).
For benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).
For physics derivations, see [PHYSICS.md](PHYSICS.md).
