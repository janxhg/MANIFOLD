#!/usr/bin/env python
"""
Numerical Validation: CUDA vs Python
=====================================
Compares outputs of CUDA kernel vs Python implementation to detect bugs.
"""

import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.cuda import ops

print("="*60)
print("CUDA vs PYTHON NUMERICAL VALIDATION")
print("="*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

if not torch.cuda.is_available():
    print("\n[SKIP] CUDA no disponible, omitiendo validación numérica CUDA vs Python.")
    sys.exit(0)

# Create two identical models
model_cuda = Manifold(
    vocab_size=10,
    dim=64,
    depth=1,
    heads=1,
    rank=16,
    integrator_type='leapfrog',
    use_scan=False
).to(device)

model_python = Manifold(
    vocab_size=10,
    dim=64,
    depth=1,
    heads=1,
    rank=16,
    integrator_type='leapfrog',
    use_scan=False
).to(device)

# Fuerza ruta Python sin desactivar CUDA: usa collect_christ=True para evitar el kernel
force_python_kwargs = {"collect_christ": True}

# Copy weights to ensure identical models
model_python.load_state_dict(model_cuda.state_dict())

# Test input
inputs = torch.randint(0, 10, (2, 5)).to(device)

print("\n[1/3] Running CUDA forward pass...")
model_cuda.eval()
with torch.no_grad():
    outputs_cuda = model_cuda(inputs, collect_christ=False)
    logits_cuda = outputs_cuda[0]
    state_cuda = outputs_cuda[1] if len(outputs_cuda) > 1 else None
    x_cuda, v_cuda = state_cuda if state_cuda is not None else (None, None)

print(f"  CUDA output: logits={logits_cuda.shape}")
print(f"  Sample logit values: {logits_cuda[0, 0, :5]}")

print("\n[2/3] Running Python forward pass...")
model_python.eval()
with torch.no_grad():
    outputs_python = model_python(inputs, **force_python_kwargs)
    logits_python = outputs_python[0]
    state_python = outputs_python[1] if len(outputs_python) > 1 else None
    x_python, v_python = state_python if state_python is not None else (None, None)

print(f"  Python output: logits={logits_python.shape}")
print(f"  Sample logit values: {logits_python[0, 0, :5]}")

print("\n[3/3] Comparing outputs...")
max_diff_logits = (logits_cuda - logits_python).abs().max().item()
max_diff_x = (x_cuda - x_python).abs().max().item()
max_diff_v = (v_cuda - v_python).abs().max().item()

print(f"  Max difference in logits: {max_diff_logits:.6e}")
print(f"  Max difference in x: {max_diff_x:.6e}")
print(f"  Max difference in v: {max_diff_v:.6e}")

# Threshold for numerical differences (should be very small, ~1e-5)
THRESHOLD = 1e-4

print("\n" + "="*60)
if max_diff_logits < THRESHOLD:
    print("✓ VALIDATION PASSED - CUDA matches Python")
    print(f"  (max diff = {max_diff_logits:.6e} < {THRESHOLD:.6e})")
else:
    print("✗ VALIDATION FAILED - CUDA differs from Python!")
    print(f"  (max diff = {max_diff_logits:.6e} >= {THRESHOLD:.6e})")
    print("\n  This indicates a numerical bug in the CUDA kernel.")
    print("  Detailed comparison:")
    print(f"    CUDA logits[0,0,:]: {logits_cuda[0,0,:].cpu().numpy()}")
    print(f"    Python logits[0,0,:]: {logits_python[0,0,:].cpu().numpy()}")
    print(f"    Difference: {(logits_cuda[0,0,:] - logits_python[0,0,:]).cpu().numpy()}")
print("="*60)

