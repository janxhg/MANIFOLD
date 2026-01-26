#!/usr/bin/env python
"""
Quick CUDA Diagnostic Test
===========================
Tests if CUDA kernels are actually being used.
"""

import torch
import sys
from pathlib import Path
import subprocess

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*60)
print("CUDA DIAGNOSTIC TEST")
print("="*60)

# Test 1: CUDA Available?
print("\n[1/5] Checking CUDA availability...")
print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")

# Test 2: Import CUDA ops
print("\n[2/5] Importing CUDA ops...")
try:
    from gfn.cuda.ops import CUDA_AVAILABLE, recurrent_manifold_fused
    print(f"  ✓ gfn.cuda.ops imported successfully")
    print(f"  CUDA_AVAILABLE = {CUDA_AVAILABLE}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

def run_kernel_smoke():
    cmd = [sys.executable, str(PROJECT_ROOT / "tests" / "test_fusion_kernel.py")]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout.strip().splitlines()[-1] if result.stdout else "")
        if result.returncode == 0:
            print("  ✓ Kernel smoke test executed in isolated process")
        else:
            print("  ✗ Kernel smoke test failed")
            print(result.stderr)
    except Exception as e:
        print(f"  ✗ Kernel smoke test error: {e}")

# Test 4: Import Model
print("\n[4/5] Importing Manifold model...")
try:
    from gfn.model import Manifold
    print(f"  ✓ Manifold imported successfully")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

# Test 5: Create small model and forward pass
print("\n[5/5] Testing model forward pass...")
try:
    from gfn.model import Manifold
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Manifold(
        vocab_size=10,
        dim=64,
        depth=2,
        heads=2,
        rank=16,
        integrator_type='heun',
        use_scan=False
    ).to(device)
    
    inputs = torch.randint(0, 10, (2, 5)).to(device)  # batch=2, seq=5
    
    model.eval()  # CRITICAL: Set to eval mode to use non-autograd path
    print(f"  Running forward pass (collect_christ=False to enable CUDA)...")
    with torch.no_grad():
        outputs = model(inputs, collect_christ=False)
        logits = outputs[0]
        state = outputs[1] if len(outputs) > 1 else None
        x_f, v_f = state if state is not None else (None, None)
    
    print(f"  ✓ Forward pass SUCCESS")
    print(f"    Output logits shape: {logits.shape}")
    
    if CUDA_AVAILABLE and torch.cuda.is_available():
        print("\n[EXTRA] Running isolated kernel smoke test...")
        run_kernel_smoke()
    
except Exception as e:
    print(f"  ✗ Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nIf you see CUDA-related prints above (✓ CUDA AVAILABLE, etc.),")
print("then CUDA is working. If not, check the error messages.")
