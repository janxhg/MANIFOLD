"""
CUDA Kernel Correctness Tests
==============================

Verifies that CUDA kernels produce numerically equivalent results to PyTorch.
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.geometry import LowRankChristoffel, LeapfrogIntegrator

def test_christoffel_correctness():
    """Test that CUDA Christoffel matches PyTorch version."""
    print("Testing Christoffel Kernel Correctness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch, dim, rank = 32, 512, 16
    
    # Create test inputs
    v = torch.randn(batch, dim, device=device)
    christoffel = LowRankChristoffel(dim, rank).to(device)
    
    # Compute with kernel (which will use CUDA if available)
    gamma_kernel = christoffel(v)
    
    # Compute reference with pure PyTorch
    proj = torch.matmul(v, christoffel.U)
    sq = proj * proj
    gamma_ref = torch.matmul(sq, christoffel.W.t())
    gamma_ref = torch.clamp(gamma_ref, -5.0, 5.0)
    
    # Compare
    max_error = torch.max(torch.abs(gamma_kernel - gamma_ref)).item()
    mean_error = torch.mean(torch.abs(gamma_kernel - gamma_ref)).item()
    
    print(f"  Max Error: {max_error:.2e}")
    print(f"  Mean Error: {mean_error:.2e}")
    
    assert max_error < 1e-4, f"Christoffel kernel error too large: {max_error}"
    print("  ✓ Christoffel kernel correct!\n")

def test_leapfrog_correctness():
    """Test that CUDA Leapfrog matches PyTorch version."""
    print("Testing Leapfrog Kernel Correctness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch, dim, rank = 32, 512, 16
    dt = 0.1
    
    # Create test inputs
    x = torch.randn(batch, dim, device=device)
    v = torch.randn(batch, dim, device=device)
    force = torch.randn(batch, dim, device=device)
    
    integrator = LeapfrogIntegrator(dim, rank, dt=dt).to(device)
    
    # Test both with and without dt_scale
    for dt_scale in [1.0, 0.5, 0.1]:
        # Compute with kernel
        x_kernel, v_kernel = integrator(x, v, force, dt_scale=dt_scale)
        
        # Compute reference (disable CUDA by patching)
        effective_dt = dt * dt_scale
        gamma_v = integrator.christoffel(v)
        v_half = v + 0.5 * effective_dt * (force - gamma_v)
        x_ref = x + effective_dt * v_half
        gamma_v_half = integrator.christoffel(v_half)
        v_ref = v_half + 0.5 * effective_dt * (force - gamma_v_half)
        
        # Compare
        x_error = torch.max(torch.abs(x_kernel - x_ref)).item()
        v_error = torch.max(torch.abs(v_kernel - v_ref)).item()
        
        print(f"  dt_scale={dt_scale}: X Error={x_error:.2e}, V Error={v_error:.2e}")
        
        assert x_error < 1e-4, f"Leapfrog X error too large: {x_error}"
        assert v_error < 1e-4, f"Leapfrog V error too large: {v_error}"
    
    print("  ✓ Leapfrog kernel correct!\n")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
    else:
        test_christoffel_correctness()
        test_leapfrog_correctness()
        print("All CUDA kernel tests passed! ✓")
