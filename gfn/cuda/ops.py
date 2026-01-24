
import torch
import torch.nn as nn
import os
import sys

# LEVEL 28: Integrated CUDA Op Pipeline
# ------------------------------------

CUDA_AVAILABLE = False
gfn_cuda = None

def get_cuda_path():
    return os.path.dirname(os.path.abspath(__file__))

# Specialized attempt to load/import the gfn_cuda module
try:
    import gfn_cuda
    CUDA_AVAILABLE = True
except ImportError:
    # Try importing from current package
    try:
        from . import gfn_cuda
        CUDA_AVAILABLE = True
    except ImportError:
        # Not found, will use Python fallback
        pass

# # # FORCE DISABLE CUDA FOR ARCHITECTURE DEBUGGING # # #
CUDA_AVAILABLE = False 
# # # # # # # # # # # # # # # # # # # # # # # # # # #

def christoffel_fused(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0):
    """
    Modular Riemannian Christoffel Symbol Projection.
    Supports Active Inference Multipliers and Periodic Toroidal Features.
    """
    if CUDA_AVAILABLE and v.is_cuda:
        from .autograd import christoffel_fused_autograd
        return christoffel_fused_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
    
    # Python Fallback (Vectorized)
    # h = U^T v
    h = torch.matmul(v, U) # [B, R]
    energy = torch.sum(h*h, dim=-1, keepdim=True)
    S = 1.0 / (1.0 + torch.sqrt(energy) + 1e-6)
    
    M = 1.0
    if plasticity != 0.0:
        E = torch.sum(v*v, dim=-1, keepdim=True) / v.shape[-1]
        M *= (1.0 + plasticity * torch.tanh(E))
    
    if x is not None and V_w is not None:
        # Periodic Singularity logic in Python fallback
        if topology == 1: pot = torch.sum(torch.sin(x) * V_w, dim=-1, keepdim=True)
        else: pot = torch.sum(x * V_w, dim=-1, keepdim=True)
        gate = torch.sigmoid(pot)
        if (gate > sing_thresh).any(): M *= sing_strength

    gamma = torch.matmul(h*h, W.t()) * S * M
    return 20.0 * torch.tanh(gamma / 20.0)

def reactive_christoffel(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0):
    """
    Top-Level Geometry Dispatcher.
    """
    if CUDA_AVAILABLE and v.is_cuda and v.dim() == 2:
        from .autograd import reactive_christoffel_autograd
        res = reactive_christoffel_autograd(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)
        if res is not None: return res
    
    return christoffel_fused(v, U, W, x, V_w, plasticity, sing_thresh, sing_strength, topology)

def leapfrog_fused(x, v, f, U, W, dt, dt_scale, steps, topology=0):
    """
    Fused Symplectic Integrator Step.
    """
    if CUDA_AVAILABLE and x.is_cuda and x.dim() == 2:
        from .autograd import leapfrog_fused_autograd
        res = leapfrog_fused_autograd(x, v, f, U, W, dt, dt_scale, steps, topology)
        if res is not None: return res
    
    # Python Fallback used for Torus (Stable Path)
    # v_half = v + 0.5 * dt * (f - christoffel(v))
    gamma = christoffel_fused(v, U, W, x, None, 0.0, 1.0, 1.0, topology)
    v_new = v + dt * (f - gamma)
    x_new = x + dt * v_new
    if topology == 1: # Toroidal Boundary
         from gfn.geometry.boundaries import apply_boundary_python
         x_new = apply_boundary_python(x_new, 1)
    return x_new, v_new

def recurrent_manifold_fused(x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x, mix_v, Wf, Wi, bf, topology=0):
    """
    Professional Trajectory Fusion for High-Performance Sequence Training.
    """
    if CUDA_AVAILABLE and x.is_cuda: # Enabled for Torus (Fixed kernels)
        from .autograd import recurrent_manifold_fused_autograd
        return recurrent_manifold_fused_autograd(x, v, f, U_stack, W_stack, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x, mix_v, Wf, Wi, bf, topology)
    return None # Use Python Sequence Loop (Autograd managed)
