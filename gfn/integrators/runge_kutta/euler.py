
"""
Euler Integrator (1st Order).
The simplest explicit integrator. 
Useful as a baseline to demonstrate the instability of low-order non-symplectic methods.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import euler_fused, CUDA_AVAILABLE
    from gfn.cuda.include.boundaries_python import apply_boundary_python
except ImportError:
    CUDA_AVAILABLE = False
    def apply_boundary_python(x, tid): return x

class EulerIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False):
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                if U is not None and W is not None:
                    return euler_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps)
            except Exception:
                pass

        curr_x, curr_v = x, v
        for _ in range(steps):
            dt = self.dt * dt_scale
            
            acc = -self.christoffel(curr_v, curr_x)
            if force is not None:
                acc = acc + force
                
            curr_x = curr_x + dt * curr_v
            curr_v = curr_v + dt * acc
            
            # Apply Boundary (Torus)
            topo_id = 1 if 'Toroidal' in self.christoffel.__class__.__name__ else 0
            curr_x = apply_boundary_python(curr_x, topo_id)
        
        return curr_x, curr_v
