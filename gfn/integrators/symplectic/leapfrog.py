
"""
Leapfrog (Kick-Drift-Kick) Symplectic Integrator.
"""
import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import leapfrog_fused, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

try:
    from gfn.geometry.boundaries import apply_boundary_python
except ImportError:
    def apply_boundary_python(x, tid): return x

class LeapfrogIntegrator(nn.Module):
    def __init__(self, christoffel, dt=0.01):
        super().__init__()
        self.christoffel = christoffel
        self.dt = dt

    def forward(self, x, v, force=None, dt_scale=1.0, steps=1, collect_christ=False, **kwargs):
        if force is None:
            force = torch.zeros_like(x)
            
        # Try Professional Fused CUDA Kernel
        if CUDA_AVAILABLE and x.is_cuda and not collect_christ:
            try:
                # Logic matrices
                U = getattr(self.christoffel, 'U', None)
                W = getattr(self.christoffel, 'W', None)
                
                if U is not None and W is not None:
                    return leapfrog_fused(x, v, force, U, W, self.dt, dt_scale, steps=steps)
            except Exception:
                pass

        curr_x, curr_v = x, v
        # Tell Christoffel to return friction separately for implicit update
        was_separate = getattr(self.christoffel, 'return_friction_separately', False)
        self.christoffel.return_friction_separately = True
        
        try:
            for _ in range(steps):
                effective_dt = self.dt * dt_scale
                h = 0.5 * effective_dt
                
                # 1. Kick (Implicit Friction)
                # v_next = (v_prev + h*(F - gamma)) / (1 + h*mu)
                res = self.christoffel(curr_v, curr_x, force=force, **kwargs)
                if isinstance(res, tuple):
                    gamma, mu = res
                else:
                    gamma, mu = res, 0.0 # Fallback
                    
                v_half = (curr_v + h * (force - gamma)) / (1.0 + h * mu)
                
                # 2. Drift (full step position)
                curr_x = curr_x + effective_dt * v_half
                
                # Apply Boundary (Torus)
                topology_id = kwargs.get('topology', 0)
                curr_x = apply_boundary_python(curr_x, topology_id)
                
                # 3. Kick (half step velocity at new pos)
                res_half = self.christoffel(v_half, curr_x, force=force, **kwargs)
                if isinstance(res_half, tuple):
                    gamma_half, mu_half = res_half
                else:
                    gamma_half, mu_half = res_half, 0.0
                    
                curr_v = (v_half + h * (force - gamma_half)) / (1.0 + h * mu_half)
        finally:
            self.christoffel.return_friction_separately = was_separate
        
        return curr_x, curr_v
