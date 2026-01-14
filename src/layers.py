import torch
import torch.nn as nn
from .geometry import LowRankChristoffel, SymplecticIntegrator, RK4Integrator, HeunIntegrator, LeapfrogIntegrator

class RiemannianGating(nn.Module):
    """
    Computes a scalar curvature-based gating mechanism.
    If curvature is high, dt should be small (complex region).
    If curvature is low (flat), dt can be large (skip connection behavior).
    """
    def __init__(self, dim):
        super().__init__()
        self.curvature_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid() # Range [0, 1]
        )
        
    def forward(self, x):
        """
        Returns a scaling factor for dt.
        """
        # Scalar curvature estimate "R"
        # We simplify this to a learned function of state x.
        g = self.curvature_net(x) 
        return g

class GLayer(nn.Module):
    """
    Geodesic Layer:
    Takes current state (x, v) and input token force F.
    Evolves state via Geodesic Flow.
    
    Available integrators:
        - 'heun': Heun's method (RK2) - Fast & stable [DEFAULT]
        - 'rk4': Runge-Kutta 4 - High accuracy
        - 'symplectic': Velocity Verlet - Energy preserving
        - 'leapfrog': Störmer-Verlet - Best symplectic
    """
    def __init__(self, dim, rank=16, base_dt=0.1, integrator_type='heun'):
        super().__init__()
        self.christoffel = LowRankChristoffel(dim, rank)
        
        if integrator_type == 'rk4':
            self.integrator = RK4Integrator(self.christoffel, dt=base_dt)
        elif integrator_type == 'heun':
            self.integrator = HeunIntegrator(self.christoffel, dt=base_dt)
        elif integrator_type == 'leapfrog':
            self.integrator = LeapfrogIntegrator(self.christoffel, dt=base_dt)
        else:  # symplectic (default fallback)
            self.integrator = SymplecticIntegrator(self.christoffel, dt=base_dt)
            
        self.gating = RiemannianGating(dim)
        self.base_dt = base_dt
        
    def forward(self, x, v, force=None):
        """
        Args:
            x: Position (Manifold coordinates) [batch, dim]
            v: Velocity (Tangent vector) [batch, dim]
            force: External force (Input token embedding) [batch, dim]
        Returns:
            x_next, v_next
        """
        # Dynamic time-step based on curvature
        # High curvature -> smaller effective dt (time dilation)
        # Gate is in [0, 1]
        gate = self.gating(x)
        
        # Golden Integration:
        # Pass curvature-based gating factor directly to the integrator
        # to scale dt dynamically (physically correct time dilation).
        x_out, v_out = self.integrator(x, v, force, dt_scale=gate)
        
        return x_out, v_out
