import torch
import torch.nn as nn

class ToroidalChristoffel(nn.Module):
    """
    Toroidal Geometry (Curved Torus).
    
    Metric for a 2D torus in 3D is:
    g = diag(r^2, (R + r cos th)^2)
    
    We generalize this to N dimensions by alternating 'inner' and 'outer' coords,
    or simply providing a periodic metric that isn't identity.
    
    Structure:
    - Theta: Local circle coordinate
    - Phi: Global circle coordinate
    """
    def __init__(self, dim, physics_config=None):
        super().__init__()
        self.dim = dim
        self.config = physics_config or {}
        # Toroidal parameters: R (major radius), r (minor radius)
        self.R = self.config.get('topology', {}).get('major_radius', 2.0)
        self.r = self.config.get('topology', {}).get('minor_radius', 1.0)
        
    def forward(self, v, x, **kwargs):
        """
        Computed Christoffel Force: Gamma(v, v)^k = Gamma^k_ij v^i v^j
        """
        if x is None: return torch.zeros_like(v)
        
        # We assume coordinates come in pairs (th, ph) or are all S1.
        # Let's implement a 'Chain Torus' where th_i affects ph_i.
        
        # 1. Coordinate parity
        # For even-dim, half are theta (inner), half are phi (outer)
        # For odd-dim, we pad or alternate.
        
        # We'll use a simple tiling: x[0] affects x[1], x[1] affects x[2]... 
        # But for symmetry, we'll use: 
        # x_even = theta (inner)
        # x_odd = phi (outer)
        
        # Precompute cos/sin of theta
        cos_th = torch.cos(x)
        sin_th = torch.sin(x)
        
        gamma = torch.zeros_like(v)
        
        # N-dimensional torus logic: 
        # For each pair (i, i+1):
        # Gamma^i_{i+1, i+1} = (R + r cos x_i) sin x_i / r
        # Gamma^{i+1}_{i+1, i} = - (r sin x_i) / (R + r cos x_i)
        
        for i in range(0, self.dim - 1, 2):
            th = x[..., i]
            ph = x[..., i+1]
            v_th = v[..., i]
            v_ph = v[..., i+1]
            
            # Gamma^th_{ph, ph}
            # Gamma_th = Gamma^th_ph_ph * v_ph^2
            term_th = (self.R + self.r * torch.cos(th)) * torch.sin(th) / self.r
            gamma[..., i] = term_th * (v_ph ** 2)
            
            # Gamma^ph_{ph, th}
            # Gamma_ph = 2 * Gamma^ph_ph_th * v_ph * v_th
            term_ph = -(self.r * torch.sin(th)) / (self.R + self.r * torch.cos(th))
            gamma[..., i+1] = 2.0 * term_ph * v_ph * v_th
            
        return gamma * 0.1 # Scaling for stability
