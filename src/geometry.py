import torch
import torch.nn as nn

class LowRankChristoffel(nn.Module):
    r"""
    Computes the Christoffel symbols \Gamma^k_{ij} using a low-rank decomposition.
    To ensure symmetry in lower indices (torsion-free), we use a symmetric decomposition:
    \Gamma^k_{ij} = \sum_{r=1}^R \lambda_{kr} * (U_{ir} * U_{jr})
    
    Args:
        dim (int): Dimension of the manifold (hidden size).
        rank (int): Rank of the decomposition.
    """
    def __init__(self, dim, rank=16):
        super().__init__()
        self.dim = dim
        self.rank = rank
        
        # Factors to reconstruct Gamma
        # U: [dim, rank] - represents the "basis" for the input indices i, j
        # W: [dim, rank] - represents the "basis" for the output index k (or weighting)
        # Init very small to start with FLAT manifold (Euclidean geometry)
        # This helps in preserving long-term dependencies (linear dynamics)
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.001)
        self.W = nn.Parameter(torch.randn(dim, rank) * 0.001)
        
        # Position Gate V: dim -> 1 (Scalar gravity well strength)
        # We start with near-zero weights so initially there are no gravity wells.
        self.V = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.V.weight)
        
    def forward(self, v, x=None):
        """
        Compute Γ(v, v) = W * (U^T v)^2
        
        If x is provided, we apply Dynamic Curvature Modulation:
        Γ_dynamic = Γ_static * (1 + sigmoid(V^T x))
        """
        # Try CUDA kernel first (Only supports static curvature for now)
        # TODO: Update CUDA kernel to support dynamic curvature
        # if x is None:
        #     try:
        #         from src.cuda.ops import christoffel_fused
        #         return torch.clamp(christoffel_fused(v, self.U, self.W), -5.0, 5.0)
        #     except:
        #         pass
        
        # PyTorch Implementation
        # v: [batch, dim]
        proj = torch.matmul(v, self.U) # [batch, rank]
        sq = proj * proj # [batch, rank]
        out = torch.matmul(sq, self.W.t()) # [batch, dim]
        
        # Dynamic Curvature Modulation (Gravity Wells)
        if x is not None:
            # V(x) -> scalar modulation
            # We want deviations from "flat" space to be localized
            modulation = torch.sigmoid(self.V(x)) # Range (0, 1)
            # Factor: 1.0 (unchanged) to 2.0 (doubled curvature)
            # Or we can make it multiplicative: out * (1 + mod)
            out = out * (1.0 + modulation)
            
        # Stability: Tight clamp prevents "exploding" curvature
        # This is CRITICAL for long-term training stability
        return torch.clamp(out, -5.0, 5.0)

class SymplecticIntegrator(nn.Module):
    r"""
    Integrates the geodesic equation: d^2x/dt^2 + \Gamma(v, v) = F
    using a symplectic method (e.g., Velocity Verlet) to preserve energy/stability.
    """
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        r"""
        One step of integration.
        
        Velocity Verlet:
        1. v_{t+0.5} = v_t + 0.5 * a(x_t, v_t) * dt
        2. x_{t+1} = x_t + v_{t+0.5} * dt
        3. a_{t+1} = a(x_{t+1}, v_{t+0.5})  (Approximation: depend on v_{t+0.5})
        4. v_{t+1} = v_{t+0.5} + 0.5 * a_{t+1} * dt
        
        Acceleration a(x, v) = F - \Gamma(v, v)
        """
        dt = self.dt * dt_scale
        
        
        # Acceleration at t
        gamma_term = self.christoffel(v, x)
        acc_t = -gamma_term
        if force is not None:
            acc_t = acc_t + force
            
        # Half step velocity
        v_half = v + 0.5 * acc_t * dt
        
        # Full step position
        x_next = x + v_half * dt
        
        # New acceleration (using v_half as approximation for velocity at t+1 for Gamma)
        # In strict geodesic, Gamma depends on position x_next (metric at x_next).
        # But our Global LowRankChristoffel assumes constant curvature field or implicit dependency.
        # If we want state-dependent curvature, ChristoffelParametrization should interpret 'x'.
        # For simplicity/efficiency as per paper "Global/Local metric", we assume 
        # local metric is predicted or Gamma is computed globally or from hidden state.
        # Let's assume standard GFN where Gamma might be somewhat constant or we just use v_half.
        
        # Re-eval gamma at new state (using x_next for dynamic curvature)
        gamma_term_next = self.christoffel(v_half, x_next) 
        acc_next = -gamma_term_next
        if force is not None:
            # Force might be constant for the step or depend on x (e.g. potential gradient)
            # Assuming constant force from input token for this step
            acc_next = acc_next + force
            
        # Full step velocity
        v_next = v_half + 0.5 * acc_next * dt
        
        return x_next, v_next

class RK4Integrator(nn.Module):
    r"""
    Runge-Kutta 4 (RK4) Integrator for the geodesic equation.
    System:
    dx/dt = v
    dv/dt = F - \Gamma(v, v)
    
    State Y = [x, v]
    """
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        r"""
        One step of RK4 integration.
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
        """
        dt = self.dt * dt_scale
        
        def dynamics(current_x, current_v):
            # dv/dt = F - Gamma(v, v, x)
            acc = -self.christoffel(current_v, current_x)
            if force is not None:
                acc = acc + force
            return acc
            
        # k1
        dx1 = v
        dv1 = dynamics(x, v)
        
        # k2
        v2 = v + 0.5 * dt * dv1
        x2 = x + 0.5 * dt * dx1
        dx2 = v2
        dv2 = dynamics(x2, v2)
        
        # k3
        v3 = v + 0.5 * dt * dv2
        x3 = x + 0.5 * dt * dx2
        dx3 = v3
        dv3 = dynamics(x3, v3)
        
        # k4
        v4 = v + dt * dv3
        x4 = x + dt * dx3
        dx4 = v4
        dv4 = dynamics(x4, v4)
        
        # Update
        x_next = x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        v_next = v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        
        return x_next, v_next

class HeunIntegrator(nn.Module):
    r"""
    Heun's Method (Improved Euler / RK2).
    2nd order accuracy with only 2 evaluations per step.
    Great balance between accuracy and speed.
    """
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        dt = self.dt * dt_scale
        
        def dynamics(current_x, current_v):
            acc = -self.christoffel(current_v, current_x)
            if force is not None:
                acc = acc + force
            return acc
            
        # k1
        dx1 = v
        dv1 = dynamics(x, v)
        
        # Predictor step (Euler)
        v_pred = v + dt * dv1
        x_pred = x + dt * dx1
        
        # k2 (using predicted velocity AND position)
        dx2 = v_pred
        dv2 = dynamics(x_pred, v_pred)
        
        # Corrector step
        x_next = x + (dt / 2.0) * (dx1 + dx2)
        v_next = v + (dt / 2.0) * (dv1 + dv2)
        
        return x_next, v_next


class LeapfrogIntegrator(nn.Module):
    r"""
    Leapfrog (Störmer-Verlet) Integrator for Geodesic Flow.
    
    A symplectic integrator that preserves the Hamiltonian structure,
    ensuring energy conservation and long-term stability.
    
    Algorithm (Kick-Drift-Kick variant):
        1. v_{1/2} = v + (dt/2) * a(x, v)           [Half-Kick]
        2. x_{new} = x + dt * v_{1/2}               [Full-Drift]
        3. v_{new} = v_{1/2} + (dt/2) * a(x_{new}, v_{1/2})  [Half-Kick]
    
    This is time-reversible and symplectic, making it ideal for
    preserving phase-space volume and preventing energy drift.
    """
    
    def __init__(self, christoffel_net, dt=0.1):
        super().__init__()
        self.christoffel = christoffel_net
        self.dt = dt
        
    def forward(self, x, v, force=None, dt_scale=1.0):
        """
        Perform one Leapfrog (Störmer-Verlet) step.
        
        Uses custom fused CUDA kernel when available for 4-5x speedup.
        
        Args:
            x: Position
            v: Velocity
            force: External force
            dt_scale: Adaptive time scaling (Golden Integration)
        """
        if force is None:
            force = torch.zeros_like(x)
            
        # Try CUDA kernel first
        try:
            from src.cuda.ops import leapfrog_fused
            return leapfrog_fused(x, v, force, self.christoffel.U, self.christoffel.W, self.dt, dt_scale)
        except:
            # Fallback to PyTorch
            effective_dt = self.dt * dt_scale
            
            # Fallback to PyTorch
            effective_dt = self.dt * dt_scale
            
            # Half-step velocity
            gamma = self.christoffel(v, x)
            v_half = v + 0.5 * effective_dt * (force - gamma)
            
            # Full-step position
            x_new = x + effective_dt * v_half
            
            # Half-step velocity again
            # Use x_new for new Gamma calculation
            gamma_half = self.christoffel(v_half, x_new)
            v_new = v_half + 0.5 * effective_dt * (force - gamma_half)
            
            return x_new, v_new
