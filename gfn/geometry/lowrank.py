import torch
import torch.nn as nn

try:
    from gfn.cuda.ops import christoffel_fused,CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False

class LowRankChristoffel(nn.Module):
    r"""
    Computes the Christoffel symbols \Gamma^k_{ij} using a low-rank decomposition.
    To ensure symmetry in lower indices (torsion-free), we use a symmetric decomposition:
    \Gamma^k_{ij} = \sum_{r=1}^R \lambda_{kr} * (U_{ir} * U_{jr})
    
    Args:
        dim (int): Dimension of the manifold (hidden size).
        rank (int): Rank of the decomposition.
    """
    def __init__(self, dim, rank=16, physics_config=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.config = physics_config or {}
        self.clamp_val = self.config.get('stability', {}).get('curvature_clamp', 5.0)
        self.is_torus = self.config.get('topology', {}).get('type', '').lower() == 'torus'
        
        # Gates for Toroidal coordinates require Fourier features [sin(x), cos(x)]
        gate_input_dim = 2 * dim if self.is_torus else dim
        
        # Factors to reconstruct Gamma
        # U: [dim, rank] - represents the "basis" for the input indices i, j
        # W: [dim, rank] - represents the "basis" for the output index k (or weighting)
        # LEVEL 7: CURVATURE INJECTION
        # Higher initialization (0.2) ensures manifold logic is energetic from step 1
        # LEVEL 22: FLAT TORUS START (Euclidean Default)
        # We initialize U, W to ZERO to ensure \Gamma = 0 initially.
        # This prevents "Metric Noise" from trapping particles.
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.W = nn.Parameter(torch.zeros(dim, rank))
        
        # Friction coefficient for Conformal Symplectic System
        self.friction = self.config.get('stability', {}).get('friction', 0.05)
        
        # Position Gate V: dim -> 1 (Scalar gravity well strength)
        # We start with near-zero weights so initially there are no gravity wells.
        self.V = nn.Linear(gate_input_dim, 1, bias=False)
        nn.init.zeros_(self.V.weight)
        
        # Adaptive Curvature Gate (The Valve): dim -> dim
        # Learns when to apply curvature vs coasting
        self.gate_proj = nn.Linear(gate_input_dim, dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0) # Start OPEN (sigmoid(2) ~ 0.88)
        
        # LEVEL 25: THE CLUTCH (Input-Dependent Friction)
        # Mechanism to switch between Hamiltonian (Coast) and Aristotelian (Write) regimes.
        # Friction = sigmoid(W_x * x + W_f * force + bias)
        
        # State component of friction gate
        self.forget_gate = nn.Linear(gate_input_dim, dim)
        nn.init.normal_(self.forget_gate.weight, std=0.01)
        
        # Force component of friction gate
        self.input_gate = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.input_gate.weight, std=0.01)
        
        # BIAS INITIALIZATION: 0.0 (Neutral/Semi-Engaged)
        # Allows gradient descent to push towards +5 (Write) or -5 (Coast) easily.
        nn.init.constant_(self.forget_gate.bias, 0.0) 
        
    def forward(self, v, x=None, force=None, **kwargs):
        """
        Compute Generalized Force: Γ(v, v) + Friction(x)*v
        
        Output represents the effective "Resistance" to motion.
        Acc = F_ext - Output
        """
        # Try Fused CUDA Kernel first (Placeholder: Kernel needs to support forget_gate)
        try:
            from gfn.cuda.ops import lowrank_christoffel_fused, CUDA_AVAILABLE
            if CUDA_AVAILABLE and v.is_cuda and v.dim() == 2:
                x_empty = torch.empty(0, device=v.device)
                V_empty = torch.empty(0, device=v.device)
                gamma_cuda = lowrank_christoffel_fused(v, self.U, self.W, x_empty, V_empty, 0.0, 1.0, 1.0)
                
                # Apply friction manually if CUDA kernel doesn't support it yet
                if x is not None:
                     friction = torch.sigmoid(self.forget_gate(x))
                     gamma_cuda = gamma_cuda + friction * v
                return torch.clamp(gamma_cuda, -self.clamp_val, self.clamp_val)
        except Exception:
            pass
    
        # Fallback: Vectorized Implementation
        if v.dim() == 3 and self.U.dim() == 3:
            proj = torch.bmm(v, self.U) 
            norm = torch.norm(proj, dim=-1, keepdim=True)
            scale = 1.0 / (1.0 + norm)
            sq = (proj * proj) * scale 
            gamma = torch.bmm(sq, self.W.transpose(1, 2)) 
        else:
            proj = torch.matmul(v, self.U)
            norm = torch.norm(proj, dim=-1, keepdim=True)
            scale = 1.0 / (1.0 + norm)
            sq = (proj * proj) * scale
            gamma = torch.matmul(sq, self.W.t())
            
        # APPLY THE CLUTCH (DYNAMIC FRICTION)
        if x is not None:
            # Map to Periodic Space if needed
            if self.is_torus:
                 x_in = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            else:
                 x_in = x
                 
            # Base friction from state
            gate_activ = self.forget_gate(x_in)
            
            # Input-dependent modulation
            if force is not None:
                gate_activ = gate_activ + self.input_gate(force)
                
            # LEVEL 25: FRICTION GAIN (Aristotelian Override)
            # STABILITY FIX: Gain 5.0 is sufficient and more stable
            mu = torch.sigmoid(gate_activ) * 5.0
            
            # If we want to return BOTH for implicit integration
            if getattr(self, 'return_friction_separately', False):
                 return gamma, mu
                 
            gamma = gamma + mu * v
            
        return gamma
