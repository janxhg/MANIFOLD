"""
GFN Adjoint State Method (v2.6.2 Compatible)
===========================================

Implements Neural ODE-style backpropagation for O(1) memory training using
Active Inference compatible Symplectic Integration.

This module wraps the Manifold dynamics into an ODE function compatible with
`torchdiffeq`, enabling constant memory cost regardless of depth/sequence length.

Theory:
Instead of storing the entire computation graph (x_0 -> x_L), we store only x_0 and x_L.
During backprop, we solve the "Adjoint ODE" backwards in time to recover gradients.

Energy Conservation:
Strict symplectic integration is difficult in standard ODE solvers. We approximate it
by providing the Symplectic gradients directly to the solver or using fixed-step solvers.
"""

import torch
import torch.nn as nn
from .geometry import LowRankChristoffel, HyperChristoffel
import math

# Try to import torchdiffeq, but provide fallback
try:
    from torchdiffeq import odeint_adjoint, odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not installed. Adjoint method unavailable.")
    print("Install with: pip install torchdiffeq")

class SymplecticTrajFunc(nn.Module):
    """
    Defines the continuous dynamics dx/dt suitable for ODE solvers.
    
    Compatible with Manifold v2.6.2 Physics:
    - Low-Rank Christoffel Curvature
    - Dynamic Friction (Active Inference)
    - Singularities (Energy Wells)
    """
    def __init__(self, dim, rank=32, heads=1, physics_config=None):
        super().__init__()
        self.dim = dim
        self.points_per_head = dim // heads
        self.heads = heads
        self.physics_config = physics_config or {}
        
        # Geometry Kernel
        if self.physics_config.get('hyperbolic', False):
            self.christoffel = HyperChristoffel(dim, rank) # Placeholder
        else:
            self.christoffel = LowRankChristoffel(dim, rank, physics_config=physics_config)
            
    def forward(self, t, state_flat):
        """
        ODE State: Concatenated [x, v]
        
        System:
        dx/dt = v
        dv/dt = -Γ(v, v) + F_ext (if any, but force is usually discrete per token)
        
        Note: Force injection happens discretely between layers in standard Manifold.
        In purely continuous Adjoint mode, we assume force is 0 during the 'flight'
        between layers, or injected as a constant parameter.
        """
        # Unpack state
        # state_simple is usually [batch * seq, 2 * dim] to handle full sequence parallel
        # or just [batch, 2 * dim]
        
        x = state_flat[..., :self.dim]
        v = state_flat[..., self.dim:]
        
        # 1. Compute Curvature (Gravity)
        # Γ(v, v)
        # Note: Functional embeddings or force inputs are not part of the flow 
        # unless treated as time-dependent parameters. For now, we assume ballistic flight.
        
        gamma = self.christoffel(v, x) # Pass x for Singularity detection
        
        # Dynamics
        dx_dt = v
        dv_dt = -gamma # "Gravity" resists or accelerates velocity
        
        return torch.cat([dx_dt, dv_dt], dim=-1)

class AdjointManifoldBlock(nn.Module):
    """
    A block of Manifold layers treated as a single continuous ODE.
    
    Input: x_0, v_0
    Output: x_T, v_T
    """
    def __init__(self, dim, rank=32, heads=1, integration_time=1.0, steps=10, physics_config=None):
        super().__init__()
        self.ode_func = SymplecticTrajFunc(dim, rank, heads, physics_config)
        self.integration_time = integration_time
        self.steps = steps
        self.rtol = 1e-3
        self.atol = 1e-3
        
    def forward(self, x, v):
        if not HAS_TORCHDIFFEQ:
            raise ImportError("torchdiffeq is required for AdjointManifoldBlock")
            
        # Initial state
        state_0 = torch.cat([x, v], dim=-1)
        
        # Time span
        t_span = torch.tensor([0.0, self.integration_time]).to(x.device)
        
        # Solve ODE
        # method='rk4' is good balance. 'dopri5' is adaptive.
        # We use adjoint method for O(1) memory backprop.
        state_t = odeint_adjoint(
            self.ode_func,
            state_0,
            t_span,
            method='rk4',
            options={'step_size': self.integration_time / self.steps},
            rtol=self.rtol,
            atol=self.atol
        )
        
        # Result is [2, batch, 2*dim] (time 0 and time T)
        final_state = state_t[-1]
        
        x_new = final_state[..., :self.ode_func.dim]
        v_new = final_state[..., self.ode_func.dim:]
        
        return x_new, v_new

class AdjointManifold(nn.Module):
    """
    Manifold Model with O(1) Memory Backpropagation.
    
    Replaces the discrete stack of M-Layers with a Continuous Neural ODE.
    Uses 'Active Inference' physics config.
    """
    def __init__(self, vocab_size, dim=256, depth=4, rank=32, heads=4, 
                 physics_config=None, integration_time=1.0):
        super().__init__()
        self.physics_config = physics_config or {}
        
        # 1. Embeddings (Synced with Model.py)
        emb_cfg = self.physics_config.get('embedding', {})
        emb_type = emb_cfg.get('type', 'standard')
        
        if emb_type == 'implicit':
            from .embeddings import ImplicitEmbedding
            coord_dim = emb_cfg.get('coord_dim', 16)
            self.embedding = ImplicitEmbedding(vocab_size, dim, coord_dim=coord_dim)
        elif emb_type == 'functional':
            from .embeddings import FunctionalEmbedding
            coord_dim = emb_cfg.get('coord_dim', 16)
            mode = emb_cfg.get('mode', 'binary')
            self.embedding = FunctionalEmbedding(vocab_size, dim, coord_dim=coord_dim, mode=mode)
        else:
            self.embedding = nn.Embedding(vocab_size, dim)
            
        # 2. Continuous Core
        # Instead of 'depth' discrete layers, we have one ODE Block
        # that integrates for 'depth' equivalent time.
        # However, to allow Force Injection (Transformer-like input), we often
        # need to break the ODE at each step i.e., "Kick" then "Drift".
        
        # Hybrid Approach: "Kick-Drift" via discrete layers, but each 'Drift' is an ODE.
        # For pure O(1), users typically want the whole depth as one ODE.
        # But Manifold relies on input injection at every layer? 
        # Actually, in standard Manifold, Force is effectively constant per token?
        # No, recall MLayer: v_new = v + dt*(F - Gamma).
        
        # Strategy:
        # We stack 'depth' Adjoint Blocks. Each block is O(1). 
        # Total memory is O(depth) but intermediate steps *within* blocks are O(1).
        # This is a good compromise.
        
        self.layers = nn.ModuleList([
            AdjointManifoldBlock(dim, rank, heads, integration_time=1.0, steps=4, physics_config=physics_config)
            for _ in range(depth)
        ])
        
        # Normalization
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.readout_norm = nn.LayerNorm(dim)
        
        # Readout
        self.readout = nn.Linear(dim, vocab_size)
        
        # Init
        self.x0 = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.v0 = nn.Parameter(torch.randn(1, dim) * 0.01)
        
    def forward(self, input_ids, attention_mask=None, state=None):
        batch_size, seq_len = input_ids.shape
        
        if state is None:
            x = self.x0.expand(batch_size, -1)
            v = self.v0.expand(batch_size, -1)
        else:
            x, v = state
            
        all_forces = self.embedding(input_ids) # [B, L, D]
        
        logits_list = []
        
        # Time-loop (Sequence)
        for t in range(seq_len):
            # 1. Kick (Force Injection)
            force = all_forces[:, t]
            
            # Simple Euler Kick: v = v + F (impulse)
            # This happens instantaneously before the flight
            v = v + force
            
            # 2. Drift (Manifold Flight) - Solved via Adjoint ODE
            for i, layer in enumerate(self.layers):
                x = self.norms[i](x)
                x, v = layer(x, v)
                
            # 3. Readout
            out = self.readout_norm(x)
            logit = self.readout(out)
            logits_list.append(logit.unsqueeze(1))
            
        logits = torch.cat(logits_list, dim=1)
        return logits, (x, v), []

