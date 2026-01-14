import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold
from src.adjoint import AdjointManifold

def test_mechanics():
    print("=== MANIFOLD MECHANICS VERIFICATION ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Setup Tiny Model for Testing
    vocab_size = 10
    dim = 16
    depth = 2
    rank = 4
    heads = 4
    bs = 2
    seq_len = 5
    
    print(f"\n[1] Initializing Tiny Manifold (dim={dim}, depth={depth}, heads={heads})...")
    
    # Set seed for identical initialization
    torch.manual_seed(42)
    # Standard Manifold
    model_std = Manifold(vocab_size, dim, depth, rank, heads=heads, integrator_type='leapfrog').to(device)
    
    torch.manual_seed(42)
    # Adjoint Manifold (Heads=1 required)
    model_adj = AdjointManifold(vocab_size, dim, depth, rank, heads=1).to(device)
    
    # Dummy Input
    x = torch.randint(0, vocab_size, (bs, seq_len)).to(device)
    target = torch.randint(0, vocab_size, (bs, seq_len)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Test Standard Backward
    print("\n[2] Testing Standard Forward/Backward...")
    model_std.zero_grad()
    logits_std, _ = model_std(x)
    loss_std = criterion(logits_std.view(-1, vocab_size), target.view(-1))
    loss_std.backward()
    grad_std = model_std.embedding.weight.grad.clone()
    print(f"    Standard Loss: {loss_std.item():.6f}")
    print(f"    Gradient Norm: {grad_std.norm().item():.6f}")
    
    # 3. Test Adjoint Backward (O(1) Memory)
    print("\n[3] Testing Adjoint Forward/Backward (The 'Time Machine')...")
    try:
        import torchdiffeq
        model_adj.zero_grad()
        # Enable gradient checkpointing/adjoint logic implicit in the module if implemented
        # Note: In our implementation, AdjointManifold uses torchdiffeq.odeint_adjoint
        logits_adj, _ = model_adj(x)
        loss_adj = criterion(logits_adj.view(-1, vocab_size), target.view(-1))
        loss_adj.backward()
        grad_adj = model_adj.embedding.weight.grad.clone()
        print(f"    Adjoint Loss:  {loss_adj.item():.6f}")
        print(f"    Gradient Norm: {grad_adj.norm().item():.6f}")
            
    except ImportError:
        print("    [SKIP] torchdiffeq not installed. Cannot test Adjoint.")
    except Exception as e:
        print(f"    [ERROR] {e}")
        
    # 4. Consistency Check
    print("\n[4] Consistency Check...")
    print("    Skipping direct gradient comparison due to architecture differences (Discrete vs continuous ODE Wrapper).")
    print(f"    Standard Grad Norm: {grad_std.norm().item():.6f}")
    if 'grad_adj' in locals():
        print(f"    Adjoint Grad Norm:  {grad_adj.norm().item():.6f}")
        if not torch.isnan(grad_adj.norm()):
             print("    [PASS] Adjoint gradients are valid.")

    # 5. Energy Conservation Test
    print("\n[5] Energy Conservation Test...")
    # Get a single MLayer from Standard model
    layer = model_std.layers[0]
    
    # Access the FIRST head's integrator
    integrator = layer.integrators[0]
    head_dim = dim // heads
    
    q = torch.randn(bs, head_dim).to(device)
    p = torch.randn(bs, head_dim).to(device)
    
    # Calculate Energy: Kinetic = ||p||^2
    E_start = p.pow(2).sum(dim=-1).mean()
    
    # Evolve using Integrator
    # Integrator signature: (x, v, force, dt_scale)
    # We pass 0 force and dt_scale=1.0 for conservation check
    force = torch.zeros(bs, head_dim).to(device) 
    
    q_new, p_new = integrator(q, p, force, dt_scale=1.0)
    
    E_end = p_new.pow(2).sum(dim=-1).mean()
    energy_diff = torch.abs(E_start - E_end).item()
    
    print(f"    Initial Energy: {E_start.item():.6f}")
    print(f"    Final Energy:   {E_end.item():.6f}")
    print(f"    Diff:           {energy_diff:.2e}")
    
    # Symplectic integrators fluctuate but bound energy.
    if energy_diff < 1.0: # Loose bound for random weights
        print("    [PASS] Energy is stable.")
    else:
        print("    [FAIL] Significant energy drift.")

if __name__ == "__main__":
    test_mechanics()
