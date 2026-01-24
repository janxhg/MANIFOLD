import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force CUDA ops load
try:
    from gfn.cuda import ops
except ImportError:
    pass

from gfn.model import Manifold
from gfn.optim import RiemannianAdam

def run_probe(seq_len=100, device='cuda'):
    print(f"--- Gradient Probe (L={seq_len}) ---")
    
    # Init Model (Same config as Superiority Benchmark)
    dim = 128
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'binary', 'coord_dim': 16},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {'enabled': True, 'plasticity': 0.1},
        'singularities': {'enabled': True, 'strength': 5.0},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.3}
    }
    
    model = Manifold(vocab_size=2, dim=dim, depth=6, heads=1, 
                    integrator_type='leapfrog', base_dt=0.3, physics_config=physics_config).to(device)
    
    # Init Friction (Match Benchmark: Bias -3.0)
    for layer in model.layers:
        target = layer.macro_manifold if hasattr(layer, 'macro_manifold') else layer
        for head in target.christoffels:
            nn.init.constant_(head.forget_gate.bias, -3.0)
            
    # Input
    x = torch.randint(0, 2, (32, seq_len), device=device)
    y = torch.cumsum(x, dim=1) % 2
    
    # Enable gradients
    model.train()
    logits, (x_final, v_final), _ = model(x)
    
    # Check Saturation
    v_abs = v_final.abs()
    flat_v = v_abs.view(-1)
    
    # Saturation Threshold (14.9 is close to 15.0 limit)
    saturated_count = (flat_v >= 14.9).sum().item()
    total_params = flat_v.numel()
    saturation_rate = 100.0 * saturated_count / total_params
    
    v_max = flat_v.max().item()
    v_mean = flat_v.mean().item()
    
    print(f"Velocity Stats: Max={v_max:.2f}, Mean={v_mean:.2f}")
    print(f"Saturation Rate (|v|>=14.9): {saturation_rate:.2f}%")
    
    # Backward
    loss = nn.BCEWithLogitsLoss()(logits[:, :, 0], y.float())
    loss.backward()
    
    # Measure Gradient at Input Layer (Parameter)
    if model.x0.grad is not None:
        grad_norm = model.x0.grad.norm().item()
        print(f"Gradient Norm (x0): {grad_norm:.6e}")
        
        if grad_norm == 0.0:
            print("!! GRADIENT DEATH !! (Norm is 0)")
        elif grad_norm < 1e-9:
            print("!! GRADIENT VANISHED !!")
    else:
        print("Error: x0 has no grad?")
        
    print("-" * 30)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {device}")
        lens = [20, 100, 500] 
        for L in lens:
            run_probe(L, device)
    else:
        print("CUDA required for this probe.")
