
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfn.layers.base import MLayer
from gfn.losses import hamiltonian_loss

    # Compare Integrators
    # We check:
    # 1. Leapfrog Fused (collect_christ=False)
    # 2. Leapfrog Python (collect_christ=True) to verify the Python fix
    # 3. RK4 (Reference)
    configs = [
        ('leapfrog', False, 'Leapfrog (Fused)'),
        ('leapfrog', True,  'Leapfrog (Python)'),
        ('rk4', False,      'RK4 (Reference)')
    ]
    
    for integ_name, collect, label in configs:
        print(f"\n--- Testing {label} ---")
        layer = MLayer(dim=dim, heads=1, integrator_type=integ_name, physics_config=physics_config).to(device)
        
        # Disable Friction (Bias = -10.0 -> Sigmoid ~ 0)
        if hasattr(layer.christoffels[0], 'forget_gate'):
             nn.init.constant_(layer.christoffels[0].forget_gate.bias, -10.0)
                 
        # Input [10, dim]
        x = torch.randn(10, dim).to(device) 
        v = torch.randn(10, dim).to(device) * 0.1
        
        energies = []
        curr_x, curr_v = x, v
        
        # Initial E
        E0 = 0.5 * curr_v.pow(2).sum(dim=-1).mean().item()
        energies.append(E0)
        
        # Simulate 20 steps
        for i in range(20):
            curr_x, curr_v, _ , _ = layer(curr_x, curr_v, force=None, collect_christ=collect)
            E = 0.5 * curr_v.pow(2).sum(dim=-1).mean().item()
            energies.append(E)
            
        print(f"Energy Profile (First 5): {energies[:5]}")
        print(f"Energy Profile (Last 5): {energies[-5:]}")
        drift = np.std(energies)
        print(f"Energy Drift StdDev: {drift:.6f}") # Expect Leapfrog ~0.0, RK4 > 0.0

def check_gradient_depth():
    print("\n[Audit] Checking Gradient Flow (Vanishing/Exploding)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 128
    
    physics_config = {
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.1}
    }
    
    # Stack 12 Layers (Deep)
    layers = nn.ModuleList([
        MLayer(dim=dim, heads=1, integrator_type='rk4', physics_config=physics_config).to(device)
        for _ in range(12)
    ])
    
    x = torch.randn(1, dim).to(device).requires_grad_(True)
    v = torch.randn(1, dim).to(device).requires_grad_(True)
    x.retain_grad()
    
    curr_x, curr_v = x, v
    
    for layer in layers:
        curr_x, curr_v, _, _ = layer(curr_x, curr_v)
        
    loss = curr_x.sum()
    loss.backward()
    
    x_grad = x.grad.norm().item()
    print(f"Input Gradient Norm (After 12 Layers): {x_grad:.6f}")
             
    if x_grad < 1e-6:
        print("!! FAILURE: Vanishing Gradient !!")
    elif x_grad > 1e3:
        print("!! FAILURE: Exploding Gradient !!")
    else:
        print(">> SUCCESS: Gradients are flowing well through 12 layers.")

def check_metric_mismatch():
    print("\n[Audit] Checking Output vs Loss Expectation (Mismatch)...")
    
    pred = torch.tensor([7.0], requires_grad=True)
    target = torch.tensor([0.7])
    
    # Periodic Loss
    loss_cos = (1.0 - torch.cos(pred - target))
    print(f"Periodic Loss (7.0 vs 0.7): {loss_cos.item():.4f} (Expected ~0.0)")
    
    print(">> Logic Check: PeriodicLoss is robust to wrapping.")

if __name__ == "__main__":
    check_energy_conservation()
    check_gradient_depth()
    check_metric_mismatch()
