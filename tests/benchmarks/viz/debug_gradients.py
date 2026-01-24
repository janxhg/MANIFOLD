
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfn.geometry.toroidal import ToroidalChristoffel
from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator

def debug_gradients():
    torch.manual_seed(42)
    dim = 128
    
    # 1. Setup Components
    geo = ToroidalChristoffel(dim)
    integ = LeapfrogIntegrator(geo, dt=0.3)
    
    # Friction Tuning (as applied in fix)
    nn.init.constant_(geo.forget_gate.bias, 1.2)
    
    # 2. Inputs (Requires Grad)
    # Use leaf tensor correctly
    x_leaf = torch.randn(1, dim) * 0.02
    x = x_leaf.clone().detach().requires_grad_(True)
    x.retain_grad()
    
    v_leaf = torch.randn(1, dim) * 0.01
    v = v_leaf.clone().detach().requires_grad_(True)
    v.retain_grad()
    
    force = torch.randn(1, dim, requires_grad=True) 
    
    print(f"Initial x: {x.mean().item():.4f}")
    
    # 3. Forward Steps (Model Depth = 6)
    curr_x, curr_v = x, v
    
    # MLayer logic simulation
    f_t = force * 25.0 # Stable Impulse Scale
    
    for layer in range(6):
        # Integrator Step
        curr_x, curr_v = integ(curr_x, curr_v, force=f_t, dt_scale=1.0)
        
        # Velocity Saturation (Crucial for Stability)
        curr_v = 20.0 * torch.tanh(curr_v / 20.0)
        
    final_x = curr_x
    print(f"Final x (Layer 6): {final_x.mean().item():.4f}")
    
    # 4. Loss (Parity Target = PI)
    target = torch.ones_like(final_x[:, 0]) * 3.14159
    pred = final_x[:, 0]
    
    # 1 - cos(pred - target)
    loss = (1.0 - torch.cos(pred - target)).mean()
    print(f"Loss: {loss.item():.4f}")
    
    # 5. Backward
    loss.backward()
    
    # 6. Check Gradients
    print("\n--- Gradient Report ---")
    print(f"Force Grad Norm: {force.grad.norm().item() if force.grad is not None else 'None'}")
    print(f"x0 Grad Norm: {x.grad.norm().item() if x.grad is not None else 'None'}")
    print(f"Friction Gate Grad Norm: {geo.forget_gate.weight.grad.norm().item() if geo.forget_gate.weight.grad is not None else 'None'}")
    
    if torch.isnan(loss) or force.grad is None or torch.isnan(force.grad.norm()):
        print("!! FAILURE: Encounted NaN or Broken Graph !!")
    else:
        print(">> SUCCESS: Gradients flowing and stable. <<")

if __name__ == "__main__":
    debug_gradients()
