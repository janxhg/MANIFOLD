
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from gfn.model import Manifold

class ParityTask:
    def __init__(self, vocab_size=2, length=20, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y_int = torch.cumsum(x, dim=1) % self.mod
        PI = 3.14159265359
        y = y_int.float() * PI
        return x, y

def run_trial(name, config, steps=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 128
    
    # Unpack config
    impulse = config.get('impulse', 10.0)
    curvature = config.get('curvature', 0.05)
    lr = config.get('lr', 1e-3)
    integrator = config.get('integrator', 'rk4')
    
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {'enabled': False},
        'singularities': {'enabled': False},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.3}
    }
    
    model = Manifold(
        vocab_size=2, dim=dim, depth=6, heads=1, 
        integrator_type=integrator,
        physics_config=physics_config,
        holographic=True
    ).to(device)
    
    # Apply Config Overrides
    if hasattr(model.embedding, 'impulse_scale'):
        model.embedding.impulse_scale = impulse
        
    # Hack to set curvature scale globally (since it's hardcoded in toroidal.py for now, we rely on the file state)
    # But wait, we can't easily change the hardcoded value in toroidal.py from here without reloading.
    # We will assume the file is at 0.05 (current state) and maybe scale down via Christoffel if possible?
    # Actually, we can check if ToroidalChristoffel is used and set a scaling attribute if we added one. 
    # Current toroidal.py implementation doesn't expose a dynamic scale attribute easily. 
    # We will proceed with Impulse/LR/Integrator grid first.
    
    # Initialize Friction
    for layer in model.layers:
         if hasattr(layer.christoffels[0], 'forget_gate'):
             nn.init.constant_(layer.christoffels[0].forget_gate.bias, 1.2)
             
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    losses = []
    accs = []
    
    task = ParityTask(length=20)
    
    # Fixed Batch for Stability Check
    x_static, y_static = task.generate_batch(128, device=device)
    
    for i in range(steps):
        optimizer.zero_grad()
        output = model(x_static, collect_christ=False)
        
        if isinstance(output, tuple):
             x_pred = output[0][:, :, 0]
        else:
             x_pred = output[:, :, 0]
             
        y = y_static.float()
        loss = (1.0 - torch.cos(x_pred - y)).mean()
        
        if torch.isnan(loss):
            return {'name': name, 'status': 'NaN', 'final_loss': np.nan, 'final_acc': 0.0}
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Acc
        with torch.no_grad():
            diff = torch.abs(x_pred - y)
            PI = 3.14159265359
            diff = torch.min(diff, 2*PI - diff)
            acc = (diff < 0.5).float().mean().item()
            
        losses.append(loss.item())
        accs.append(acc)
        
    # Analyze
    loss_std = np.std(losses[-50:]) # Volatility of last 50 steps
    final_loss = np.mean(losses[-10:])
    final_acc = np.mean(accs[-10:])
    
    return {
        'name': name,
        'status': 'OK',
        'final_loss': final_loss,
        'final_acc': final_acc,
        'volatility': loss_std
    }

def run_diagnostics():
    configs = [
        {'name': 'Baseline (Current)', 'impulse': 25.0, 'lr': 1e-3, 'integrator': 'rk4'},
        {'name': 'Low Impulse', 'impulse': 10.0, 'lr': 1e-3, 'integrator': 'rk4'},
        {'name': 'High Impulse', 'impulse': 50.0, 'lr': 1e-3, 'integrator': 'rk4'},
        {'name': 'Low LR', 'impulse': 25.0, 'lr': 1e-4, 'integrator': 'rk4'},
        {'name': 'Leapfrog (Check)', 'impulse': 25.0, 'lr': 1e-3, 'integrator': 'leapfrog'},
    ]
    
    print(f"{'Experiment':<25} | {'Status':<5} | {'Loss':<8} | {'Acc':<8} | {'Volat.':<8}")
    print("-" * 65)
    
    for cfg in configs:
        res = run_trial(cfg['name'], cfg)
        print(f"{res['name']:<25} | {res['status']:<5} | {res['final_loss']:.4f}   | {res['final_acc']*100:.1f}%   | {res['volatility']:.4f}")

if __name__ == "__main__":
    run_diagnostics()
