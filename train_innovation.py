import torch
import torch.nn as nn
import torch.optim as optim
import time
from gfn.model import Manifold
from gfn.cuda.ops import CUDA_AVAILABLE

def generate_parity_data(batch_size, seq_len, dim):
    """
    Cumulative Parity Task: y_t = sum(x_0...x_t) mod 2
    """
    x = torch.randint(0, 2, (batch_size, seq_len))
    y = torch.cumsum(x, dim=1) % 2
    return x, y

def train_innovation():
    print("="*60)
    print("GFN INNOVATION BENCHMARK: HOLOGRAPHIC + FRACTAL + SINGULARITY")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configure the Innovation Stack
    physics_cfg = {
        'topology': {
            'type': 'torus', # Topological prior for cyclic logic (Parity)
            'major_radius': 2.0,
            'minor_radius': 1.0
        },
        'active_inference': {
            'enabled': True,
            'reactive_curvature': {
                'enabled': True,
                'plasticity': 0.2
            },
            'singularities': {
                'enabled': True,
                'threshold': 0.85, # Trigger black hole when confident
                'strength': 15.0
            }
        },
        'fractal': {
            'enabled': True,
            'threshold': 0.4,
            'alpha': 0.3
        },
        'holographic': True # Zero-Shot Geometric Readout (Identity)
    }
    
    dim = 64
    batch_size = 32
    seq_len = 50
    
    # The vocab size is 2 (0 and 1)
    # With holographic=True, the state itself is the prediction.
    # For a torus, theta=0 means 0, theta=pi means 1.
    model = Manifold(
        vocab_size=2,
        dim=dim,
        depth=2,
        rank=16,
        heads=2,
        integrator_type='symplectic',
        physics_config=physics_cfg,
        holographic=True
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    print(f"\n[INFO] Starting training on {device}...")
    print(f"[INFO] Holographic: {model.holographic}")
    
    for step in range(101):
        x, y = generate_parity_data(batch_size, seq_len, dim)
        x, y = x.to(device), y.to(device)
        
        # In holographic mode, we want the first channel of x to represent the parity
        # We need to map y (0, 1) to the target coordinates (0, pi)
        target_coords = y.float() * 3.14159265
        
        optimizer.zero_grad()
        
        # Forward pass
        # With holographic identity readout, logits = x_final
        # x_final shape: [batch, seq_len, dim]
        logits, _, _ = model(x)
        
        # We only care about the first dimension for parity readout in this test
        # (Assuming the model learns to use the first head/dimension)
        # Actually, if we use Torus, the index i is theta.
        prediction = logits[:, :, 0] # theta_0
        
        # L2 loss in periodic space: 1 - cos(pred - target)
        loss = (1.0 - torch.cos(prediction - target_coords)).mean()
        
        loss.backward()
        
        # Gradient clipping for stability in adaptive manifolds
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 10 == 0:
            with torch.no_grad():
                # Accuracy: how many are within pi/4 of the target
                dist = (prediction - target_coords).abs() % (2 * 3.14159)
                dist = torch.min(dist, 2 * 3.14159 - dist)
                acc = (dist < (3.14159 / 4)).float().mean()
            
            print(f"Step {step:3d} | Loss: {loss.item():.6f} | Accuracy: {acc.item()*100:6.2f}%")
            
    print("\n" + "="*60)
    if acc > 0.9:
        print("✓ INNOVATION VALIDATED: The Hyper-Torus learned the logic holographically.")
    else:
        print("✗ VALIDATION INCOMPLETE: Manifold did not converge in time.")
    print("="*60)

if __name__ == "__main__":
    train_innovation()
