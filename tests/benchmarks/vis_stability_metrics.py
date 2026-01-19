import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam

def run_stability_test(steps=100):
    print(f"🔬 Running Gradient Norm & Energy Stability Test ({steps} steps)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Model
    model = Manifold(
        vocab_size=1000,
        dim=256,
        depth=6,
        heads=4,
        integrator_type='leapfrog', # Symplectic
        physics_config={
           'embedding': {'type': 'functional'},
           'active_inference': {'enabled': False} 
        }
    ).to(device)
    
    # Optimizer
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics
    grad_norms = []
    energy_history = []
    loss_history = []
    
    model.train()
    
    for step in range(steps):
        # Dummy Data
        inputs = torch.randint(0, 1000, (8, 32)).to(device)
        targets = torch.randint(0, 1000, (8, 32)).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass returning state
        logits, (x_final, v_final), _ = model(inputs)
        
        # Calculate Hamiltonian Estimate (Energy)
        # H = T + V approx 0.5 * v^2 (ignoring detailed potential for now)
        # We average over batch and sequence/dim
        kinetic_energy = 0.5 * torch.norm(v_final, dim=-1).pow(2).mean().item()
        energy_history.append(kinetic_energy)
        
        # Loss
        loss = criterion(logits.view(-1, 1000), targets.view(-1))
        loss_history.append(loss.item())
        
        loss.backward()
        
        # Calculate Gradient Norm (Global)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # Step
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={loss.item():.4f} | GradNorm={total_norm:.4f} | Energy={kinetic_energy:.4f}")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Gradient Norm
    axes[0].plot(grad_norms, color='#E76F51', linewidth=2)
    axes[0].set_title("Gradient Norm Stability", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Global L2 Norm")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Hamiltonian Energy
    axes[1].plot(energy_history, color='#2A9D8F', linewidth=2)
    axes[1].set_title("Hamiltonian Energy Conservation (Kinetic)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Energy (0.5 * v^2)")
    axes[1].grid(True, alpha=0.3)
    
    # 3. Loss
    axes[2].plot(loss_history, color='#264653', linewidth=2)
    axes[2].set_title("Training Loss", fontsize=14, fontweight='bold')
    axes[2].set_ylabel("Cross Entropy")
    axes[2].set_xlabel("Training Steps")
    axes[2].grid(True, alpha=0.3)
    
    output_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "stability"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"stability_metrics_{timestamp}.png"
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n✅ Test Complete. Plot saved to: {plot_path}")
    print(f"Final Grad Norm: {grad_norms[-1]:.4f}")
    print(f"Energy Variance: {np.var(energy_history):.6f}")

if __name__ == "__main__":
    run_stability_test()
