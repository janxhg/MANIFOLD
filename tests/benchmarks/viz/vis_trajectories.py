"""
Professional Phase Space Visualizer (Toroidal Topology)
=======================================================
Visualizes the Hamiltonian dynamics of the GFN Hyper-Torus.
Plots (Theta vs Phi) and (Position vs Velocity) poincaré sections.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Ensure Project Root is in Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, ParityTask

def plot_phase_portrait(history, logger, step_name):
    """
    Plots the phase space trajectory onto a 2D Torus surface (unrolled).
    """
    # history: [Batch, Time, D]
    x_traj = history['x'] # [B, T, D]
    v_traj = history['v'] # [B, T, D]
    
    # Project to 2D for visualization (First 2 dimensions: Theta, Phi)
    theta = x_traj[0, :, 0].cpu().numpy()
    phi   = x_traj[0, :, 1].cpu().numpy()
    v_theta = v_traj[0, :, 0].cpu().numpy()
    
    # 1. Toroidal Configuration Space (Theta vs Phi)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Unroll toroidal coordinates [0, 2pi]
    ax1.plot(theta % (2*np.pi), phi % (2*np.pi), 'b-', linewidth=1.5, alpha=0.8, label='Particle Orbit')
    ax1.scatter(theta[0] % (2*np.pi), phi[0] % (2*np.pi), c='g', s=100, marker='o', label='Start')
    ax1.scatter(theta[-1] % (2*np.pi), phi[-1] % (2*np.pi), c='r', s=100, marker='x', label='End')
    ax1.set_title(f"Configuration Space (Torus Surface) - {step_name}",fontsize=14, fontweight='bold')
    ax1.set_xlabel(r"$\theta$ (Dim 0)")
    ax1.set_ylabel(r"$\phi$ (Dim 1)")
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(0, 2*np.pi)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Phase Space Section (Theta vs v_Theta)
    # Energy E ~ v^2. Stable orbits form closed loops.
    points = np.array([theta, v_theta]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color by time
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(0, len(theta))
    
    # Efficient collection plotting
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(len(theta)))
    lc.set_linewidth(1.5)
    
    ax2.add_collection(lc)
    ax2.set_xlim(theta.min()-0.5, theta.max()+0.5)
    ax2.set_ylim(v_theta.min()-1.0, v_theta.max()+1.0)
    
    ax2.set_title(f"Poincaré Section (Theta vs Momentum) - {step_name}", fontsize=14, fontweight='bold')
    ax2.set_xlabel(r"Position $\theta$")
    ax2.set_ylabel(r"Momentum $p_\theta$")
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    cbar = fig.colorbar(lc, ax=ax2)
    cbar.set_label("Time Step")
    
    plt.tight_layout()
    logger.save_plot(fig, f"trajectory_phase_{step_name}.png")
    plt.close(fig)

def run_trajectory_analysis():
    logger = ResultsLogger("trajectories", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🌌 Initializing Hyper-Torus Phase Space Analysis...")
    
    # 1. Config (Matching Superiority Paper)
    dim = 128
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.2}
    }
    
    model = Manifold(
        vocab_size=2, dim=dim, depth=6, heads=4, 
        physics_config=physics_config, impulse_scale=80.0
    ).to(device)
    model.eval()
    
    # 2. Generate Data (Parity Sequence)
    # L=100 ensures enough orbits to see the limit cycle
    L = 100 
    task = ParityTask(length=L)
    x, y_class, y_angle = task.generate_batch(1, device=device)
    
    print(f"  [*] Simulating Hamiltonian Flow (L={L})...")
    
    # 3. Trace Trajectory
    # We need to capture internal state at each step
    history = {'x': [], 'v': []}
    state = None
    
    with torch.no_grad():
        for t in range(L):
            input_t = x[:, t:t+1]
            out = model(input_t, state=state)
            
            # Manifold returns (val, state, ...) or ((x,v), state, ...)
            # Check return signature
            # Assuming Holographic/Explicit return:
            # Output is typically [Batch, 1, Dim] for x
            
            # Let's inspect the state returned
            # state is (x, v) tuples usually
            if isinstance(out, tuple):
                # out[0] -> readout/x_next
                # out[1] -> state (x, v) next
                state = out[1]
                
            x_curr, v_curr = state
            history['x'].append(x_curr)
            history['v'].append(v_curr)
            
    # Stack
    history['x'] = torch.cat(history['x'], dim=1) # [1, L, D]
    history['v'] = torch.cat(history['v'], dim=1)
    
    # 4. Viz
    plot_phase_portrait(history, logger, "parity_orbit")
    print("  [+] Phase portrait saved.")

if __name__ == "__main__":
    run_trajectory_analysis()
