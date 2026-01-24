"""
Thermodynamic Gating (Clutch) Visualizer
========================================
Visualizes the 'Dash-and-Stop' friction mechanics.
Shows how the 'Clutch' ($\mu$) engages to capture info and disengages to store it.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.bench_utils import ResultsLogger, ParityTask

def plot_clutch_mechanics(frictions, inputs, logger):
    """
    Plots Friction Coefficient overlayed with Input Impulses.
    """
    time = np.arange(len(frictions))
    
    plt.figure(figsize=(15, 6))
    
    # 1. Friction (The Clutch)
    plt.plot(time, frictions, 'r-', linewidth=2.5, label='Friction Coeff ($\mu$)')
    plt.fill_between(time, 0, frictions, color='r', alpha=0.1)
    
    # 2. Inputs (The Driver)
    # inputs: [1, L]
    in_vals = inputs[0].cpu().numpy()
    # Normalize inputs for plotting
    plt.bar(time, in_vals * np.max(frictions), color='k', alpha=0.3, width=0.3, label='Input Token')
    
    plt.title("Thermodynamic Gating: The Clutch Mechanism", fontsize=16, fontweight='bold')
    plt.xlabel("Time Step")
    plt.ylabel("Dissipation Strength")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate Regimes
    # High Mu -> Writing
    # Low Mu -> Remembering
    
    logger.save_plot(plt.gcf(), "clutch_dynamics.png")
    plt.close()

def run_friction_viz():
    logger = ResultsLogger("friction", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("⚙️ Analyzing Thermodynamic Gating...")
    
    # 1. Config
    # We need a model that relies on friction gates (Torus default)
    physics_config = {
        'embedding': {'type': 'functional', 'coord_dim': 16}, 
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.2} 
        # Friction Gates are enabled by default in Toroidal manifold
    }
    
    model = Manifold(vocab_size=2, dim=128, depth=2, physics_config=physics_config).to(device)
    model.eval()
    
    # 2. Sequence
    L = 40
    task = ParityTask(length=L)
    x, _, _ = task.generate_batch(1, device=device)
    
    # 3. Simulate and Capture Friction
    frictions = []
    state = None
    
    print("  [*] Tracking Clutch Engagement...")
    with torch.no_grad():
        for t in range(L):
            input_t = x[:, t:t+1]
            
            # We want to measure the friction gate activation.
            # Ideally we'd hook the module. 
            # Proxy: We measure v_decay. 
            # v_next = v + ... - mu*v
            # If v drops sharply while force is high, mu is high.
            
            # Better Proxy: model.layers[0].macro_manifold.christoffels[0].forget_gate(x)
            # We'll calculate it manually for Viz accuracy.
            
            # 1. Get State PRE-update
            # But we only have state POST-update from previous step.
            x_prev = state[0] if state else torch.zeros(1, 128, device=device)
            
            # Manual Gate Calc (Head 0)
            layer = model.layers[0].macro_manifold
            christ = layer.christoffels[0]
            
            # x -> sin/cos
            scale = 1.0 # Assuming head_dim scale
            # We are just grabbing global x for roughness
            head_dim = 128 // 4
            x_head = x_prev[:, :head_dim] 
            
            x_in = torch.cat([torch.sin(x_head), torch.cos(x_head)], dim=-1)
            gate_activ = christ.forget_gate(x_in) 
            
            # Add input effect? 
            # In code: gate_activ += input_gate(force)
            # We skip force calc for simplicity, just show state-dependent friction component
            
            mu = torch.sigmoid(gate_activ.mean()).item() * 10.0
            frictions.append(mu)
            
            # Step model
            out = model(input_t, state=state)
            state = out[1] if isinstance(out, tuple) else None

    print("  [+] Clutch profile captured.")
    plot_clutch_mechanics(frictions, x, logger)

if __name__ == "__main__":
    run_friction_viz()
