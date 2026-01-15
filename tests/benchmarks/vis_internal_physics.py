import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

def analyze_model_internals(checkpoint_path, input_text="999 + 1 = 1000"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔬 Analyzing Manifold Internals (X-Ray Mode)...")
    print(f"Input: '{input_text}'")
    
    # 1. Setup Model (Assuming Math Oracle vocabulary for now)
    # Mapping: '0'-'9', '+', '-', '*', '=', ' '
    vocab = "0123456789+-*= "
    token_to_id = {c: i for i, c in enumerate(vocab)}
    id_to_token = {i: c for c, i in token_to_id.items()}
    
    # Simple tokenizer
    input_ids = torch.tensor([token_to_id[c] for c in input_text]).unsqueeze(0).to(device)
    
    # Use v1.0 Golden Config dimensions or load from ckpt if available
    # For visualization, we can use a standard 512-dim model
    model = Manifold(vocab_size=len(vocab), dim=512, depth=8, heads=8).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        print("!! WARNING: No checkpoint found. Using random weights (Structural Visualization).")

    model.eval()
    
    # 2. Capture Metrics
    # We'll run step by step to capture internal state
    seq_len = input_ids.size(1)
    curvature_history = []
    hamiltonian_history = []
    fractal_activity = []
    
    with torch.no_grad():
        # First-pass to get embedding forces
        forces = model.embedding(input_ids)
        
        x = model.x0.expand(1, -1)
        v = model.v0.expand(1, -1)
        
        for t in range(seq_len):
            force = forces[:, t]
            
            step_curvature = 0.0
            step_hamiltonian = 0.0
            step_fractal = 0.0
            
            # Layer-by-layer evolution to capture internals
            curr_x, curr_v = x, v
            for layer_idx, layer in enumerate(model.layers):
                # We need to peek into the layer forward
                # For this tool, we'll re-calculate the metrics manually 
                # to avoid changing the core library code too much
                
                # Estimate Curvature (L2 norm of Christoffel output)
                # In Manifold v1.0, layers are FractalMLayers
                # Macro Christoffel
                macro_gamma = 0.0
                for c in layer.macro_manifold.christoffels:
                    macro_gamma += torch.norm(c(curr_v)).item()
                
                # Check for Fractal Tunneling (Simplified check)
                is_tunneling = False
                if hasattr(layer, 'macro_manifold'):
                    # Curvature estimate same as in FractalMLayer
                    v_norm = torch.norm(curr_v, dim=-1, keepdim=True)
                    rel_v = curr_v / (v_norm + 1e-6)
                    # Use first head as proxy
                    gamma = layer.macro_manifold.christoffels[0](rel_v)
                    curv = torch.norm(gamma, dim=-1)
                    if curv.mean() > layer.threshold:
                        is_tunneling = True
                        step_fractal += 1.0
                
                # Symplectic Update
                curr_x, curr_v = layer(curr_x, curr_v, force)
                
                # Hamiltonian Estimate (H = Kinetic + Potential)
                # K = 0.5 * v^2, P = -grad(phi) which is related to force
                # Here we use the Geodesic Deviation as a proxy for Energy
                ke = 0.5 * torch.norm(curr_v)**2
                step_hamiltonian += ke.item()
                step_curvature += macro_gamma
                
            curvature_history.append(step_curvature / len(model.layers))
            hamiltonian_history.append(step_hamiltonian / len(model.layers))
            fractal_activity.append(step_fractal)
            
            x, v = curr_x, curr_v

    # 3. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    sns.set_style("whitegrid")
    
    tokens = [id_to_token[i.item()] for i in input_ids[0]]
    x_ticks = np.arange(len(tokens))
    
    # Plot 1: Curvature (The "Difficulty" Meter)
    axes[0].plot(curvature_history, color='#E76F51', linewidth=3, marker='o', label="Semantic Curvature (Γ)")
    axes[0].set_ylabel("Curvature Magnitude", fontsize=12)
    axes[0].set_title("Cognitive Physics: Internal Difficulty Map", fontsize=14, fontweight='bold')
    axes[0].fill_between(x_ticks, 0, curvature_history, color='#E76F51', alpha=0.2)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Hamiltonian (The "Stability" Meter)
    axes[1].plot(hamiltonian_history, color='#2A9D8F', linewidth=3, marker='s', label="Hamiltonian Energy (H)")
    axes[1].set_ylabel("System Energy", fontsize=12)
    axes[1].set_title("System Stability: Hamiltonian Evolution", fontsize=14, fontweight='bold')
    axes[1].fill_between(x_ticks, 0, hamiltonian_history, color='#2A9D8F', alpha=0.2)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Fractal Activity (The "Zoom" Meter)
    axes[2].bar(x_ticks, fractal_activity, color='#264653', alpha=0.8, label="Fractal Tunneling (Active Layers)")
    axes[2].set_ylabel("Active Sub-Layers", fontsize=12)
    axes[2].set_title("Recursive Resolution: Fractal Activity", fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(tokens, fontsize=12)
    axes[2].set_xlabel("Input Sequence", fontsize=14)
    
    plt.tight_layout()
    
    # Save Result
    output_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "internal_physics_xray.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ X-Ray analysis complete!")
    print(f"Report saved to: {out_path}")
    print(f"Visual evidence of 'Carry Thinking': Look for Curvature peaks at logic transitions (e.g. '+', '=', or carries).")

if __name__ == "__main__":
    ckpt = "checkpoints/v1_0_math_oracle/epoch_1.pt" # Default path
    text = "999 + 1 = 1000"
    
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    if len(sys.argv) > 2:
        text = sys.argv[2]
        
    analyze_model_internals(ckpt, text)
