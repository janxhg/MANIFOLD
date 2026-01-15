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

def visualize_fractal_zoom(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("💠 Visualizing Fractal Tunneling (Recursive Zoom)...")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    model = Manifold(vocab_size=len(vocab), dim=512, depth=1, heads=1).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    layer = model.layers[0]
    # In FractalMLayer, we have macro_manifold and micro_manifold
    macro = layer.macro_manifold
    micro = layer.micro_manifold
    
    # 2. Generate Grid
    # We'll look at a small window to see "Recursive Self-Similarity"
    grid_size = 40
    
    # Macro scale: [-2, 2]
    # Micro scale: [-0.1, 0.1] (Zoomed in 20x)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_style("white")
    
    with torch.no_grad():
        # Plot Macro
        x_m = np.linspace(-2, 2, grid_size)
        y_m = np.linspace(-2, 2, grid_size)
        X_m, Y_m = np.meshgrid(x_m, y_m)
        mag_m = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                v = torch.zeros(1, 512).to(device)
                v[0, 0] = X_m[i, j]
                v[0, 1] = Y_m[i, j]
                gamma = macro.christoffels[0](v)
                mag_m[i, j] = torch.norm(gamma).item()
        
        axes[0].imshow(mag_m, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
        axes[0].set_title("Macro-Manifold (Global Context)", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("v_0")
        axes[0].set_ylabel("v_1")
        
        # Plot Micro
        x_z = np.linspace(-0.2, 0.2, grid_size)
        y_z = np.linspace(-0.2, 0.2, grid_size)
        X_z, Y_z = np.meshgrid(x_z, y_z)
        mag_z = np.zeros((grid_size, grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                v = torch.zeros(1, 512).to(device)
                v[0, 0] = X_z[i, j]
                v[0, 1] = Y_z[i, j]
                # In FractalMLayer, micro is triggered when macro curvature is high
                # Here we just visualize the micro manifold structure directly
                gamma = micro.christoffels[0](v)
                mag_z[i, j] = torch.norm(gamma).item()
        
        axes[1].imshow(mag_z, extent=[-0.2, 0.2, -0.2, 0.2], origin='lower', cmap='plasma')
        axes[1].set_title("Micro-Manifold (Local Resolution Zoom)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("v_0 (Zoomed)")
        axes[1].set_ylabel("v_1 (Zoomed)")

    plt.suptitle("Fractal Manifolds: Recursive Tunneling & Zoom\n(Nested Geometric Resolution for Complex Logic)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "fractal_zoom_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Fractal Zoom comparison saved to: {results_dir}/fractal_zoom_comparison.png")

if __name__ == "__main__":
    ckpt = "checkpoints/v0.3/epoch_0.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    visualize_fractal_zoom(ckpt)
