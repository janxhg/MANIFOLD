
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src import GFN

def visualize_curvature(checkpoint_path):
    device = torch.device('cpu') # Visualization on CPU is enough
    
    print("Generating Manifold Visualization...")
    model = GFN(vocab_size=20, dim=512, depth=12, rank=16).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
             model.load_state_dict(ckpt['model_state_dict'])
        except:
             print("Warning: Weight mismatch, using random weights.")
    
    # Extract First Layer's Christoffel Network
    # We want to see the curvature field Gamma(v, v)
    # v is High Dim (512), so we take a 2D slice
    
    layer = model.layers[0]
    christoffel = layer.christoffel
    
    # Generate 2D grid in velocity space
    # Slice dimensions 0 and 1
    grid_size = 50
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    magnitudes = np.zeros((grid_size, grid_size))
    
    with torch.no_grad():
        for i in range(grid_size):
            for j in range(grid_size):
                v_sample = torch.zeros(1, 512)
                v_sample[0, 0] = X[i, j]
                v_sample[0, 1] = Y[i, j]
                
                # Compute Gamma(v, v)
                gamma = christoffel(v_sample)
                
                # Store L2 norm of the curvature vector (Force magnitude)
                magnitudes[i, j] = torch.norm(gamma).item()
                
    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(magnitudes, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
    plt.colorbar(label='Curvature Magnitude ||Γ(v,v)||')
    plt.title("Learned Manifold Curvature (Layer 0, Dim 0-1 Slice)", fontsize=14)
    plt.xlabel("Velocity Component v_0")
    plt.ylabel("Velocity Component v_1")
    
    os.makedirs("tests/professional/results", exist_ok=True)
    plt.savefig("tests/professional/results/vis_manifold.png")
    plt.close()
    print("Manifold Visualization Saved.")

if __name__ == "__main__":
    ckpt = "checkpoints/medium_fast/epoch_15.pt"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    visualize_curvature(ckpt)
