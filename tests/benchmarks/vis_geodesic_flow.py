import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path
from sklearn.decomposition import PCA

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

def plot_geodesic_flow(checkpoint_path, text="123 + 456 = 579"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🌀 Visualizing Geodesic Flow for: '{text}'")
    
    # 1. Setup
    vocab = "0123456789+-*= "
    token_to_id = {c: i for i, c in enumerate(vocab)}
    input_ids = torch.tensor([token_to_id[c] for c in text]).unsqueeze(0).to(device)
    
    model = Manifold(vocab_size=len(vocab), dim=512, depth=8, heads=8).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    model.eval()
    
    # 2. Extract Trajectory
    trajectory = []
    
    with torch.no_grad():
        x = model.x0.expand(1, -1)
        v = model.v0.expand(1, -1)
        forces = model.embedding(input_ids)
        
        for t in range(input_ids.size(1)):
            force = forces[:, t]
            
            # Record state
            trajectory.append(x.clone().cpu().numpy())
            
            # Step
            x, v = model.layers[0](x, v, force) # One layer or all?
            # For better flow, let's just use the first layer's evolution per token
    
    traj_data = np.concatenate(trajectory, axis=0) # [seq_len, dim]
    
    # 3. PCA to 3D
    pca = PCA(n_components=3)
    traj_3d = pca.fit_transform(traj_data)
    
    # 4. Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sns.set_style("white")
    
    # Plot line
    ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='#2A9D8F', linewidth=4, alpha=0.8)
    
    # Plot points with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(traj_3d)))
    ax.scatter(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], c=colors, s=100, edgecolors='black')
    
    # Annotate tokens
    for i, char in enumerate(text):
        ax.text(traj_3d[i, 0], traj_3d[i, 1], traj_3d[i, 2], char, fontsize=12, fontweight='bold')

    ax.set_title(f"Manifold v1.0: Geodesic Reasoning Trajectory\n'{text}'", fontsize=16, fontweight='bold')
    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")
    ax.set_zlabel("Latent PC3")
    
    # Save
    results_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "geodesic_flow_3d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Geodesic Flow visualization saved to: {results_dir}/geodesic_flow_3d.png")

if __name__ == "__main__":
    ckpt = "checkpoints/v1_0_math_oracle/epoch_1.pt"
    text = "123 + 456 = 579"
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    plot_geodesic_flow(ckpt, text)
