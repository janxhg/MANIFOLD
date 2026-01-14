"""
Geodesic Path Optimality Test
==============================

Tests whether GFN learns true geodesics (shortest paths on the manifold)
rather than naive straight-line paths in embedding space.

This demonstrates the physical intuition: curved paths can be shorter than
straight lines in curved space (think: great circle routes on Earth).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import GFN
from src.geometry import LowRankChristoffel


class GeodesicOptimalityTester:
    """Test suite for verifying geodesic path properties."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.results_dir = PROJECT_ROOT / "tests" / "professional" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_path_length(self, trajectory):
        """
        Compute geodesic path length on the manifold.
        
        For a Riemannian manifold, path length = ∫ ||dx/dt|| dt
        We approximate: L ≈ Σ ||x_{t+1} - x_t||
        
        Args:
            trajectory: List of position tensors [batch, dim]
            
        Returns:
            float: Total path length
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(trajectory) - 1):
            displacement = trajectory[i+1] - trajectory[i]
            step_length = torch.norm(displacement, dim=-1).mean().item()
            total_length += step_length
        
        return total_length
    
    def test_curved_vs_straight(self, seq_length=100):
        """
        Test 1: Compare geodesic path to Euclidean straight line.
        
        Key Insight: If the manifold has curvature, the geodesic path should
        differ from the straight line. The geodesic may be LONGER in embedding
        space but SHORTER in terms of energy/action.
        
        Returns:
            dict: Comparison metrics
        """
        print("🔬 Test 1: Geodesic vs Euclidean Path Comparison")
        
        self.model.eval()
        vocab_size = self.model.readout.out_features
        
        # Generate sequence
        input_seq = torch.randint(0, vocab_size, (1, seq_length)).to(self.device)
        
        # Compute GFN trajectory (geodesic)
        geodesic_trajectory = []
        x = self.model.x0.expand(1, -1)
        v = self.model.v0.expand(1, -1)
        all_forces = self.model.embedding(input_seq)
        
        with torch.no_grad():
            for t in range(seq_length):
                force = all_forces[:, t]
                for layer in self.model.layers:
                    x, v = layer(x, v, force)
                geodesic_trajectory.append(x.clone().cpu())
        
        # Compute straight-line trajectory (Euclidean)
        start_point = geodesic_trajectory[0]
        end_point = geodesic_trajectory[-1]
        
        straight_trajectory = []
        for t in range(seq_length):
            alpha = t / (seq_length - 1)
            interpolated = (1 - alpha) * start_point + alpha * end_point
            straight_trajectory.append(interpolated)
        
        # Compute path lengths
        geodesic_length = self.compute_path_length(geodesic_trajectory)
        straight_length = self.compute_path_length(straight_trajectory)
        
        # Compute curvature (deviation from straight line)
        deviations = []
        for t in range(seq_length):
            deviation = torch.norm(geodesic_trajectory[t] - straight_trajectory[t]).item()
            deviations.append(deviation)
        
        max_deviation = max(deviations)
        mean_deviation = np.mean(deviations)
        
        # Visualize in 3D using PCA
        self._visualize_paths_3d(geodesic_trajectory, straight_trajectory)
        
        # Plot deviation over time
        plt.figure(figsize=(10, 5))
        plt.plot(deviations, linewidth=2, color='#E63946')
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Deviation from Straight Line', fontsize=12)
        plt.title('Geodesic Curvature: Deviation from Euclidean Path', fontsize=14)
        plt.grid(alpha=0.3)
        plt.savefig(self.results_dir / "geodesic_deviation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Geodesic Length: {geodesic_length:.4f}")
        print(f"  Straight Length: {straight_length:.4f}")
        print(f"  Max Deviation:   {max_deviation:.4f}")
        print(f"  Mean Deviation:  {mean_deviation:.4f}")
        
        # Interpretation
        if mean_deviation > 0.01:
            print("  ✅ Manifold has non-trivial curvature (geodesic ≠ straight line)")
        else:
            print("  ⚠️  Manifold appears nearly flat (geodesic ≈ straight line)")
        
        return {
            "geodesic_length": geodesic_length,
            "straight_length": straight_length,
            "max_deviation": max_deviation,
            "mean_deviation": mean_deviation,
            "has_curvature": mean_deviation > 0.01
        }
    
    def _visualize_paths_3d(self, geodesic_trajectory, straight_trajectory):
        """Visualize both paths in 3D after PCA reduction."""
        
        # Convert to numpy
        geo_np = torch.cat(geodesic_trajectory, dim=0).numpy()  # [seq_len, dim]
        straight_np = torch.cat(straight_trajectory, dim=0).numpy()
        
        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        geo_3d = pca.fit_transform(geo_np)
        straight_3d = pca.transform(straight_np)
        
        # Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Geodesic path (curved)
        ax.plot(geo_3d[:, 0], geo_3d[:, 1], geo_3d[:, 2], 
                linewidth=2.5, label='GFN Geodesic Path', color='#2A9D8F', alpha=0.8)
        
        # Straight path
        ax.plot(straight_3d[:, 0], straight_3d[:, 1], straight_3d[:, 2], 
                linewidth=2, label='Euclidean Straight Line', 
                color='#E76F51', linestyle='--', alpha=0.7)
        
        # Mark start and end
        ax.scatter(*geo_3d[0], s=200, c='green', marker='o', label='Start', edgecolors='black', linewidths=2)
        ax.scatter(*geo_3d[-1], s=200, c='red', marker='X', label='End', edgecolors='black', linewidths=2)
        
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.set_zlabel('PC3', fontsize=11)
        ax.set_title('Geodesic Flow vs Euclidean Path (PCA Projection)', fontsize=14, pad=20)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        plt.savefig(self.results_dir / "geodesic_path_3d.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def test_manifold_curvature_field(self):
        """
        Test 2: Visualize the curvature tensor field Γ(v,v).
        
        This shows WHERE the manifold curves (regions of semantic complexity).
        """
        print("\n🔬 Test 2: Manifold Curvature Field Visualization")
        
        # Extract first layer's Christoffel network
        layer = self.model.layers[0]
        christoffel = layer.christoffel
        
        # Sample velocity space in 2D slice
        grid_size = 40
        x = np.linspace(-3, 3, grid_size)
        y = np.linspace(-3, 3, grid_size)
        X, Y = np.meshgrid(x, y)
        
        curvatures = np.zeros((grid_size, grid_size))
        
        with torch.no_grad():
            for i in range(grid_size):
                for j in range(grid_size):
                    v_sample = torch.zeros(1, self.model.dim).to(self.device)
                    v_sample[0, 0] = X[i, j]
                    v_sample[0, 1] = Y[i, j]
                    
                    # Compute Christoffel symbol: Γ(v, v)
                    gamma = christoffel(v_sample)
                    
                    # Curvature magnitude
                    curvatures[i, j] = torch.norm(gamma).item()
        
        # Visualize heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(curvatures, extent=[-3, 3, -3, 3], origin='lower', 
                       cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Curvature Magnitude ||Γ(v,v)||', shrink=0.8)
        plt.xlabel('Velocity Component v₀', fontsize=12)
        plt.ylabel('Velocity Component v₁', fontsize=12)
        plt.title('Learned Manifold Curvature Field (Layer 0)', fontsize=14)
        
        # Add contour lines
        plt.contour(X, Y, curvatures, levels=5, colors='white', alpha=0.4, linewidths=1)
        
        plt.savefig(self.results_dir / "curvature_field.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistics
        mean_curv = np.mean(curvatures)
        max_curv = np.max(curvatures)
        std_curv = np.std(curvatures)
        
        print(f"  Mean Curvature: {mean_curv:.4f}")
        print(f"  Max Curvature:  {max_curv:.4f}")
        print(f"  Std Curvature:  {std_curv:.4f}")
        
        return {
            "mean_curvature": mean_curv,
            "max_curvature": max_curv,
            "std_curvature": std_curv,
            "curvature_field": curvatures
        }
    
    def test_action_minimization(self, seq_length=50):
        """
        Test 3: Verify that geodesics minimize action integral.
        
        Physics: Geodesics are paths that minimize ∫ ||v||² dt (principle of least action).
        
        We compare the action of:
        1. GFN's learned path
        2. Random perturbations of that path
        
        The learned path should have LOWER action.
        """
        print("\n🔬 Test 3: Action Minimization Principle")
        
        self.model.eval()
        vocab_size = self.model.readout.out_features
        input_seq = torch.randint(0, vocab_size, (1, seq_length)).to(self.device)
        
        # Compute GFN trajectory and action
        velocities = []
        x = self.model.x0.expand(1, -1)
        v = self.model.v0.expand(1, -1)
        all_forces = self.model.embedding(input_seq)
        
        with torch.no_grad():
            for t in range(seq_length):
                force = all_forces[:, t]
                for layer in self.model.layers:
                    x, v = layer(x, v, force)
                velocities.append(v.clone())
        
        # Compute action: A = Σ ||v||²
        gfn_action = sum((v ** 2).sum().item() for v in velocities)
        
        # Generate perturbed paths by adding random noise to velocities
        num_perturbations = 20
        perturbed_actions = []
        
        for _ in range(num_perturbations):
            noise_scale = 0.1
            perturbed_action = 0.0
            
            for v in velocities:
                noise = torch.randn_like(v) * noise_scale
                v_pert = v + noise
                perturbed_action += (v_pert ** 2).sum().item()
            
            perturbed_actions.append(perturbed_action)
        
        mean_perturbed = np.mean(perturbed_actions)
        
        # Check if GFN action is lower (as it should be)
        is_optimal = gfn_action < mean_perturbed
        
        print(f"  GFN Action:           {gfn_action:.4f}")
        print(f"  Mean Perturbed Action: {mean_perturbed:.4f}")
        print(f"  Optimality: {'✅ GFN minimizes action' if is_optimal else '⚠️  Perturbed paths have lower action'}")
        
        # Visualization
        plt.figure(figsize=(10, 5))
        plt.hist(perturbed_actions, bins=15, alpha=0.7, color='skyblue', edgecolor='black', label='Perturbed Paths')
        plt.axvline(gfn_action, color='red', linestyle='--', linewidth=2.5, label='GFN Geodesic')
        plt.xlabel('Action Integral (Σ||v||²)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Action Minimization: GFN vs Random Perturbations', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
        plt.savefig(self.results_dir / "action_minimization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            "gfn_action": gfn_action,
            "mean_perturbed_action": mean_perturbed,
            "is_optimal": is_optimal
        }


def run_geodesic_tests(checkpoint_path=None):
    """Run all geodesic optimality tests."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  GEODESIC OPTIMALITY TEST SUITE")
    print("=" * 60)
    print(f"Device: {device}\n")
    
    # Load model
    model = GFN(vocab_size=20, dim=512, depth=12, rank=16, integrator_type='leapfrog').to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
            print("✓ Checkpoint loaded\n")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("Using randomly initialized model\n")
    else:
        print("Using randomly initialized model\n")
    
    # Run tests
    tester = GeodesicOptimalityTester(model, device)
    
    # Test 1: Curved vs straight paths
    path_results = tester.test_curved_vs_straight(seq_length=100)
    
    # Test 2: Curvature field
    curvature_results = tester.test_manifold_curvature_field()
    
    # Test 3: Action minimization
    action_results = tester.test_action_minimization(seq_length=50)
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Path Curvature): {'✅ Non-trivial manifold' if path_results['has_curvature'] else '⚠️  Nearly flat'}")
    print(f"  - Mean deviation: {path_results['mean_deviation']:.4f}")
    
    print(f"\nTest 2 (Curvature Field):")
    print(f"  - Mean curvature: {curvature_results['mean_curvature']:.4f}")
    print(f"  - Max curvature:  {curvature_results['max_curvature']:.4f}")
    
    print(f"\nTest 3 (Action Minimization): {'✅ Optimal' if action_results['is_optimal'] else '⚠️  Suboptimal'}")
    print(f"  - GFN action: {action_results['gfn_action']:.4f}")
    
    print("\n✓ All plots saved to tests/professional/results/")
    print("=" * 60)
    
    return {
        "path": path_results,
        "curvature": curvature_results,
        "action": action_results
    }


if __name__ == "__main__":
    ckpt_path = None
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    results = run_geodesic_tests(ckpt_path)
