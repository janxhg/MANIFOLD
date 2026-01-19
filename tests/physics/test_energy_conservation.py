"""
Energy Conservation Test
=========================

Validates GFN's core innovation: Hamiltonian-constrained dynamics that prevent
gradient explosion by enforcing energy conservation laws.

This test provides scientific proof that GFN respects physics-informed constraints.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import GFN, INTEGRATORS
from gfn.geometry import LowRankChristoffel


class EnergyConservationTester:
    """Professional-grade energy conservation verification."""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.results_dir = PROJECT_ROOT / "tests" / "professional" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def test_long_sequence_drift(self, seq_length=1000, tolerance=0.01):
        """
        Test 1: Verify energy drift is minimal over long sequences.
        
        Physics Principle: Hamiltonian systems conserve energy. If energy
        drifts significantly, the integrator is unstable.
        
        Args:
            seq_length: Length of test sequence
            tolerance: Max allowed relative energy drift (default 1%)
            
        Returns:
            dict: Metrics including drift percentage and stability score
        """
        print(f"🔬 Test 1: Long Sequence Energy Drift (T={seq_length})")
        
        self.model.eval()
        
        # Generate random sequence
        vocab_size = self.model.readout.out_features
        input_seq = torch.randint(0, vocab_size, (1, seq_length)).to(self.device)
        
        # Track energy at each timestep
        energies = []
        velocities_list = []
        
        # Initialize state
        x = self.model.x0.expand(1, -1)
        v = self.model.v0.expand(1, -1)
        
        # Pre-compute forces
        all_forces = self.model.embedding(input_seq)  # [1, seq_len, dim]
        
        with torch.no_grad():
            for t in range(seq_length):
                force = all_forces[:, t]
                
                # Evolve through layers
                for layer in self.model.layers:
                    x, v = layer(x, v, force)
                
                # Compute kinetic energy: E = ||v||²
                energy = (v ** 2).sum().item()
                energies.append(energy)
                velocities_list.append(v.clone())
        
        # Analyze drift
        energies = np.array(energies)
        initial_energy = energies[0]
        final_energy = energies[-1]
        
        # Relative drift
        relative_drift = abs(final_energy - initial_energy) / (initial_energy + 1e-8)
        
        # Max deviation from initial
        max_deviation = np.max(np.abs(energies - initial_energy)) / (initial_energy + 1e-8)
        
        # Stability score (1.0 = perfect, 0.0 = unstable)
        stability_score = max(0.0, 1.0 - relative_drift / tolerance)
        
        # Visualization
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Energy over time
        plt.subplot(1, 2, 1)
        plt.plot(energies, linewidth=1.5, color='#2E86AB')
        plt.axhline(y=initial_energy, color='r', linestyle='--', label='Initial Energy', alpha=0.7)
        plt.fill_between(
            range(len(energies)),
            initial_energy * (1 - tolerance),
            initial_energy * (1 + tolerance),
            alpha=0.2, color='green', label=f'±{tolerance*100:.0f}% Tolerance'
        )
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Energy ||v||²', fontsize=12)
        plt.title(f'Hamiltonian Energy Conservation (Drift: {relative_drift*100:.2f}%)', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Plot 2: Relative deviation
        plt.subplot(1, 2, 2)
        rel_deviations = (energies - initial_energy) / (initial_energy + 1e-8) * 100
        plt.plot(rel_deviations, linewidth=1.5, color='#A23B72')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=tolerance*100, color='r', linestyle='--', alpha=0.5, label=f'Tolerance: ±{tolerance*100:.0f}%')
        plt.axhline(y=-tolerance*100, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Relative Deviation (%)', fontsize=12)
        plt.title('Energy Stability Analysis', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "energy_conservation_long_sequence.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Pass/Fail
        passed = relative_drift < tolerance
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"  Initial Energy: {initial_energy:.4f}")
        print(f"  Final Energy:   {final_energy:.4f}")
        print(f"  Relative Drift: {relative_drift*100:.3f}%")
        print(f"  Max Deviation:  {max_deviation*100:.3f}%")
        print(f"  Status: {status}")
        
        return {
            "passed": passed,
            "relative_drift": relative_drift,
            "max_deviation": max_deviation,
            "stability_score": stability_score,
            "initial_energy": initial_energy,
            "final_energy": final_energy
        }
    
    def test_integrator_comparison(self, seq_length=500):
        """
        Test 2: Compare stability of different integrators.
        
        Hypothesis: Symplectic integrators (Leapfrog) should preserve energy
        better than naive methods (Euler).
        
        Returns:
            dict: Comparative metrics for each integrator
        """
        print(f"\n🔬 Test 2: Integrator Stability Comparison")
        
        results = {}
        vocab_size = 20
        dim = 256
        
        # Test each integrator
        integrator_types = ['heun', 'rk4', 'leapfrog']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, int_type in enumerate(integrator_types):
            print(f"  Testing: {int_type.upper()}")
            
            # Create model with specific integrator
            model = GFN(vocab_size=vocab_size, dim=dim, depth=6, rank=16, integrator_type=int_type).to(self.device)
            model.eval()
            
            # Generate sequence
            input_seq = torch.randint(0, vocab_size, (1, seq_length)).to(self.device)
            
            # Track energies
            energies = []
            x = model.x0.expand(1, -1)
            v = model.v0.expand(1, -1)
            all_forces = model.embedding(input_seq)
            
            with torch.no_grad():
                for t in range(seq_length):
                    force = all_forces[:, t]
                    for layer in model.layers:
                        x, v = layer(x, v, force)
                    energy = (v ** 2).sum().item()
                    energies.append(energy)
            
            energies = np.array(energies)
            initial = energies[0]
            drift = abs(energies[-1] - initial) / (initial + 1e-8)
            
            # Store results
            results[int_type] = {
                "drift": drift,
                "energies": energies,
                "mean_energy": np.mean(energies),
                "std_energy": np.std(energies)
            }
            
            # Plot
            ax = axes[idx]
            ax.plot(energies, linewidth=1.5, label=f'{int_type.upper()}')
            ax.axhline(y=initial, color='r', linestyle='--', alpha=0.5, label='Initial')
            ax.set_xlabel('Timestep', fontsize=11)
            ax.set_ylabel('Energy ||v||²', fontsize=11)
            ax.set_title(f'{int_type.upper()}\nDrift: {drift*100:.2f}%', fontsize=13)
            ax.legend()
            ax.grid(alpha=0.3)
            
            print(f"    Drift: {drift*100:.3f}%")
        
        plt.suptitle('Integrator Stability Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.results_dir / "integrator_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return results
    
    def test_adversarial_stability(self, num_trials=10):
        """
        Test 3: Stress test with adversarial inputs designed to trigger instability.
        
        Adversarial patterns:
        - Repeated same token (constant force)
        - High-frequency alternating tokens
        - Random noise
        
        Returns:
            dict: Stability metrics under adversarial conditions
        """
        print(f"\n🔬 Test 3: Adversarial Stability Test")
        
        self.model.eval()
        vocab_size = self.model.readout.out_features
        
        adversarial_patterns = {
            "constant": lambda: torch.zeros(1, 200, dtype=torch.long).to(self.device),  # All zeros
            "alternating": lambda: torch.tensor([[i % 2 for i in range(200)]]).to(self.device),
            "random": lambda: torch.randint(0, vocab_size, (1, 200)).to(self.device)
        }
        
        results = {}
        
        for pattern_name, pattern_fn in adversarial_patterns.items():
            print(f"  Testing: {pattern_name}")
            
            nan_count = 0
            max_energy_spike = 0
            
            for trial in range(num_trials):
                input_seq = pattern_fn()
                
                x = self.model.x0.expand(1, -1)
                v = self.model.v0.expand(1, -1)
                all_forces = self.model.embedding(input_seq)
                
                energies = []
                has_nan = False
                
                with torch.no_grad():
                    for t in range(input_seq.size(1)):
                        force = all_forces[:, t]
                        for layer in self.model.layers:
                            x, v = layer(x, v, force)
                        
                        energy = (v ** 2).sum().item()
                        
                        if torch.isnan(v).any() or torch.isinf(v).any():
                            has_nan = True
                            nan_count += 1
                            break
                        
                        energies.append(energy)
                
                if not has_nan and len(energies) > 1:
                    spike = max(energies) / (energies[0] + 1e-8)
                    max_energy_spike = max(max_energy_spike, spike)
            
            results[pattern_name] = {
                "nan_frequency": nan_count / num_trials,
                "max_energy_spike": max_energy_spike
            }
            
            print(f"    NaN Frequency: {nan_count}/{num_trials}")
            print(f"    Max Energy Spike: {max_energy_spike:.2f}x")
        
        # Verdict
        all_stable = all(r["nan_frequency"] == 0 for r in results.values())
        status = "✅ STABLE" if all_stable else "⚠️  UNSTABLE"
        print(f"\n  Overall Status: {status}")
        
        return results


def run_comprehensive_suite(checkpoint_path=None):
    """Run all energy conservation tests."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("  GFN ENERGY CONSERVATION TEST SUITE")
    print("=" * 60)
    print(f"Device: {device}\n")
    
    # Load or create model
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
        print("Using randomly initialized model (for demonstration)\n")
    
    # Run tests
    tester = EnergyConservationTester(model, device)
    
    # Test 1: Long sequence drift
    drift_results = tester.test_long_sequence_drift(seq_length=1000, tolerance=0.05)
    
    # Test 2: Integrator comparison
    integrator_results = tester.test_integrator_comparison(seq_length=500)
    
    # Test 3: Adversarial stability
    adversarial_results = tester.test_adversarial_stability(num_trials=10)
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Long Sequence): {'✅ PASS' if drift_results['passed'] else '❌ FAIL'}")
    print(f"  - Energy drift: {drift_results['relative_drift']*100:.3f}%")
    print(f"  - Stability score: {drift_results['stability_score']:.3f}")
    
    print(f"\nTest 2 (Integrator Comparison):")
    for int_type, res in integrator_results.items():
        print(f"  - {int_type.upper()}: {res['drift']*100:.3f}% drift")
    
    print(f"\nTest 3 (Adversarial Stability):")
    for pattern, res in adversarial_results.items():
        print(f"  - {pattern}: {res['nan_frequency']*100:.0f}% NaN rate")
    
    print("\n✓ All plots saved to tests/professional/results/")
    print("=" * 60)
    
    return {
        "drift": drift_results,
        "integrator": integrator_results,
        "adversarial": adversarial_results
    }


if __name__ == "__main__":
    # Allow checkpoint path as argument
    ckpt_path = None
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    results = run_comprehensive_suite(ckpt_path)
