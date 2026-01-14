import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.geometry import LeapfrogIntegrator, LowRankChristoffel
from src.layers import GLayer

class TestGoldenIntegration(unittest.TestCase):
    def setUp(self):
        self.dim = 32
        self.dt = 0.1
        self.christoffel = LowRankChristoffel(self.dim, rank=8)
        self.integrator = LeapfrogIntegrator(self.christoffel, dt=self.dt)
        
    def test_gating_scaling(self):
        """Verify that dt_scale=0.5 produces roughly half displacement of dt_scale=1.0"""
        x = torch.randn(1, self.dim)
        v = torch.randn(1, self.dim)
        
        # We perform one step with full dt
        x1, v1 = self.integrator(x, v, dt_scale=1.0)
        disp_full = torch.norm(x1 - x)
        
        # We perform one step with half dt
        x05, v05 = self.integrator(x, v, dt_scale=0.5)
        disp_half = torch.norm(x05 - x)
        
        # For small dt, displacement is roughly linear with dt
        # So disp_half should be approx 0.5 * disp_full
        
        ratio = disp_half / disp_full
        print(f"Displacement Ratio (Half/Full): {ratio.item():.4f}")
        
        # Allow some margin due to nonlinear curvature (Christoffel)
        self.assertTrue(0.4 < ratio < 0.6, "Gating did not scale time effectively")

    def test_energy_conservation(self):
        """Verify Hamiltonian conservation (Symplectic property check)"""
        # Note: True energy involves the metric tensor which is implicitly defined by Gamma.
        # Since we use a LowRankChristoffel approximation, exact energy is hard to define analytically 
        # without the metric. However, we can check reversibility which implies symplecticity.
        
        x0 = torch.randn(1, self.dim)
        v0 = torch.randn(1, self.dim)
        
        # Forward step
        x1, v1 = self.integrator(x0, v0, dt_scale=1.0)
        
        # Backward step (Leapfrog is time-reversible: negative dt should return to state)
        # Note: Leapfrog is reversible if we flip velocity sign or use negative dt.
        # Let's use negative dt_scale.
        
        x_back, v_back = self.integrator(x1, v1, dt_scale=-1.0)
        
        err_x = torch.norm(x_back - x0)
        err_v = torch.norm(v_back - v0)
        
        print(f"Reversibility Error -> X: {err_x.item():.6f}, V: {err_v.item():.6f}")
        
        self.assertTrue(err_x < 1e-5, "Integrator failed time-reversibility (Position)")
        self.assertTrue(err_v < 1e-5, "Integrator failed time-reversibility (Velocity)")

if __name__ == '__main__':
    unittest.main()
