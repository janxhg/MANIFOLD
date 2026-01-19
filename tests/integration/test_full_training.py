"""
Quick Integration Test (CI/CD Safe)
====================================

Fast sanity checks for continuous integration.
- Runs on CPU
- Completes in < 2 minutes
- Verifies core functionality without heavy computation
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn import GFN, GFNLoss, RiemannianAdam
from gfn.geometry import LowRankChristoffel, HeunIntegrator, LeapfrogIntegrator


class TestCoreArchitecture:
    """Test fundamental architecture components."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cpu')  # CI-safe
    
    @pytest.fixture
    def small_model(self, device):
        """Small model for fast testing."""
        return GFN(vocab_size=10, dim=32, depth=2, rank=4).to(device)
    
    def test_model_creation(self, small_model):
        """Test model instantiation."""
        assert small_model is not None
        assert isinstance(small_model, nn.Module)
        
        # Check components exist
        assert hasattr(small_model, 'embedding')
        assert hasattr(small_model, 'layers')
        assert hasattr(small_model, 'readout')
    
    def test_forward_pass(self, small_model, device):
        """Test basic forward pass."""
        batch_size = 2
        seq_len = 10
        vocab_size = 10
        
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        
        logits, (x, v) = small_model(inputs)
        
        # Check shapes
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert x.shape == (batch_size, small_model.dim)
        assert v.shape == (batch_size, small_model.dim)
        
        # Check no NaN/Inf
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_backward_pass(self, small_model, device):
        """Test gradient computation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10
        
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        
        criterion = GFNLoss()
        logits, _ = small_model(inputs)
        
        loss, _ = criterion(logits, targets)
        loss.backward()
        
        # Check gradients exist and are finite
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    
    def test_energy_conservation_basic(self, small_model, device):
        """Basic energy conservation check."""
        seq_len = 20
        inputs = torch.randint(0, 10, (1, seq_len)).to(device)
        
        energies = []
        x = small_model.x0.expand(1, -1)
        v = small_model.v0.expand(1, -1)
        all_forces = small_model.embedding(inputs)
        
        with torch.no_grad():
            for t in range(seq_len):
                force = all_forces[:, t]
                for layer in small_model.layers:
                    x, v = layer(x, v, force)
                energy = (v ** 2).sum().item()
                energies.append(energy)
        
        # Energy should not explode
        assert max(energies) < 1e6, "Energy exploded"
        
        # Energy drift should be reasonable
        import numpy as np
        drift = abs(energies[-1] - energies[0]) / (energies[0] + 1e-8)
        assert drift < 1.0, f"Energy drift too high: {drift*100:.1f}%"


class TestGeometry:
    """Test geometric components."""
    
    def test_christoffel_computation(self):
        """Test Christoffel symbol computation."""
        dim = 16
        rank = 4
        christoffel = LowRankChristoffel(dim, rank)
        
        v = torch.randn(2, dim)
        gamma = christoffel(v)
        
        assert gamma.shape == (2, dim)
        assert torch.isfinite(gamma).all()
    
    def test_integrator_stability(self):
        """Test integrator doesn't produce NaN."""
        dim = 16
        integrator = HeunIntegrator(dim, rank=4)
        
        x = torch.randn(2, dim)
        v = torch.randn(2, dim)
        force = torch.randn(2, dim)
        
        x_new, v_new = integrator(x, v, force)
        
        assert torch.isfinite(x_new).all()
        assert torch.isfinite(v_new).all()


class TestOptimizer:
    """Test Riemannian optimizer."""
    
    def test_optimizer_step(self):
        """Test optimizer can take a step."""
        param = nn.Parameter(torch.randn(10, 10))
        optimizer = RiemannianAdam([param], lr=0.01)
        
        # Fake gradient
        param.grad = torch.randn_like(param)
        
        # Should not crash
        optimizer.step()
        
        # Parameter should change
        assert param.grad is not None


class TestLoss:
    """Test loss functions."""
    
    def test_gfn_loss_computation(self):
        """Test GFN loss computation."""
        batch_size = 2
        seq_len = 5
        vocab_size = 10
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        criterion = GFNLoss(lambda_h=0.01)
        loss, loss_dict = criterion(logits, targets)
        
        assert torch.isfinite(loss)
        assert loss.item() >= 0
        assert 'ce' in loss_dict
        assert 'total' in loss_dict


@pytest.mark.parametrize("integrator_type", ['heun', 'rk4', 'leapfrog'])
def test_different_integrators(integrator_type):
    """Test model works with different integrators."""
    model = GFN(vocab_size=10, dim=16, depth=2, rank=4, integrator_type=integrator_type)
    inputs = torch.randint(0, 10, (1, 5))
    
    logits, _ = model(inputs)
    
    assert logits.shape == (1, 5, 10)
    assert torch.isfinite(logits).all()


def test_parameter_count():
    """Test model has reasonable parameter count."""
    model = GFN(vocab_size=20, dim=512, depth=12, rank=16)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Should be around 1-2M parameters
    assert 0.5 < params < 5.0, f"Unexpected param count: {params:.2f}M"


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
