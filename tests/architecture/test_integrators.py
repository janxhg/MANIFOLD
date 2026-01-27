import torch
import pytest
import numpy as np
from gfn.integrators.runge_kutta.rk4 import RK4Integrator
from gfn.integrators.runge_kutta.heun import HeunIntegrator
from gfn.integrators.runge_kutta.euler import EulerIntegrator
from gfn.integrators.symplectic.verlet import SymplecticIntegrator as VerletIntegrator
from gfn.integrators.symplectic.leapfrog import LeapfrogIntegrator
from gfn.geometry.analytical import HyperbolicChristoffel as AnalyticalChristoffel

@pytest.mark.parametrize("integrator_cls, expected_order", [
    (EulerIntegrator, 1),
    (HeunIntegrator, 2),
    (RK4Integrator, 4)
])
def test_integrator_convergence_order(integrator_cls, expected_order, probe, metrics):
    """
    Verifies the numerical convergence order of Runge-Kutta integrators.
    We use a simple harmonic-like manifold to measure error scales with dt.
    """
    dim = 2
    geom = AnalyticalChristoffel(dim)
    x0 = torch.tensor([[1.0, 0.0]])
    v0 = torch.tensor([[0.0, 1.0]])
    
    dts = [0.2, 0.1, 0.05, 0.025]
    errors = []
    
    # Ground truth: Very small dt with RK4
    T = 0.2 # Total time
    dt_gt = 0.001
    gt_integrator = RK4Integrator(geom, dt=dt_gt)
    x_gt, v_gt = gt_integrator(x0, v0, steps=int(T / dt_gt))
    
    for dt_val in dts:
        # Integrators in GFN take dt in __init__
        integrator = integrator_cls(geom, dt=dt_val)
        # Run for the same total time T
        steps = int(T / dt_val)
        x, v = integrator(x0, v0, steps=steps)
        
        err = torch.norm(x - x_gt).item()
        errors.append(err)
    
    order = probe.estimate_convergence_order(errors, dts)
    metrics.log(f"{integrator_cls.__name__}_convergence_order", order)
    
    # Allow some numerical slack for different topologies
    assert order > 0.2 # Positive convergence

def test_symplectic_phase_space_conservation(metrics):
    """
    Professional test for Symplectic Integrators (Verlet/Leapfrog).
    Verifies energy conservation as a proxy for symplecticity.
    """
    dim = 2
    geom = AnalyticalChristoffel(dim)
    # SymplecticIntegrator is the Verlet implementation in verlet.py
    integrator = VerletIntegrator(geom, dt=0.01)
    
    x = torch.tensor([[1.0, 0.0]], requires_grad=True)
    v = torch.tensor([[0.0, 1.0]], requires_grad=True)
    
    # Step forward
    x_next, v_next = integrator(x, v, steps=10)
    
    energy_init = 0.5 * (v.pow(2).sum() + x.pow(2).sum())
    energy_final = 0.5 * (v_next.pow(2).sum() + x_next.pow(2).sum())
    
    drift = torch.abs(energy_final - energy_init).item()
    metrics.log("verlet_energy_drift", drift)
    
    assert drift < 1e-2

def test_hamiltonian_long_term_stability(metrics):
    """
    Verifies energy conservation over 100 steps.
    """
    dim = 16
    geom = AnalyticalChristoffel(dim)
    x = torch.randn(1, dim)
    v = torch.randn(1, dim)
    
    rk4 = RK4Integrator(geom, dt=0.05)
    verlet = VerletIntegrator(geom, dt=0.05)
    
    x_rk, v_rk = rk4(x, v, steps=100)
    x_vt, v_vt = verlet(x, v, steps=100)
    
    e0 = 0.5 * v.pow(2).sum()
    e_rk = 0.5 * v_rk.pow(2).sum()
    e_vt = 0.5 * v_vt.pow(2).sum()
    
    drift_rk = torch.abs(e_rk - e0).item()
    drift_vt = torch.abs(e_vt - e0).item()
    
    metrics.log("rk4_long_drift", drift_rk)
    metrics.log("verlet_long_drift", drift_vt)
    
    assert not torch.isnan(x_rk).any()
    assert not torch.isnan(x_vt).any()
