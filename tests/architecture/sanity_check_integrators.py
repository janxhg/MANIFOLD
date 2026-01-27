
import torch
from gfn.geometry import ToroidalChristoffel
from gfn.integrators import (
    RK4Integrator, HeunIntegrator, DormandPrinceIntegrator, EulerIntegrator,
    SymplecticIntegrator, LeapfrogIntegrator, YoshidaIntegrator, 
    ForestRuthIntegrator, OmelyanIntegrator, CouplingFlowIntegrator, NeuralIntegrator
)

def test_all_integrators():
    dim = 4
    christ = ToroidalChristoffel(dim)
    
    x = torch.randn(1, dim)
    v = torch.randn(1, dim)
    f = torch.randn(1, dim)
    
    integrators = [
        RK4Integrator(christ),
        HeunIntegrator(christ),
        DormandPrinceIntegrator(christ),
        EulerIntegrator(christ),
        SymplecticIntegrator(christ),
        LeapfrogIntegrator(christ),
        YoshidaIntegrator(christ),
        ForestRuthIntegrator(christ),
        OmelyanIntegrator(christ),
        CouplingFlowIntegrator(christ),
        NeuralIntegrator(christ, dim=dim)
    ]
    
    print("Testing integrator consistency...")
    for integ in integrators:
        name = integ.__class__.__name__
        try:
            x_next, v_next = integ(x, v, force=f, steps=1)
            print(f"[PASS] {name} ran successfully.")
            # Check wrapping
            if x_next.max() > 6.284 or x_next.min() < -0.001:
                print(f"[FAIL] {name} failed wrapping consistency! x range: [{x_next.min().item():.2f}, {x_next.max().item():.2f}]")
        except Exception as e:
            print(f"[FAIL] {name} crashed: {e}")

if __name__ == "__main__":
    test_all_integrators()
