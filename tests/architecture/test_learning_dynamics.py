import torch
import pytest
from gfn.geometry.hyper import HyperChristoffel
from gfn.geometry.analytical import HyperbolicChristoffel

def test_gradient_flow_curvature(metrics):
    """
    Professional analysis of gradient norms across the ODE flow.
    Verifies that the manifold does not cause gradient explosion/vanishing.
    """
    dim = 256
    geom = HyperChristoffel(dim=dim, rank=32)
    # Inject noise into EVERYTHING so gradients Flow
    torch.nn.init.normal_(geom.gate_u.weight, std=0.01)
    torch.nn.init.normal_(geom.gate_w.weight, std=0.01)
    torch.nn.init.normal_(geom.U, std=0.01)
    torch.nn.init.normal_(geom.W, std=0.01)
    
    x = torch.randn(1, dim, requires_grad=True)
    v = torch.randn(1, dim, requires_grad=True)
    
    # Compute force
    gamma = geom(v, x)
    
    # Check backprop through geometry
    loss = gamma.pow(2).sum()
    loss.backward()
    
    grad_x_norm = x.grad.norm().item()
    grad_v_norm = v.grad.norm().item()
    
    metrics.log("grad_x_norm", grad_x_norm)
    metrics.log("grad_v_norm", grad_v_norm)
    
    assert 1e-5 < grad_x_norm < 1e5
    assert 1e-5 < grad_v_norm < 1e5

def test_hessian_spectrum_proxy(metrics):
    """
    Estimates the conditioning of the local geometric landscape.
    Uses the trace of the Fisher Information Matrix proxy.
    """
    dim = 64
    geom = HyperChristoffel(dim=dim, rank=16)
    x = torch.randn(1, dim)
    v = torch.randn(1, dim)
    
    # Stochastic trace estimation (Hutchinson's trace estimator)
    traces = []
    for _ in range(10):
        z = torch.randn(1, dim)
        with torch.enable_grad():
            x_test = x.clone().detach().requires_grad_(True)
            gamma = geom(v, x_test)
            gz = (gamma * z).sum()
            grad = torch.autograd.grad(gz, x_test, create_graph=True)[0]
            trace_est = (grad * z).sum().item()
            traces.append(trace_est)
            
    avg_trace = sum(traces) / len(traces)
    metrics.log("hessian_trace_avg", avg_trace)
    
    # Trace should be finite
    assert not np.isnan(avg_trace)

import numpy as np
