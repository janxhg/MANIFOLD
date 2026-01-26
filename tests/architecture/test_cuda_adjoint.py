import torch
import pytest
import numpy as np
from gfn.model import Manifold
from gfn.cuda import ops

def get_numerical_grad(model, inputs, param_name, eps=1e-3):
    """
    Computes numerical gradient of total loss w.r.t a parameter.
    """
    param = dict(model.named_parameters())[param_name]
    orig_data = param.data.clone()
    
    grad_num = torch.zeros_like(param)
    
    # Flatten for iteration
    flat_param = param.view(-1)
    flat_grad = grad_num.view(-1)
    
    for i in range(flat_param.numel()):
        # + eps
        flat_param[i] = orig_data.view(-1)[i] + eps
        logits_p, _, _ = model(inputs)
        loss_p = logits_p.pow(2).sum()
        
        # - eps
        flat_param[i] = orig_data.view(-1)[i] - eps
        logits_m, _, _ = model(inputs)
        loss_m = logits_m.pow(2).sum()
        
        flat_grad[i] = (loss_p - loss_m) / (2 * eps)
        flat_param[i] = orig_data.view(-1)[i] # Reset
        
    return grad_num

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("topology", ["low_rank", "torus"])
def test_cuda_adjoint_consistency(topology, metrics):
    """
    Compares analytical gradients from CUDA kernel with numerical gradients.
    """
    dim = 16
    rank = 4
    batch = 2
    seq_len = 3
    
    model = Manifold(
        vocab_size=10,
        dim=dim,
        depth=1,
        heads=1,
        rank=rank,
        integrator_type='leapfrog',
        topology_type=topology,
        use_scan=False
    ).cuda()
    
    # Enable gradients for critical parameters
    for p in model.parameters():
        p.requires_grad = True
        
    inputs = torch.randint(0, 10, (batch, seq_len)).cuda()
    
    # 1. Analytical Gradient (CUDA)
    logits, _, _ = model(inputs)
    loss = logits.pow(2).sum()
    loss.backward()
    
    results = {}
    
    # Check U, W, and Forget Gates
    param_targets = []
    if topology == "low_rank":
        param_targets = ["layers.0.christoffel_adapter.U", "layers.0.christoffel_adapter.W"]
    else:
        # Torus analytical doesn't have U/W usually, but check forget gates and singularity
        param_targets = []
        
    # Check forget gates (Always present)
    param_targets.append("layers.0.forget_gate.weight")
    
    # Check singularity parameters if active
    if hasattr(model.layers[0], 'singularity'):
        param_targets.append("layers.0.singularity.weight")

    for p_name in param_targets:
        analytical = dict(model.named_parameters())[p_name].grad
        if analytical is None: continue
        
        numerical = get_numerical_grad(model, inputs, p_name)
        
        cos_sim = torch.nn.functional.cosine_similarity(analytical.flatten(), numerical.flatten(), dim=0)
        rel_err = torch.norm(analytical - numerical) / (torch.norm(numerical) + 1e-9)
        
        results[f"{p_name}_cos_sim"] = cos_sim.item()
        results[f"{p_name}_rel_err"] = rel_err.item()
        
        metrics.log(f"{p_name}_cos_sim", cos_sim)
        metrics.log(f"{p_name}_rel_err", rel_err)
        
        print(f"\n[{p_name}] Cosine Similarity: {cos_sim:.6f}, Rel Error: {rel_err:.6e}")
        
        # Threshold: Adjoint should be very accurate for short sequences
        # Leapfrog with high plasticity might have some drift, but > 0.99 cos_sim is expected.
        assert cos_sim > 0.95, f"Gradient mismatch in {p_name}: CosSim={cos_sim}"

    metrics.log("topology", topology)
    metrics.finish()
