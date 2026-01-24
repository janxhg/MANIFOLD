
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gfn.model import Manifold
from tests.benchmarks.viz.vis_gfn_superiority import ParityTask

def debug_run():
    device = 'cpu' # Debug on CPU for simplicity and printing
    print("--- GFN Diagnostic Run ---")
    
    # 1. Setup Model matching the failing test
    dim = 128
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {'enabled': False}, 
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.3}
    }
    
    manifold = Manifold(
        vocab_size=2, dim=dim, depth=6, heads=1, 
        integrator_type='leapfrog',
        physics_config=physics_config,
        holographic=True
    ).to(device)
    
    # Apply Test Script Settings
    if hasattr(manifold.embedding, 'impulse_scale'):
        print(f"[*] Setting impulse_scale to 12.0 (User Config)")
        manifold.embedding.impulse_scale = 12.0
    
    # Initialize Friction (The previous fix)
    for layer in manifold.layers:
        if hasattr(layer.christoffels[0], 'forget_gate'):
             nn.init.constant_(layer.christoffels[0].forget_gate.bias, 2.0)

    # 2. Generate Data
    task = ParityTask(length=5) # Short sequence
    x, y = task.generate_batch(2, device=device) # Batch 2
    
    print(f"\n[Data] Input x: {x[0].tolist()}")
    print(f"[Data] Target y: {y[0].tolist()}")
    
    # 3. Hook for Embeddings
    emb_output = []
    def emb_hook(module, input, output):
        emb_output.append(output)
        print(f"\n[Embedding Hook] Output Shape: {output.shape}")
        print(f"[Embedding Hook] Output Mean: {output.mean().item():.4f}, Max: {output.max().item():.4f}")
        print(f"[Embedding Hook] Output at dim 0 (Force): {output[0, :, 0].tolist()}")
        print(f"[Embedding Hook] Input IDs: {input[0][0].tolist()}")
        
    manifold.embedding.register_forward_hook(emb_hook)
    
    # 4. Run Forward
    manifold.train()
    optimizer = torch.optim.AdamW(manifold.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    print("\n[Forward Pass]...")
    output = manifold(x, collect_christ=False)
    
    # 5. Analyze Output
    if isinstance(output, tuple):
        # output[0] is logits [batch, seq_len, dim]
        logits = output[0]
        x_pred = logits[:, :, 0]
    else:
        logits = output
        x_pred = logits[:, :, 0]
        
    print(f"[Output] Logits Mean: {logits.mean().item():.4f}")
    print(f"[Output] x_pred (Dim 0): {x_pred[0].tolist()}")
    
    # 6. Compute Loss
    y_float = y.float()
    loss = (1.0 - torch.cos(x_pred - y_float)).mean()
    print(f"[Loss] Value: {loss.item():.4f}")
    
    # 7. Backward
    loss.backward()
    
    # 8. Check Gradients
    print("\n[Gradient Check]")
    if manifold.embedding.out_proj.weight.grad is not None:
         grad_norm = manifold.embedding.out_proj.weight.grad.norm().item()
         print(f"Embedding Gradient Norm: {grad_norm:.6f}")
    else:
         print("Embedding Gradient: None")
         
    if manifold.layers[0].christoffels[0].forget_gate.weight.grad is not None:
         grad_norm = manifold.layers[0].christoffels[0].forget_gate.weight.grad.norm().item()
         print(f"Friction Gate Gradient Norm: {grad_norm:.6f}")
         
    if manifold.x0.grad is not None:
        print(f"x0 Gradient Norm: {manifold.x0.grad.norm().item():.6f}")

    # 9. Verify Bug: Check impulse_scale value AFTER forward
    print(f"\n[Verification] Current impulse_scale in embedding: {manifold.embedding.impulse_scale}")

if __name__ == "__main__":
    debug_run()
