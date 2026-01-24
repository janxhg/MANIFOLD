"""
Professional GFN Superiority Dashboard
======================================
Comprehensive comparison of Manifold GFN vs Transformer on state-tracking tasks.
Visualizes convergence speed, long-context generalization, and VRAM scaling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import GFN Models & Physics
from gfn.model import Manifold
from gfn.optim import RiemannianAdam
from gfn.losses import geodesic_regularization, hamiltonian_loss, CircularDistanceLoss

# Import Baselines & Utils
from tests.benchmarks.baselines import MicroGPT
from tests.benchmarks.bench_utils import ResultsLogger, PerformanceStats

class PeriodicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # pred: [batch, 1] angle in radians
        # target: [batch, 1] angle in radians
        # L = 1 - cos(pred - target)
        # Minimized when pred = target + 2k*pi
        return (1.0 - torch.cos(pred - target)).mean()

class ParityTask:
    """Parity Check (Modulo 2) for state tracking."""
    def __init__(self, vocab_size=2, length=20, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y_int = torch.cumsum(x, dim=1) % self.mod
        
        # Scaling for Manifold (Topological) vs GPT (Classification)
        PI = 3.14159265359
        y_angle = y_int.float() * PI
        return x, y_int, y_angle

def train_step_manifold(model, optimizer, scheduler, inputs, targets, device):
    optimizer.zero_grad()
    # CRITICAL: collect_christ=False to enable CUDA fused kernel
    # Model output depends on architecture. 
    # If Manifold, it returns (x_next, v_next, context, christoffels)
    # We want x_next.
    output = model(inputs, collect_christ=False)
    
    if isinstance(output, tuple):
        # output[0] is logits [batch, seq_len, dim] (because Holographic Identity readout)
        x_pred = output[0] # [batch, seq_len, dim]
    else:
        x_pred = output
        
    y_float = targets.float() # [batch, seq_len]
    
    # HOLOGRAPHIC MODE: Pred is state x [Batch, Seq, Dim]. Target is y [Batch, Seq].
    # CRITICAL FIX: Supervise ALL dimensions to prevent unsupervised drift noise
    y_expanded = y_float.unsqueeze(-1).expand_as(x_pred)
    
    criterion = CircularDistanceLoss()
    loss_val = criterion(x_pred, y_expanded)
    
    loss_phy = 0.0
    loss_ham = 0.0
    # Unpack Model Output (logits, (x,v), christoffels, v_seq, x_seq, all_forces)
    if isinstance(output, tuple) and len(output) >= 6:
        christoffels = output[2]
        v_seq = output[3]
        x_seq = output[4]
        all_forces = output[5]
        
        if christoffels:
            loss_phy = geodesic_regularization(None, christoffels, lambda_g=0.001)
        
            # Level 15: Riemannian Hamiltonian Conservation (Force-Aware)
            # CRITICAL FIX: Hamiltonian Loss needs a metric_fn. 
            # We use the metric from the first head of the first layer if available.
            def first_head_metric(x):
                 return model.layers[0].christoffels[0].get_metric(x) if hasattr(model.layers[0].christoffels[0], 'get_metric') else torch.ones_like(x)

            loss_ham = hamiltonian_loss(v_seq, states=x_seq, metric_fn=first_head_metric, lambda_h=0.0, forces=all_forces)
            
    total_loss = loss_val + loss_phy + loss_ham
    
    if torch.isnan(total_loss):
        print("NaN detected in loss!")
        return total_loss, 0.0
        
    total_loss.backward()
    
    # Level 23: Relaxed Clipping (Soft Norm)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    if scheduler: scheduler.step()
    
    # Accuracy: Check mean distance < 1.0 rad
    with torch.no_grad():
        PI = 3.14159265359
        TWO_PI = 2.0 * PI
        diff = torch.abs(x_pred - y_expanded) % TWO_PI
        diff = torch.min(diff, TWO_PI - diff)
        acc = (diff.mean(dim=-1) < 1.0).float().mean().item()
    
    return total_loss.item(), acc

def train_step_gpt(model, optimizer, scheduler, inputs, targets, device):
    optimizer.zero_grad()
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, 2), targets.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    if scheduler: scheduler.step()
    
    preds = logits.argmax(dim=-1)
    acc = (preds == targets).float().mean().item()
    return loss.item(), acc

def train_model(model, max_steps=1000, device='cuda'):
    is_manifold = isinstance(model, Manifold)
    if is_manifold:
        torus_params = []
        std_params = []
        for name, p in model.named_parameters():
            if 'x0' in name or 'v0' in name or 'impulse_scale' in name or 'gate' in name:
                torus_params.append(p)
            else:
                std_params.append(p)

        optimizer = optim.AdamW([
            {'params': std_params, 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': torus_params, 'lr': 1e-2, 'weight_decay': 0}
        ])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=max_steps, pct_start=0.2)
    model.train()
    
    history = {"loss": [], "acc": []}
    
    for i in range(max_steps):
        L = 20
        
        task = ParityTask(length=L)
        x, y_class, y_angle = task.generate_batch(128, device=device)
        
        if is_manifold:
            loss, acc = train_step_manifold(model, optimizer, scheduler, x, y_angle, device)
        else:
            loss, acc = train_step_gpt(model, optimizer, scheduler, x, y_class, device)
            
        history["loss"].append(loss)
        history["acc"].append(acc)
        
        if i % 100 == 0 and is_manifold:
            with torch.no_grad():
                output = model(x[:1, :L], collect_christ=False)
                x_vals = [st[0, 0].item() for st in output[4]]
                y_vals = [y_angle[0, t].item() for t in range(L)]
                all_f = output[5]
                f_norms = [all_f[0, t].norm().item() for t in range(L)]
                
                # Robust Layer Access
                target_layer = model.layers[0]
                if hasattr(target_layer, 'macro_manifold'):
                     target_layer = target_layer.macro_manifold
                
                gate_bias = target_layer.christoffels[0].forget_gate.bias.mean().item() if hasattr(target_layer, 'christoffels') else 0.0
                
                x0_grad = model.x0.grad.norm().item() if model.x0.grad is not None else 0.0
                emb_layer = model.embedding.out_proj if hasattr(model.embedding, 'out_proj') else model.embedding
                emb_grad = next(emb_layer.parameters()).grad.norm().item() if next(emb_layer.parameters()).grad is not None else 0.0
                
                imp_val = model.embedding.impulse_scale.item() if hasattr(model.embedding, 'impulse_scale') else 0.0
                print(f"[Step {i}] Loss: {loss:.4f} | Acc: {acc*100:.1f}% | Impulse: {imp_val:.1f}")
                for t in range(L):
                    print(f"  t={t}: x={x_vals[t]:.3f}, target={y_vals[t]:.3f}, force={f_norms[t]:.3f}")
                print(f"  Friction Bias: {gate_bias:.3f} | Gradients -> x0: {x0_grad:.2e}, Emb: {emb_grad:.2e}")
                sys.stdout.flush()
        elif i % 10 == 0:
            if i % 50 == 0:
                print(f"Step {i}... Loss: {loss:.4f}, Acc: {acc*100:.1f}%")
                sys.stdout.flush()
            
    return history

def evaluate_scaling(model, lengths, device='cuda'):
    model.eval()
    results = {"acc": [], "mem": []}
    
    for L in lengths:
        task = ParityTask(length=L)
        x, y_class, y_angle = task.generate_batch(100, device=device)
        
        def run_inf():
            with torch.no_grad():
                if isinstance(model, Manifold):
                    # O(1) Streaming Inference
                    state = None
                    preds_list = []
                    for t in range(x.shape[1]):
                        out = model(x[:, t:t+1], state=state)
                        l = out[0]
                        state = out[1]
                        # On Torus: Check distance to PI
                        PI = 3.14159265359
                        TWO_PI = 2.0 * PI
                        dist_to_pi = torch.min(torch.abs(l - PI) % TWO_PI, TWO_PI - (torch.abs(l - PI) % TWO_PI))
                        dist_to_0 = torch.min(torch.abs(l) % TWO_PI, TWO_PI - (torch.abs(l) % TWO_PI))
                        # Multi-dimensional Torus: average distance across dimensions
                        d_pi = dist_to_pi.mean(dim=-1).view(-1)
                        d_0 = dist_to_0.mean(dim=-1).view(-1)
                        preds_list.append((d_pi < d_0).long())
                    return torch.stack(preds_list, dim=1)
                else:
                    return model(x).argmax(dim=-1)

        mem = PerformanceStats.measure_peak_memory(model, run_inf)
        preds = run_inf()
        acc = (preds == y_class).float().mean().item()
        
        results["acc"].append(acc)
        results["mem"].append(mem)
        print(f"  L={L}: Acc={acc*100:.1f}% | Mem={mem:.1f}MB")
        torch.cuda.empty_cache()
            
    return results

def run_superiority_benchmark():
    logger = ResultsLogger("superiority", category="viz")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Models
    dim = 128 # Sufficient capacity
    physics_config = {
        'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16}, 
        'readout': {'type': 'implicit', 'coord_dim': 16},
        'active_inference': {
            'enabled': True, 
            'dynamic_time': {'enabled': True},
            'reactive_curvature': {'enabled': True, 'plasticity': 0.2},
            'singularities': {'enabled': True, 'strength': 20.0, 'threshold': 0.8}
        },
        'fractal': {'enabled': True, 'threshold': 0.5, 'alpha': 0.2},
        'topology': {'type': 'torus'},
        'stability': {'base_dt': 0.2}
    }
    
    # FLOW SCALE (Recalibrated for dt=0.2)
    # Norm(Force) \approx 80 -> impulse_scale \approx 20.
    # DASH-AND-STOP SCALE (Balanced for Capture)
    impulse_scale = 80.0 
    manifold = Manifold(
        vocab_size=2, dim=dim, depth=6, heads=4, 
        integrator_type='leapfrog',
        physics_config=physics_config,
        impulse_scale=impulse_scale, 
        holographic=True
    ).to(device)
    
    # LEVEL 29: KICKSTART INITIALIZATION
    with torch.no_grad():
        # LEVEL 34: SUPER-FLOW INITIALIZATION
        manifold.x0.data.fill_(0.1) 
        manifold.x0.requires_grad = True
        manifold.v0.data.fill_(0.01) 
        manifold.v0.requires_grad = True
        # HIGH STATIC FRICTION (Stiff by default to prevent drift)
        # bias +2.0 -> mu \approx 0.88 * 10 = 8.8
        # FORCE-TRIGGERED LUBRICATION (Force reduces friction)
        for layer in manifold.layers:
             target = layer.macro_manifold if hasattr(layer, 'macro_manifold') else layer
             if hasattr(target, 'christoffels'):
                 for head_geo in target.christoffels:
                  if hasattr(head_geo, 'forget_gate'):
                       nn.init.constant_(head_geo.forget_gate.bias, 2.0)
                  if hasattr(head_geo, 'input_gate'):
                       # Negative weight: Force UP -> Activ DOWN -> Friction DOWN
                       nn.init.constant_(head_geo.input_gate.weight, -0.5) 
    

    # LEVEL 8: EXPLICIT ZERO DISSIPATION & FLAT METRIC
    # Removed manual patch. Handled by LowRankChristoffel internally.
    gpt = MicroGPT(vocab_size=2, dim=dim, depth=6, heads=1, max_len=100000).to(device)
    
    # 2. Training Phase
    # Increased steps for Fractal convergence
    h_m = train_model(manifold, max_steps=6000, device=device)
    h_g = train_model(gpt, max_steps=6000, device=device)
    
    # 3. Scaling Phase
    # Paper claims 100,000. Let's push to 10,000 for this run.
    lengths = [20, 100, 500, 1000, 5000, 10000]
    print("\n--- Evaluating Manifold Scaling ---")
    s_m = evaluate_scaling(manifold, lengths, device)
    print("\n--- Evaluating GPT Scaling ---")
    s_g = evaluate_scaling(gpt, lengths, device)
    
    # 4. Dashboard Plotting
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # Plot A: Convergence Loss
    axes[0, 0].plot(h_m["loss"], color='#E76F51', label='Manifold (Hamiltonian)', alpha=0.8)
    axes[0, 0].plot(h_g["loss"], color='#264653', label='Transformer (CE)', alpha=0.8)
    axes[0, 0].set_title("Training Convergence (Loss)", fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()

    # Plot B: Accuracy Trend
    axes[0, 1].plot(np.convolve(h_m["acc"], np.ones(10)/10, mode='valid'), color='#E76F51', label='Manifold')
    axes[0, 1].plot(np.convolve(h_g["acc"], np.ones(10)/10, mode='valid'), color='#264653', label='Transformer')
    axes[0, 1].set_title("Learning Dynamics (Smoothed Accuracy)", fontweight='bold')
    axes[0, 1].legend()

    # Plot C: Long-Context Generalization
    axes[1, 0].plot(lengths, s_m["acc"], 'o-', color='#2A9D8F', label='Manifold (O(1))', linewidth=3)
    axes[1, 0].plot(lengths, s_g["acc"], 's--', color='#E9C46A', label='Transformer (O(N²))', linewidth=3)
    axes[1, 0].set_title("Out-of-Distribution Generalization", fontweight='bold')
    axes[1, 0].set_xlabel("Sequence Length")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    # Plot D: VRAM Scaling
    axes[1, 1].plot(lengths, s_m["mem"], 'o-', color='#2A9D8F', label='Manifold', linewidth=3)
    axes[1, 1].plot(lengths, s_g["mem"], 's--', color='#E9C46A', label='Transformer', linewidth=3)
    axes[1, 1].set_title("Memory Scaling (VRAM)", fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel("Sequence Length")
    axes[1, 1].set_ylabel("Peak MB")
    axes[1, 1].legend()

    fig.suptitle("Manifold GFN vs Transformer: Superiority Dashboard", fontsize=22, fontweight='bold', y=0.96)
    logger.save_plot(fig, "gfn_superiority_dashboard.png")
    
    # Save Metrics
    logger.save_json({
        "hyperparameters": {"dim": dim, "lengths": lengths},
        "manifold": {"final_acc": h_m["acc"][-1], "ood_scaling": s_m},
        "transformer": {"final_acc": h_g["acc"][-1], "ood_scaling": s_g}
    })

if __name__ == "__main__":
    run_superiority_benchmark()
