import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time


# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Models
from gfn.model import Manifold
from gfn.optim import RiemannianAdam

class ParityTask:
    """Parity Check Task (Cumulative XOR)"""
    def __init__(self, vocab_size=2, length=50, mod=2):
        self.vocab_size = vocab_size
        self.length = length
        self.mod = mod
        
    def generate_batch(self, batch_size, device='cpu'):
        x = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y = torch.cumsum(x, dim=1) % self.mod
        return x, y

def train_one_epoch(model, dtype, steps=200, lr=1e-3, device='cuda'):
    """Trains for a fixed number of steps and monitors stability (NaNs/Inf)"""
    
    # Use Riemannian Adam
    optimizer = RiemannianAdam(model.parameters(), lr=lr, retraction='normalize')
    criterion = nn.BCEWithLogitsLoss()
    
    task = ParityTask(length=50) # Moderate length to test stability
    
    loss_history = []
    grad_norms = []
    
    model.train()
    
    # Enable AMP for FP16/BF16 if native dtype is float32 but we want mixed precision
    # OR we can cast the model to dtype directly.
    # For strict geometric tests, we usually cast the weights.
    
    if dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        
    try:
        start_time = time.time()
        pbar = tqdm(range(steps), desc=f"Testing {dtype}")
        
        for _ in pbar:
            x, y = task.generate_batch(64, device=device)
            
            # Map targets to bits
            coord_dim = model.physics_config.get('embedding', {}).get('coord_dim', 16)
            mask = 2**torch.arange(coord_dim).to(device)
            target_bits = (y.unsqueeze(-1) & mask) > 0
            target_bits = target_bits.float() 

            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    logits, _, _ = model(x)
                    loss = criterion(logits, target_bits)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # For clipping/monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Direct cast (e.g. Bfloat16 or Float32)
                # Note: PyTorch modules must be .to(dtype) for direct execution
                # But input must match.
                x_in = x # integer input
                logits, _, _ = model(x_in)
                loss = criterion(logits, target_bits.to(dtype=logits.dtype))
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"❌ Stability Failure (NaN/Inf) at step {_}")
                return loss_history, grad_norms, False
                
            loss_history.append(loss.item())
            grad_norms.append(grad_norm.item())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'grad': f"{grad_norm.item():.2f}"})
            
        return loss_history, grad_norms, True
        
    except RuntimeError as e:
        print(f"❌ Runtime Error: {e}")
        return loss_history, grad_norms, False

def run_precision_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔬 Manifold Precision Stability Benchmark")
    print(f"Goal: Evaluate geometric stability under FP32 vs BF16 vs FP16")
    
    dtypes = {
        'FP32': torch.float32,
        'BF16 (Brain Float)': torch.bfloat16,
        'FP16 (Half)': torch.float16
    }
    
    results = {}
    
    for name, dtype in dtypes.items():
        if dtype == torch.float16 and device.type == 'cpu':
            print(f"Skipping {name} (Not supported on CPU)")
            continue
            
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
             print(f"Skipping {name} (Hardware not supported)")
             continue
             
        print(f"\nEvaluating {name}...")
        
        # Init Model
        model = Manifold(
            vocab_size=2, dim=128, depth=6, heads=4,
            physics_config={
                'embedding': {'type': 'functional', 'mode': 'linear', 'coord_dim': 16},
                'readout': {'type': 'implicit', 'coord_dim': 16},
                'active_inference': {'enabled': True}, # Complex dynamics
                'stability': {'base_dt': 0.1} # Conservative dt
            }
        ).to(device)
        
        # Apply dtype cast (except for FP16 which usually uses AMP)
        if dtype != torch.float16:
             model.to(dtype=dtype)
        
        losses, grads, success = train_one_epoch(model, dtype, steps=500, device=device)
        
        results[name] = {
            'success': success,
            'final_loss': losses[-1] if losses else 100.0,
            'avg_grad_norm': sum(grads)/len(grads) if grads else 100.0,
            'loss_history': losses
        }
    
    # Check if results folder exists
    out_dir = PROJECT_ROOT / "tests" / "benchmarks" / "results" / "precision_stability"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Metrics
    with open(out_dir / "precision_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        if data['success']:
            plt.plot(data['loss_history'], label=f"{name} (Avg Grad: {data['avg_grad_norm']:.2f})")
    
    plt.title("Training Stability by Precision Format")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "stability_chart.png")
    
    print(f"\n✅ Benchmark Complete. Results saved to {out_dir}")

if __name__ == "__main__":
    run_precision_benchmark()
