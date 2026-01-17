import torch
import torch.nn as nn
import sys
import gc
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Create results directory
RESULTS_DIR = PROJECT_ROOT / 'tests' / 'benchmarks' / 'results' / 'inf_scaling'
os.makedirs(RESULTS_DIR, exist_ok=True)

from src.model import Manifold

def measure_vram(vocab_size, embedding_type, readout_type='standard', device='cuda'):
    """
    Measures Peak VRAM and Param Count.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Config
    physics_config = {
        'embedding': {
            'type': embedding_type,
            'coord_dim': 16
        },
        'readout': {
            'type': readout_type
        },
        'active_inference': {'enabled': False}, # Disable for pure memory test
        'mixture': {'enabled': False}
    }
    
    try:
        # Init Model
        model = Manifold(
            vocab_size=vocab_size,
            dim=256, # Fixed hidden dim
            depth=2, # Shallow for embedding focus
            heads=4,
            integrator_type='heun',
            physics_config=physics_config
        ).to(device)
        
        # Params in Millions
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Dummy Input
        batch_size = 8 # Reduce batch size to allow testing 1M even on small GPU?
        seq_len = 32
        x = torch.randint(0, min(vocab_size, 100), (batch_size, seq_len)).to(device) # Inputs are safe indices
        
        # Forward + Backward
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        
        logits, _, _ = model(x)
        
        # If implicit readout, output is [batch, seq, coord_dim]
        # Loss must be custom or dummy
        loss = logits.mean()
        loss.backward()
        
        # Measure Peak VRAM
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        
        del model, x, logits, loss, optimizer
        return params, peak_mem
        
    except Exception as e:
        print(f"OOM or Error for {vocab_size} {embedding_type}/{readout_type}: {e}")
        return None, None

def run_benchmark():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    os.makedirs(run_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available. VRAM measurement invalid.")
        return

    # Check 1M and 10M?
    vocab_sizes = [10_000, 100_000, 500_000, 1_000_000] 
    
    results = []
    
    modes = [
        {'name': 'Standard', 'emb': 'standard', 'read': 'standard'},
        {'name': 'Functional', 'emb': 'functional', 'read': 'standard'},
        {'name': 'Infinite', 'emb': 'functional', 'read': 'implicit'}, # The Goal
    ]
    
    print(f"{'Vocab':<10} | {'Mode':<10} | {'Params (M)':<12} | {'VRAM (MB)':<10}")
    print("-" * 50)
    
    for v in vocab_sizes:
        for mode in modes:
            p, mem = measure_vram(v, mode['emb'], mode['read'], device)
            
            if p is not None:
                print(f"{v:<10} | {mode['name']:<10} | {p:<12.2f} | {mem:<10.2f}")
                results.append({
                    'Vocab': v, 
                    'Mode': mode['name'], 
                    'Params': p, 
                    'VRAM': mem,
                    'Embedding': mode['emb'],
                    'Readout': mode['read']
                })
            else:
                # Log failure
                results.append({
                    'Vocab': v, 
                    'Mode': mode['name'], 
                    'Params': None, 
                    'VRAM': None,
                    'Error': 'OOM'
                })
             
    # Save Data
    df = pd.DataFrame(results)
    json_path = run_dir / 'data.json'
    df.to_json(json_path, orient='records', indent=2)
    print(f"\nData saved to {json_path}")
    
    # Plotting
    try:
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Params
        plt.subplot(1, 2, 1)
        for m in modes:
            name = m['name']
            data = df[(df['Mode'] == name) & (df['Params'].notna())]
            if not data.empty:
                plt.plot(data['Vocab'], data['Params'], marker='o', label=name)
        plt.title("Model Parameters vs Vocab Size")
        plt.xlabel("Vocab Size")
        plt.ylabel("Parameters (Millions)")
        plt.legend()
        plt.grid(True)
        # Log scale if massive difference?
        # plt.yscale('log')
        
        # Subplot 2: VRAM
        plt.subplot(1, 2, 2)
        for m in modes:
            name = m['name']
            data = df[(df['Mode'] == name) & (df['VRAM'].notna())]
            if not data.empty:
                plt.plot(data['Vocab'], data['VRAM'], marker='o', label=name)
        plt.title("Peak VRAM (Train Step) vs Vocab Size")
        plt.xlabel("Vocab Size")
        plt.ylabel("VRAM (MB)")
        plt.legend()
        plt.grid(True)
        
        plot_path = run_dir / 'inf_scaling.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_benchmark()
