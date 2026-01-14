
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src import GFN, AdjointGFN
try:
    from tests.professional.baselines import MicroGPT
except ImportError:
    from baselines import MicroGPT

def benchmark_memory_speed(model, batch_size, seq_len, device, model_name):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create dummy data
    # GFN expects [batch, seq_len]
    inputs = torch.randint(00, 20, (batch_size, seq_len)).to(device)
    
    # Warmup
    try:
        with torch.no_grad():
            output, _ = model(inputs) if "GFN" in model_name else (model(inputs), None)
    except RuntimeError as e:
        if "out of memory" in str(e):
            return {"memory": float('nan'), "throughput": 0}
        raise e
            
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Measurement Loop (Forward + Backward to simulate training load)
    # We use a dummy loss to trigger backward
    iters = 5
    total_tokens = batch_size * seq_len * iters
    
    try:
        for _ in range(iters):
            if "GFN" in model_name:
                output, _ = model(inputs)
            else: # GPT
                output = model(inputs)
                
            loss = output.mean()
            loss.backward()
            model.zero_grad()
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2 # MB
        duration = end_time - start_time
        throughput = total_tokens / duration # tokens/sec
        
        return {"memory": peak_mem, "throughput": throughput}
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM for {model_name} at L={seq_len}")
            return {"memory": float('nan'), "throughput": 0}
        raise e

def run_suite():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    # Configuration to match roughly 1.6M params
    vocab_size = 20
    dim = 512
    depth = 12 # Matches GFN depth
    rank = 16  # For GFN
    heads = 4  # For GPT (512/4 = 128 dim head)
    
    # Initialize models
    models = {
        "MicroGPT (Baseline)": MicroGPT(vocab_size, dim, depth, heads).to(device),
        "GFN-Adjoint (Ours)": AdjointGFN(vocab_size, dim, depth, rank).to(device)
    }
    
    # Print param counts
    for name, m in models.items():
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name}: {params:.2f}M params")

    lengths = [128, 512, 1024, 2048]
    batch_size = 4 # Keep small to push sequence length limits
    
    results = []
    
    for name, model in models.items():
        print(f"\nBenchmarking {name}...")
        model.train() # Training mode for memory test
        
        for length in lengths:
            print(f"  Length {length}...", end="", flush=True)
            metrics = benchmark_memory_speed(model, batch_size, length, device, name)
            print(f" Mem: {metrics['memory']:.1f}MB, Speed: {metrics['throughput']:.1f} tok/s")
            
            results.append({
                "Model": name,
                "Sequence Length": length,
                "Peak Memory (MB)": metrics['memory'],
                "Throughput (tok/s)": metrics['throughput']
            })
            
    # Save results
    df = pd.DataFrame(results)
    os.makedirs("tests/professional/results", exist_ok=True)
    df.to_csv("tests/professional/results/benchmark_data.csv", index=False)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    # Memory Plot
    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(data=df, x="Sequence Length", y="Peak Memory (MB)", hue="Model", marker="o", linewidth=2.5)
    plot.set(xscale="log", yscale="log")
    plt.title("O(1) Memory Verification: GFN vs Transformer", fontsize=14)
    plt.ylabel("Peak VRAM (MB) [Log Scale]")
    plt.xlabel("Sequence Length [Log Scale]")
    plt.savefig("tests/professional/results/memory_scaling.png")
    plt.close()
    
    # Throughput Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Sequence Length", y="Throughput (tok/s)", hue="Model")
    plt.title("Training Throughput Comparison", fontsize=14)
    plt.savefig("tests/professional/results/throughput.png")
    plt.close()
    
    print("\nBenchmarks Complete. Plots saved to tests/professional/results/")

if __name__ == "__main__":
    run_suite()
