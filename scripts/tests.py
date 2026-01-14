import torch
import torch.nn as nn
import time
from src.model import GFN
from src.geometry import LowRankChristoffel, SymplecticIntegrator, RK4Integrator
from src.layers import GLayer

def benchmark_component(name, func, iter=500):
    # Warmup
    for _ in range(10):
        func()
    torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(iter):
        func()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"{name:.<40} {dt:.4f}s ({iter} iters)")
    return dt

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    dim = 256
    rank = 32
    batch_size = 128
    seq_len = 32
    
    v = torch.randn(batch_size, dim).to(device)
    x = torch.randn(batch_size, dim).to(device)
    force = torch.randn(batch_size, dim).to(device)
    
    # 1. Christoffel Bench
    christoffel = LowRankChristoffel(dim, rank).to(device)
    benchmark_component("Christoffel Forward", lambda: christoffel(v))
    
    # 2. Integrator Bench
    symp = SymplecticIntegrator(christoffel)
    rk4 = RK4Integrator(christoffel)
    
    benchmark_component("Symplectic Integrator (2 steps)", lambda: symp(x, v, force))
    benchmark_component("RK4 Integrator (4 steps)", lambda: rk4(x, v, force))
    
    # 3. Model Depth Bench
    def run_model(d):
        model = GFN(vocab_size=100, dim=dim, depth=d, rank=rank).to(device)
        input_ids = torch.randint(0, 100, (batch_size, seq_len)).to(device)
        return lambda: model(input_ids)

    benchmark_component("GFN Model (Depth 4)", run_model(4), iter=50)
    benchmark_component("GFN Model (Depth 8)", run_model(8), iter=50)
    benchmark_component("GFN Model (Depth 16)", run_model(16), iter=50)

if __name__ == "__main__":
    main()
