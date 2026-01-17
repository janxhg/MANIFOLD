from runner import ExperimentRunner
import torch
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class SortingTask:
    def __init__(self, vocab_size=100, length=10):
        self.vocab_size = vocab_size
        self.length = length
        self.SEP = vocab_size
        self.EOS = vocab_size + 1
        self.full_vocab = vocab_size + 2
        
    def generate_batch(self, batch_size, device='cpu'):
        x_raw = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        y_raw, _ = torch.sort(x_raw, dim=1)
        inputs = x_raw
        sep = torch.full((batch_size, 1), self.SEP, device=device)
        outputs = y_raw
        eos = torch.full((batch_size, 1), self.EOS, device=device)
        full_seq = torch.cat([inputs, sep, outputs, eos], dim=1)
        src = full_seq[:, :-1]
        tgt = full_seq[:, 1:]
        return src, tgt

def run():
    runner = ExperimentRunner(experiment_name="Arch_Topology")
    
    base_config = {
        'max_steps': 1500,
        'dim': 64, 
        'depth': 2,
        'heads': 4,
        'use_scan': False, # Geodesic
        'integrator': 'heun'
    }
    
    # Compare:
    # 1. Base (Dim 64, Depth 2)
    # 2. Deep (Dim 64, Depth 6)
    # 3. Wide (Dim 128, Depth 2)
    # 4. Many Heads (Dim 64, Heads 8)
    
    grid = {
        'depth': [2, 6],
        'dim': [64, 128],
        'heads': [4, 8]
    }
    
    # We want specific comparisons, not valid full product (too many).
    # Let's manual list useful ones for "Grid" API
    # Or just run full grid? 2x2x2 = 8 runs. Feasible.
    
    runner.run_grid(base_config, grid, SortingTask)

if __name__ == "__main__":
    run()
