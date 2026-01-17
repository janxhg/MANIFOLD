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
    runner = ExperimentRunner(experiment_name="Loss_Components")
    
    base_config = {
        'max_steps': 1500,
        'dim': 64, 
        'depth': 4,
        'heads': 4,
        'use_scan': False
    }
    
    # Note: Currently Manifold needs to expose traces to support these.
    # Assuming code will be updated to support traces, OR testing placebo effect.
    # Actually, if loss is effectively computed (model.py updated), this works.
    
    grid = {
        'lambda_h': [0.0, 0.01, 0.1],  # Hamiltonian
        'lambda_c': [0.0, 0.05],       # Curiosity
    }
    
    runner.run_grid(base_config, grid, SortingTask)

if __name__ == "__main__":
    run()
