import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from pathlib import Path
from tqdm import tqdm
import random

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold

class SortingTask:
    def __init__(self, vocab_size=100, length=10):
        self.vocab_size = vocab_size
        self.length = length
        self.SEP = vocab_size
        self.EOS = vocab_size + 1
        self.full_vocab = vocab_size + 2
        
    def generate_batch(self, batch_size, device='cpu'):
        # Input: Random numbers
        # shape: [batch, length]
        x_raw = torch.randint(0, self.vocab_size, (batch_size, self.length), device=device)
        
        # Sort them
        y_raw, _ = torch.sort(x_raw, dim=1)
        
        # Create Causal Sequence: [Input] [SEP] [Output]
        # We need to construct the full tensor
        
        # 1. Input part
        inputs = x_raw
        
        # 2. SEP token
        sep = torch.full((batch_size, 1), self.SEP, device=device)
        
        # 3. Output part (Sorted)
        outputs = y_raw
        
        # 4. EOS token
        eos = torch.full((batch_size, 1), self.EOS, device=device)
        
        # Full concatenated sequence: [Input] [SEP] [Output] [EOS]
        full_seq = torch.cat([inputs, sep, outputs, eos], dim=1)
        
        # Src: [:-1], Tgt: [1:]
        src = full_seq[:, :-1]
        tgt = full_seq[:, 1:]
        
        return src, tgt

def train_until_convergence(model, task, max_steps=5000, lr=1e-3, device='cuda'):
    # Use standard AdamW as in the successful benchmark
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    pbar = tqdm(range(max_steps), desc=f"Training Manifold on Sorting")
    normalized_loss = 5.0 # Init high
    
    for i in pbar:
        x, y = task.generate_batch(64, device=device)
        
        optimizer.zero_grad()
        logits, _, _ = model(x)
            
        loss = criterion(logits.reshape(-1, task.full_vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        loss_val = loss.item()
        scheduler.step(loss_val)
        
        # Exponential moving average
        normalized_loss = 0.95 * normalized_loss + 0.05 * loss_val
        
        if i % 10 == 0:
             # Calculate validation accuracy (exact match on sorted part)
             with torch.no_grad():
                 preds = torch.argmax(logits, dim=-1)
                 # Indices of sorted part: length+1 to end
                 start_idx = task.length + 1
                 pred_sort = preds[:, start_idx:]
                 true_sort = y[:, start_idx:]
                 
                 # Correct order: compare -> reduce columns -> convert to float -> mean
                 correct = (pred_sort == true_sort).all(dim=1).float().mean().item()
                 
             pbar.set_postfix({
                 'loss': f"{normalized_loss:.4f}", 
                 'acc': f"{correct*100:.1f}%",
                 'lr': f"{optimizer.param_groups[0]['lr']:.5f}"
             })
        
    return

def run_sorting_v2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config from successful Parity benchmark
    dim = 64
    depth = 2
    heads = 4
    vocab_range = 100
    
    # Total vocab = range + SEP + EOS
    total_vocab = vocab_range + 2
    
    print("Initializing Manifold (Parity Config)...")
    model = Manifold(
        vocab_size=total_vocab, 
        dim=dim, 
        depth=depth, 
        heads=heads, 
        rank=16,
        use_scan=False, # Sequential for stability
        # Same physics config as Parity
        physics_config={'active_inference': {'enabled': True, 'reactive_curvature': {'enabled': True, 'plasticity': 0.05}}}
    ).to(device)
    
    task = SortingTask(vocab_size=vocab_range, length=10)
    
    print("Starting Training...")
    train_until_convergence(model, task, max_steps=5000, lr=3e-3, device=device)

if __name__ == "__main__":
    run_sorting_v2()
