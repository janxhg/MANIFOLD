import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam
import yaml
import argparse

class SortingDataset(Dataset):
    """
    Sorting Task: Input random sequence, Output sorted sequence.
    Example: [5, 1, 8] -> [1, 5, 8]
    
    Why this tests Geometry:
    Sorting requires comparing elements (quadratic interaction).
    Manifold's Christoffel symbols Γ(v,v) provide exactly this quadratic structure.
    Linear models (SSMs without selection) struggle with this.
    """
    def __init__(self, num_samples, seq_len, vocab_size=100):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        print(f"Generating {num_samples} sorting samples (len {seq_len}, range 0-{vocab_size-1})...")
        self.samples = self._generate_samples()
        
    def _generate_samples(self):
        samples = []
        for _ in range(self.num_samples):
            # Random sequence of numbers
            # We use vocab_size - 2 to reserve space for special tokens if needed (not used here yet)
            vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
            
            input_seq = torch.tensor(vals, dtype=torch.long)
            target_seq = torch.tensor(sorted(vals), dtype=torch.long)
            
            samples.append((input_seq, target_seq))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def evaluate_accuracy(model, val_loader, device):
    """Calculate exact full-sequence match accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            # For this simple task, we can just run forward pass
            # Note: Ideally we run autoregressive generation for sorting, 
            # but to test 'reasoning' capacity we can treat it as pos-to-pos mapping 
            # or seq-to-seq.
            
            # Manifold is casual (autoregressive). 
            # So we feed: [Input Seq] <SEP> [Sorted Seq (shifted)]
            # But let's simplify for the "Geometric Capability Test":
            # Can it map Input -> Output directly? 
            # No, Manifold is causal. It needs to read the input first.
            
            # Revised approach for Causal Model:
            # Input: [Numbers...] <SEP>
            # Target is Autoregressive generation of sorted list.
            
            pass # (Evaluated inside training loop with proper setup)
            
    return 0.0

# Redefining dataset to be Causal-friendly
class CausalSortingDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.SEP = vocab_size     # Token for separator
        self.EOS = vocab_size + 1 # Token for end
        
        # Actual vocab size for model = range + 2 special
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random sequence
        vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
        sorted_vals = sorted(vals)
        
        # Input: [vals] [SEP] [sorted]
        # Target: [vals] [SEP] [sorted] [EOS] (Shifted by 1 in training)
        
        full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
        
        src = torch.tensor(full_seq[:-1], dtype=torch.long)
        tgt = torch.tensor(full_seq[1:], dtype=torch.long)
        
        return src, tgt

def train_sorting():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GEOMETRY MODE: {'SCAN (Linearized)' if config['model']['use_scan'] else 'FULL GEODESIC (Non-linear)'}")
    
    vocab_range = config['task']['vocab']
    real_vocab_size = vocab_range + 2 # + SEP, EOS
    
    # Update model config with real vocab size
    config['model']['vocab_size'] = real_vocab_size
    
    train_ds = CausalSortingDataset(config['task']['num_train'], config['task']['seq_len'], vocab_range)
    val_ds = CausalSortingDataset(config['task']['num_val'], config['task']['seq_len'], vocab_range)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])
    
    model = Manifold(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        integrator_type=config['physics']['solver'],
        use_scan=config['model']['use_scan'],
        physics_config=config['physics']
    ).to(device)
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
    
    criterion = nn.CrossEntropyLoss()
    # SWITCH TO ADAMW (Standard stable optimizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    best_acc = 0.0
    SEP = vocab_range
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            logits, _, _ = model(src)
            loss = criterion(logits.view(-1, real_vocab_size), tgt.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            total_loss += loss.item()
            
            if i % config['training']['log_every'] == 0:
                print(f"| Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")
                
        # Validation (Exact Match on Sorted Part)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                logits, _, _ = model(src)
                preds = torch.argmax(logits, dim=-1)
                
                # Check only the sorted part (after SEP)
                # Find SEP in input
                # Since we know fixed length for this demo:
                # [Input 10] [SEP 1] [Sorted 10] [EOS 1]
                # Indices: 0-9 input, 10 sep, 11-20 sorted
                
                seq_len = config['task']['seq_len']
                sorted_preds = preds[:, seq_len + 1 : seq_len + 1 + seq_len]
                sorted_tgts = tgt[:, seq_len + 1 : seq_len + 1 + seq_len]
                
                # Matches row-wise
                row_matches = torch.all(sorted_preds == sorted_tgts, dim=1)
                correct += row_matches.sum().item()
                total += src.size(0)
                
        val_acc = correct / total
        print(f"| End of Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Validation Accuracy: {val_acc*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(config['training']['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), f"{config['training']['save_dir']}/best_model.pt")
            print(f"✓ New Best Model Saved!")

if __name__ == '__main__':
    train_sorting()
