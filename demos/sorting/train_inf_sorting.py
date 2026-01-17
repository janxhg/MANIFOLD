import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import sys
from pathlib import Path
import yaml
import argparse
import time
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model import Manifold
from src.optim import RiemannianAdam
from src.embeddings import FunctionalEmbedding # Need this for coordinate generation

class CausalSortingDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.SEP = vocab_size     # Token for separator
        self.EOS = vocab_size + 1 # Token for end
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate random sequence
        vals = [random.randint(0, self.vocab_size - 1) for _ in range(self.seq_len)]
        sorted_vals = sorted(vals)
        
        # Input: [vals] [SEP] [sorted]
        # Target: [vals] [SEP] [sorted] [EOS]
        full_seq = vals + [self.SEP] + sorted_vals + [self.EOS]
        
        src = torch.tensor(full_seq[:-1], dtype=torch.long)
        tgt = torch.tensor(full_seq[1:], dtype=torch.long)
        
        return src, tgt

def get_functional_coords(token_ids, coord_dim, device):
    """
    Reverse-engineers the FunctionalEmbedding logic to get the Target Coordinates.
    Same logic as src/embeddings.py FunctionalEmbedding.forward
    """
    # 1. Frequency Initialization (Must match Manual Init)
    # We should ideally expose this from the model, but for now we replicate logic
    # freqs = exp(-log(10000) * i / d)
    freqs = torch.exp(torch.arange(0, coord_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / coord_dim)).to(device)
    
    # 2. Compute Coords
    x = token_ids.unsqueeze(-1).float()
    args = x * freqs
    coords = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    return coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demos/sorting.yaml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    vocab_range = config['task']['vocab']
    real_vocab_size = vocab_range + 2 # + SEP, EOS
    config['model']['vocab_size'] = real_vocab_size
    
    # Dataset
    train_ds = CausalSortingDataset(config['task']['num_train'], config['task']['seq_len'], vocab_range)
    val_ds = CausalSortingDataset(config['task']['num_val'], config['task']['seq_len'], vocab_range)
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], num_workers=0)
    
    # Model
    model = Manifold(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        integrator_type=config['physics']['solver'],
        use_scan=config['model']['use_scan'],
        physics_config=config['physics']
    ).to(device)
    
    # Detect Infinite Mode
    is_infinite = (config['physics']['readout']['type'] == 'implicit')
    coord_dim = config['physics']['embedding']['coord_dim']
    
    if is_infinite:
        print(f"\n[*] INFINITE MODE DETECTED")
        print(f"    - Input: Functional (O(1))")
        print(f"    - Output: Implicit Coordinate Regression (O(1))")
        print(f"    - Coord Dim: {coord_dim}")
        
    # Params
    total = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total/1e6:.2f}M\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    if is_infinite:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    best_acc = 0.0
    
    # Pre-compute valid coordinate vectors for KNN decoding (Validation only)
    if is_infinite:
        print("[*] Pre-computing vocab coordinates for validation decoding...")
        all_ids = torch.arange(real_vocab_size).to(device)
        vocab_coords = get_functional_coords(all_ids, coord_dim, device) # [Vocab, Coord]
        
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            # Forward
            # Logits are [Batch, Seq, OutDim]
            # If Infinite: OutDim = CoordDim
            # If Standard: OutDim = VocabSize
            pred, _, _ = model(src)
            
            if is_infinite:
                # Target: Coordinates of Tgt IDs
                tgt_coords = get_functional_coords(tgt, coord_dim, device)
                loss = criterion(pred, tgt_coords)
            else:
                loss = criterion(pred.reshape(-1, real_vocab_size), tgt.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                pred, _, _ = model(src)
                
                if is_infinite:
                    # Decode Coords -> IDs via Nearest Neighbor
                    # Pred: [B, L, C]
                    # Vocab: [V, C]
                    # Dist: [B, L, V]
                    # We can use cdist or simple dot product if normalized? They are Sins, so normalized-ish.
                    # Let's use negative L2 distance
                    
                    B, L, C = pred.shape
                    flat_pred = pred.view(-1, C) # [B*L, C]
                    
                    # Compute distances to all vocab words
                    dists = torch.cdist(flat_pred, vocab_coords) # [B*L, V]
                    flat_ids = torch.argmin(dists, dim=-1) # [B*L] Best match ID
                    
                    decoded_ids = flat_ids.view(B, L)
                    preds = decoded_ids
                else:
                    preds = torch.argmax(pred, dim=-1)
                    
                # Check Sorted Part (Logic reuse)
                seq_len = config['task']['seq_len']
                start_check = seq_len 
                
                sorted_preds = preds[:, start_check : start_check + seq_len]
                sorted_tgts = tgt[:, start_check : start_check + seq_len]
                
                row_matches = torch.all(sorted_preds == sorted_tgts, dim=1)
                correct += row_matches.sum().item()
                total_samples += src.size(0)
                
        val_acc = correct / total_samples
        curr_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | LR: {curr_lr:.2e}")
        
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            print("--> New Best Checkpoint")

if __name__ == '__main__':
    main()
