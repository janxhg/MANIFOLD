import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random

# Add project root to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from gfn.model import Manifold
from gfn.optim import RiemannianAdam
import yaml
import argparse

class CopyTaskDataset(Dataset):
    """
    Copy Task: Model must copy a sequence of random digits
    
    Format: [input sequence] <sep> [target sequence]
    Example: 1 4 7 2 <sep> 1 4 7 2 <eos>
    
    This tests the model's ability to:
    1. Remember arbitrary sequences
    2. Reproduce them exactly
    3. Handle variable-length inputs
    """
    def __init__(self, num_samples, min_len, max_len, vocab_size=10):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Special tokens
        self.SEP = vocab_size      # Separator between input and output
        self.EOS = vocab_size + 1  # End of sequence
        
        # Generate all samples upfront
        print(f"Generating {num_samples} copy task samples (len {min_len}-{max_len})...")
        self.samples = self._generate_samples()
        
    def _generate_samples(self):
        samples = []
        for _ in range(self.num_samples):
            # Random sequence length
            seq_len = random.randint(self.min_len, self.max_len)
            
            # Random sequence of digits
            sequence = [random.randint(0, self.vocab_size - 1) for _ in range(seq_len)]
            
            # Format: input = [seq] <sep> [seq], target = shifted by 1
            full_seq = sequence + [self.SEP] + sequence + [self.EOS]
            
            # Autoregressive: predict next token
            input_seq = full_seq[:-1]
            target_seq = full_seq[1:]
            
            samples.append((input_seq, target_seq))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        
        # Convert to tensors
        src = torch.tensor(input_seq, dtype=torch.long)
        tgt = torch.tensor(target_seq, dtype=torch.long)
        
        return src, tgt

def collate_fn(batch):
    """Pad sequences to same length in batch"""
    srcs, tgts = zip(*batch)
    
    # Find max length (src and tgt should already be same length)
    max_len = max(len(s) for s in srcs)
    
    # Pad sequences
    srcs_padded = []
    tgts_padded = []
    
    for src, tgt in zip(srcs, tgts):
        pad_len = max_len - len(src)
        srcs_padded.append(torch.cat([src, torch.zeros(pad_len, dtype=torch.long)]))
        tgts_padded.append(torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)]))
    
    return torch.stack(srcs_padded), torch.stack(tgts_padded)

def evaluate_accuracy(model, val_loader, device, vocab_size):
    """Calculate exact sequence match accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits, _, _ = model(src)
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Check exact match (ignoring padding)
            for pred, target in zip(preds, tgt):
                # Find first EOS token
                eos_idx = (target == vocab_size + 1).nonzero(as_tuple=True)[0]
                if len(eos_idx) > 0:
                    pred = pred[:eos_idx[0]]
                    target = target[:eos_idx[0]]
                
                if torch.equal(pred, target):
                    correct += 1
                total += 1
    
    return correct / total if total > 0 else 0.0

def train_copy_task():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/demos/copy_task.yaml')
    args = parser.parse_args()
    
    # Load Config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create datasets
    train_dataset = CopyTaskDataset(
        num_samples=config['task']['num_train'],
        min_len=config['task']['min_length'],
        max_len=config['task']['max_length'],
        vocab_size=10
    )
    
    val_dataset = CopyTaskDataset(
        num_samples=config['task']['num_val'],
        min_len=config['task']['min_length'],
        max_len=config['task']['max_length'],
        vocab_size=10
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        collate_fn=collate_fn
    )
    
    # Initialize Model
    model = Manifold(
        vocab_size=config['model']['vocab_size'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        integrator_type=config['physics']['solver'],
        use_scan=config['model']['use_scan'],
        physics_config=config['physics']
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = RiemannianAdam(model.parameters(), lr=config['training']['lr'])
    
    # Cosine Annealing LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=1e-6
    )
    
    # Training Loop
    print("Starting Copy Task Training...\n")
    best_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        for i, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            logits, _, _ = model(src)
            loss = criterion(logits.view(-1, config['model']['vocab_size']), tgt.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % config['training']['log_every'] == 0:
                cur_loss = total_loss / (i + 1)
                print(f"| Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {cur_loss:.4f}")
        
        # Validation
        val_acc = evaluate_accuracy(model, val_loader, device, 10)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"| End of Epoch {epoch+1} | Val Accuracy: {val_acc*100:.2f}% | LR: {current_lr:.2e}")
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        os.makedirs(config['training']['save_dir'], exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, f"{config['training']['save_dir']}/checkpoint_epoch_{epoch+1}.pt")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, f"{config['training']['save_dir']}/best_model.pt")
            print(f"✓ New best model saved (Accuracy: {val_acc*100:.2f}%)\n")

if __name__ == '__main__':
    train_copy_task()
