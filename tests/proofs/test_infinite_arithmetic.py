import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from gfn.model import Manifold

class AdditionDataset(Dataset):
    def __init__(self, num_samples, length, base=10):
        self.num_samples = num_samples
        self.length = length
        self.base = base
        # Vocab: 0-9 (digits), 10 (+), 11 (=), 12 (pad/eos)
        self.PLUS = 10
        self.EQ = 11
        self.EOS = 12
        self.vocab_size = 13
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate two numbers of random length up to self.length
        # To force "carry" logic, we ensure high probability of large digits
        len1 = np.random.randint(1, self.length + 1)
        len2 = np.random.randint(1, self.length + 1)
        
        num1 = [np.random.randint(0, self.base) for _ in range(len1)]
        num2 = [np.random.randint(0, self.base) for _ in range(len2)]
        
        # Calculate sum
        val1 = int("".join(map(str, num1)))
        val2 = int("".join(map(str, num2)))
        res = val1 + val2
        res_digits = [int(d) for d in str(res)]
        
        # Sequence: [num1] + [num2] = [res]
        # Input: digits1 + PLUS + digits2 + EQ
        # Target: res_digits + EOS
        
        src = torch.tensor(num1 + [self.PLUS] + num2 + [self.EQ], dtype=torch.long)
        tgt = torch.tensor(res_digits + [self.EOS], dtype=torch.long)
        
        return src, tgt

def collate_fn(batch):
    # Dynamic padding
    srcs, tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    
    pad_idx = 12
    
    padded_src = torch.full((len(srcs), max_src), pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(srcs), max_tgt), pad_idx, dtype=torch.long)
    
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        padded_src[i, :len(s)] = s
        padded_tgt[i, :len(t)] = t
        
    return padded_src, padded_tgt

def train_and_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Infinite Arithmetic Test (Device: {device})")
    
    # 1. Train on Small Length (e.g., 5-10 digits)
    TRAIN_LEN = 10
    print(f"[*] Training on numbers up to {TRAIN_LEN} digits...")
    
    train_ds = AdditionDataset(5000, TRAIN_LEN)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    model = Manifold(
        vocab_size=13,
        dim=64,
        depth=2,
        heads=2,
        integrator_type='leapfrog',
        physics_config={'active_inference': {'enabled': True}} # Enable "thinking"
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=12)
    
    # Training Loop
    EPOCHS = 5
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            # Autoregressive training on target?
            # Manifold is recurrent. We accept src and predict tgt?
            # For simplicity in this demo, let's treat it as Seq2Seq with single forward?
            # Manifold forward(x) returns logits for next token.
            # Here we have [Input] -> [Output]. 
            # Ideally we feed [Input] and see if it generates [Output].
            # Training: standard next-token prediction on full sequence?
            # Let's concatenate: SRC + TGT. Train to predict TGT part.
            
            full_seq = torch.cat([src, tgt], dim=1) # [B, S+T]
            input_seq = full_seq[:, :-1]
            target_seq = full_seq[:, 1:]
            
            logits, _, _ = model(input_seq)
            
            # Only calculate loss on the Answer part
            # We need to mask out the Question part logic or just let it learn everything
            # Let's learn everything for simplicity
            loss = criterion(logits.reshape(-1, 13), target_seq.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    # 2. Test on Huge Length (e.g., 50 digits)
    # Note: 50 digits is "Infinite" compared to 10 for a Transformer
    TEST_LEN = 50
    print(f"\n[*] Testing generalization on {TEST_LEN} digits (5x training length)...")
    
    test_ds = AdditionDataset(100, TEST_LEN)
    
    model.eval()
    correct = 0
    total = 0
    
    for i in range(10): # Test 10 samples qualitatively
        src, tgt = test_ds[i]
        src = src.unsqueeze(0).to(device) # [1, L]
        tgt_len = len(tgt)
        
        # Generation
        curr_seq = src
        generated = []
        
        with torch.no_grad():
            state = None
            # Pre-fill context
            # Optimally we should run forward once on src to get state, then loop.
            # Manifold supports this via state passing.
            
            # 1. Warmup state on Question
            _, state, _ = model(src)
            
            # 2. Generate Answer
            input_token = curr_seq[:, -1:] # Last token (EQ)
            
            for _ in range(tgt_len):
                logits, state, _ = model(input_token, initial_state=state)
                next_token = torch.argmax(logits, dim=-1)
                generated.append(next_token.item())
                input_token = next_token
                
                if next_token.item() == 12: # EOS
                    break
        
        # Decode
        src_str = "".join([str(d.item()) for d in src[0] if d < 10])
        # Need to split by plus... simple parsing for display
        
        gen_str = "".join([str(d) for d in generated if d < 10])
        tgt_str = "".join([str(d.item()) for d in tgt if d < 10])
        
        is_correct = (gen_str == tgt_str)
        if is_correct: correct += 1
        total += 1
        
        print(f"Sample {i}: {src_str[:15]}... = {tgt_str[:10]}... | Pred: {gen_str[:10]}... [{'✅' if is_correct else '❌'}]")
        
    print(f"\nFinal Generalization Accuracy on {TEST_LEN}-digit sums: {correct}/{total}")

if __name__ == "__main__":
    train_and_eval()
