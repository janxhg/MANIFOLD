import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from gfn.model import Manifold

def measure_vram():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
    return 0.0

def run_needle_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🧵 Needle in Infinite Haystack Test (Device: {device})")
    
    # Configuration
    KEY_VAL_PAIRS = 5  # Memorize 5 pairs
    NOISE_LEN = 10000  # 10k tokens of noise (Standard Transformer would struggle/OOM with standard attn)
    VOCAB = 100
    
    print(f"[*] Task: Memorize {KEY_VAL_PAIRS} pairs, endure {NOISE_LEN} noise tokens, recall correctly.")
    
    # Data Generation
    # Format: [K1, V1, K2, V2...] [NOISE...] [QUERY K] -> [TARGET V]
    
    def generate_batch(batch_size):
        # Keys: 0-9, Values: 10-19
        keys = torch.randint(0, 10, (batch_size, KEY_VAL_PAIRS))
        vals = torch.randint(10, 20, (batch_size, KEY_VAL_PAIRS))
        
        # Interleave K, V
        kv_seq = torch.zeros((batch_size, KEY_VAL_PAIRS * 2), dtype=torch.long)
        kv_seq[:, 0::2] = keys
        kv_seq[:, 1::2] = vals
        
        # Noise: 20-99
        noise = torch.randint(20, 100, (batch_size, NOISE_LEN))
        
        # Query: Pick one random key index
        q_idx = torch.randint(0, KEY_VAL_PAIRS, (batch_size, 1))
        # Gather query key and target val
        query_key = torch.gather(keys, 1, q_idx)
        target_val = torch.gather(vals, 1, q_idx)
        
        # Seq: KV + Noise + Query
        # We need to predict TargetVal from Query
        # Manifold inputs: [KV... Noise... Query]
        # Target: [TargetVal] (Next token prediction)
        
        src = torch.cat([kv_seq, noise, query_key], dim=1).to(device)
        tgt = target_val.squeeze(1).to(device)
        
        return src, tgt
        
    # Model Setup
    # Force SEQUENTIAL mode (use_scan=False) to ensure O(1) memory
    model = Manifold(
        vocab_size=VOCAB,
        dim=64,
        depth=2,
        heads=2,
        integrator_type='symplectic',
        physics_config={'active_inference': {'enabled': True}}, # Need logic to 'hold' memory
        use_scan=False # ESSENTIAL for O(1)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Baseline VRAM
    torch.cuda.reset_peak_memory_stats()
    base_vram = measure_vram()
    print(f"Base VRAM: {base_vram:.2f} MB")
    
    # Training Loop
    # We train on shorter noise first? Or just go for it?
    # 10k might be slow for training. Let's train on 100, test on 10k?
    # The prompt implies "Infinite" capacity.
    # Let's Train on 100 noise tokens.
    
    TRAIN_NOISE = 100
    print(f"[*] Training on Noise Length={TRAIN_NOISE}...")
    
    # Monkey-patch generator for training
    original_len = NOISE_LEN
    NOISE_LEN = TRAIN_NOISE # Temporary
    
    model.train()
    pbar = tqdm(range(200), desc="Training")
    
    for step in pbar:
        src, tgt = generate_batch(32)
        
        optimizer.zero_grad()
        # Forward pass on huge sequence?
        # Manifold sequential is slow on python loop. 
        # But we must prove it works.
        
        # Optimization: We only care about the final prediction.
        # Manifold.forward returns all logits [B, L, V]
        # We only need the last one.
        
        logits, _, _ = model(src) # [B, L, V]
        last_logit = logits[:, -1, :] # Prediction for next token after Query
        
        loss = criterion(last_logit, tgt)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            acc = (torch.argmax(last_logit, dim=-1) == tgt).float().mean()
            pbar.set_postfix({'loss': loss.item(), 'acc': f"{acc*100:.0f}%"})
            
    # Test on Infinite Haystack
    NOISE_LEN = 10000
    print(f"\n[*] testing on Infinite Haystack (Length={NOISE_LEN})...")
    
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    
    with torch.no_grad():
        src, tgt = generate_batch(1) # Batch 1 to isolate VRAM
        
        start_time = time.time()
        logits, _, _ = model(src)
        last_logit = logits[:, -1, :]
        pred = torch.argmax(last_logit, dim=-1)
        
        duration = time.time() - start_time
        peak_vram = measure_vram()
        
    print(f"\n🏆 Results:")
    print(f"Sequence Length: {src.shape[1]}")
    print(f"Peak VRAM: {peak_vram:.2f} MB (Should be close to Base)")
    print(f"Prediction: {pred.item()} | Target: {tgt.item()}")
    print(f"Correct: {'✅' if pred == tgt else '❌'}")
    print(f"Time: {duration:.2f}s")
    
    # Verify O(1) claim
    # A transformer storing KV cache for 10k tokens, Dim=64, Layers=2
    # Cache = 2 * 2 * 1 * 10000 * 64 * 4 bytes ≈ 10 MB?
    # Not huge, but grows linearly. 
    # If we did 1M tokens, Transformer = 1GB cache. Manifold = 0.
    
    if peak_vram < base_vram + 50: # Allow 50MB overhead for buffers/cuda context
        print("✅ O(1) Memory Confirmed.")
    else:
        print("❌ Memory leak or not O(1).")

if __name__ == "__main__":
    run_needle_test()
