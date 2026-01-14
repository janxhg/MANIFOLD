import torch
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GFN
from src.adjoint import AdjointGFN

def check_gradients():
    print("=== STABILITY: GRADIENT CHECK ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Config
    # Uses Deep model to check for Vanishing Gradients
    vocab = 10
    dim = 64
    depth = 24 # Deep!
    rank = 16
    bs = 4
    seq = 32
    
    print(f"Initializing Deep GFN (Depth={depth})...")
    model = GFN(vocab, dim, depth, rank, integrator_type='leapfrog').to(device)
    
    x = torch.randint(0, vocab, (bs, seq)).to(device)
    target = torch.randint(0, vocab, (bs, seq)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    print("Forward Pass...")
    logits, _ = model(x)
    loss = criterion(logits.view(-1, vocab), target.view(-1))
    
    print("Backward Pass...")
    loss.backward()
    
    print("\nGradient Analysis:")
    has_grad = False
    min_grad = float('inf')
    max_grad = 0.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            min_grad = min(min_grad, g_norm)
            max_grad = max(max_grad, g_norm)
            has_grad = True
            
            if torch.isnan(param.grad).any():
                print(f"[FAIL] NaN Gradient in {name}")
                return
            if g_norm == 0.0:
                print(f"[WARN] Zero Gradient in {name}")
                
    if not has_grad:
        print("[FAIL] No gradients computed!")
        return

    print(f"    Min Grad Norm: {min_grad:.2e}")
    print(f"    Max Grad Norm: {max_grad:.2e}")
    
    if min_grad < 1e-9:
        print("[WARN] Potential Vanishing Gradients")
    elif max_grad > 100.0:
        print("[WARN] Potential Exploding Gradients")
    else:
        print("[PASS] Gradients are healthy (Deep Network Flowing).")

if __name__ == "__main__":
    check_gradients()
