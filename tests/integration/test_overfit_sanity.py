import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import Manifold
from src.losses import GFNLoss
from src.optim import RiemannianAdam

def test_overfit_rtx_config():
    print("=== DIAGNOSIS: OVERFIT SINGLE BATCH (RTX CONFIG) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # RTX Configuration (Large)
    vocab_size = 16
    dim = 256 # Reduced slightly for speed in test, but deep enough to break if broken
    depth = 12
    rank = 64
    heads = 4
    
    print(f"Initializing Model (dim={dim}, depth={depth}, heads={heads})...")
    model = Manifold(vocab_size, dim, depth, rank, heads=heads, integrator_type='leapfrog').to(device)
    optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
    criterion = GFNLoss(lambda_h=0.01)
    
    # Fixed Batch (Overfit Target)
    # "1+1=2"
    x = torch.tensor([[1, 10, 1, 13, 2, 15] for _ in range(4)]).to(device) # Batch 4
    # Target: Shifted by 1
    # Input:  1 + 1 = 2 <EOS>
    # Target: + 1 = 2 <EOS> <PAD>
    # Actually, GFN forward handles masking usually.
    # Let's just assume we want to predict next token.
    targets = x.clone() # Simple Auto-regressive loss self-supervision
    
    print("Starting Training Loop (100 steps)...")
    model.train()
    
    initial_loss = None
    
    for i in range(100):
        optimizer.zero_grad()
        logits, _ = model(x)
        
        # Shift logits and targets
        # Logits: [B, T, V] -> Predict t+1
        # Target: [B, T] -> t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        
        loss, _ = criterion(shift_logits, shift_targets)
        loss.backward()
        
        # Clip grad to prevent explosion in deep net
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if i == 0: initial_loss = loss.item()
        if i % 10 == 0:
            print(f"Step {i}: Loss {loss.item():.4f}")
            
    print(f"Final Loss: {loss.item():.4f}")
    
    if loss.item() < 0.1:
        print("[SUCCESS] Model can memorize (Gradients are working).")
        print("--> Conclusion: The 69-epoch failure was Hyperparameters (Batch Size/LR), not Architecture.")
    else:
        print("❌ FAILURE: Model cannot even memorize 1 example.")
        print("--> Conclusion: Gradients are broken or Architecture is unstable.")

if __name__ == "__main__":
    test_overfit_rtx_config()
