
import torch
from gfn.losses import GFNLoss

def test_hamiltonian_loss_args():
    print("Testing GFNLoss Hamiltonian integration...")
    loss_fn = GFNLoss(lambda_h=0.1)
    
    B, S, D = 2, 4, 16
    logits = torch.randn(B, S, 10)
    targets = torch.randint(0, 10, (B, S))
    
    # 5 steps of velocities
    velocities = [torch.randn(B, D) for _ in range(5)]
    
    try:
        total_loss, loss_dict = loss_fn(logits, targets, velocities=velocities)
        print(f"[PASS] GFNLoss executed without errors. Total: {total_loss.item():.4f}")
        if "hamiltonian" in loss_dict:
            print(f"[PASS] Hamiltonian loss component: {loss_dict['hamiltonian']:.4f}")
        else:
            print("[FAIL] Hamiltonian loss component missing from dict!")
    except Exception as e:
        print(f"[FAIL] GFNLoss crashed: {e}")

if __name__ == "__main__":
    test_hamiltonian_loss_args()
