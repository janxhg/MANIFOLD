
import torch
from gfn.geometry import ToroidalChristoffel

def test_toroidal_differentiability():
    print("Testing Toroidal Differentiability...")
    dim = 4
    christ = ToroidalChristoffel(dim)
    
    # Create input that would trigger the old torch.sign/clamp boundary
    # R=2, r=1. cos(th) = -1 => th = PI
    x = torch.tensor([[3.14159, 0.0, 0.0, 0.0]], requires_grad=True)
    v = torch.ones(1, dim)
    
    try:
        gamma = christ(v, x)
        loss = gamma.pow(2).sum()
        loss.backward()
        
        print(f"[PASS] Gradients computed successfully. x.grad: {x.grad}")
        if torch.isnan(x.grad).any():
            print("[FAIL] Gradients contain NaNs!")
        else:
            print("[PASS] Gradients are clean (no NaNs).")
            
    except Exception as e:
        print(f"[FAIL] Differentiability test crashed: {e}")

if __name__ == "__main__":
    test_toroidal_differentiability()
