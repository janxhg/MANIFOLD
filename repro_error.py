import torch
from gfn.model import Manifold

def reproduce_error():
    print("Starting reproduction...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = 64
    physics_config = {
        'topology': {'type': 'torus'},
        'singularities': {'enabled': True, 'strength': 5.0, 'threshold': 0.8},
        'active_inference': {'enabled': True, 'plasticity': 0.1}
    }
    
    # Init model
    model = Manifold(vocab_size=10, dim=dim, depth=2, heads=2, holographic=True, physics_config=physics_config).to(device)
    
    # Input
    x = torch.randint(0, 10, (1, 5), device=device) # Batch 1, Seq 5
    
    print("Running forward...")
    try:
        logits, state, reg, _, x_seq, forces = model(x)
        print("Forward SUCCESS")
        
        # Check backward
        print("Running backward...")
        loss = logits.sum()
        loss.backward()
        print("Backward SUCCESS")
        
    except Exception as e:
        print(f"\n[REPRO:ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce_error()
