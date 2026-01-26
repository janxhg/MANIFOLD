
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

with open("diag_out.txt", "w") as f:
    try:
        from gfn.cuda import gfn_cuda
        f.write("[SUCCESS] Imported gfn_cuda\n")
    except ImportError as e:
        f.write(f"[FAIL] Could not import gfn_cuda: {e}\n")
        sys.exit(1)

    f.write("\n--- Docstring for recurrent_manifold_fused ---\n")
    f.write(str(gfn_cuda.recurrent_manifold_fused.__doc__) + "\n")

    f.write("\n--- Attempting Dummy Call (Named Args) ---\n")
    try:
        # helper to create tensor
        def t(shape): return torch.zeros(shape, device='cuda', dtype=torch.float32)

        kwargs = {
            "x_state": t((1,1)), 
            "v_state": t((1,1)), 
            "forces": t((1,1,1)), # Forces usually B, L, D
            "U_stack": t((1,1,1)), # L*H, D, R
            "W_stack": t((1,1,1)),
            "dt": 0.1,
            "dt_scales": t((1,)),
            "forget_rates": t((1,)),
            "num_heads": 1,
            "plasticity": 0.0,
            "sing_thresh": 1.0,
            "sing_strength": 1.0,
            "mix_x": t((0,)),
            "mix_v": t((0,)),
            "W_forget_stack": t((0,)),
            "W_input_stack": t((0,)),
            "b_forget_stack": t((0,)),
            "W_potential_stack": t((0,)),
            "b_potential_stack": t((0,)),
            "topology": 0,
            "R": 2.0,
            "r": 1.0
        }
        
        f.write(f"Calling with {len(kwargs)} named args\n")
        ret = gfn_cuda.recurrent_manifold_fused(**kwargs)
        f.write("[SUCCESS] Keyword Call worked!\n")
    except Exception as e:
        f.write(f"[FAIL] Keyword Call failed: {e}\n")
