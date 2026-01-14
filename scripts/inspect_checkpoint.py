import torch
import sys
import os

def inspect_checkpoint(path):
    print(f"=== INSPECTING: {path} ===")
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    try:
        # Load safe (weights_only=False because older pytorch versions might pickle stuff)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        state_dict = ckpt
        if 'model_state_dict' in ckpt:
            print(f"✅ Found metadata (Epoch: {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?')})")
            state_dict = ckpt['model_state_dict']
        else:
            print("⚠️ Raw state_dict (No metadata)")
            
        keys = list(state_dict.keys())
        print(f"Total Keys: {len(keys)}")
        
        # Infer Architecture
        if 'embedding.weight' in state_dict:
            vocab, dim = state_dict['embedding.weight'].shape
            print(f"Observed Embedding: Vocab={vocab}, Dim={dim}")
        
        # Infer Depth
        layer_indices = sorted(list(set([int(k.split('.')[1]) for k in keys if k.startswith('layers.')])))
        if layer_indices:
            depth = max(layer_indices) + 1
            print(f"Observed Depth: {depth}")
            
            # Infer Rank from first layer
            # layers.0.christoffel.U
            if f'layers.0.christoffel.U' in state_dict:
                rank = state_dict['layers.0.christoffel.U'].shape[1]
                print(f"Observed Rank: {rank}")
        
        print("\nStructure Guess:")
        print(f"dim={dim}, depth={depth}, rank={rank}, vocab_size={vocab}")
        
    except Exception as e:
        print(f"❌ Error loading: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        print("Usage: python inspect_checkpoint.py <path>")
