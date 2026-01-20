
import torch
import sys
from pathlib import Path

ckpt_path = r""

try:
    state_dict = torch.load(ckpt_path, map_location='cpu')
    print(f"Keys: {list(state_dict.keys())[:5]} ...")
    
    if 'model_config' in state_dict:
        print(f"CONFIG: {state_dict['model_config']}")
    else:
        print("No model_config found in checkpoint.")


except Exception as e:
    print(f"Error: {e}")
