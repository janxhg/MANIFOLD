import torch
import sys
from pathlib import Path

# Add project root to path so 'src' can be found
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import GFN
from src.math_dataset import MathDataset

def chat_with_gfn(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load structure (Matching Current "Medium" Model)
    # Dim 512, Depth 12, Rank 64, Vocab 16
    dataset = MathDataset(size=10, max_digits=8) 
    model = GFN(vocab_size=16, dim=512, depth=12, rank=64, integrator_type='leapfrog').to(device)
    
    try:
        # Load Raw State Dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"Loaded epoch {checkpoint.get('epoch', '?')} (Loss: {checkpoint.get('loss', '?')})")
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.eval()
        print(f"--- GFN Math CLI (Model: {model_path}) ---")
    except Exception as e:
        print(f"Error: Could not load {model_path}. Detail: {e}")
        return

    while True:
        prompt = input("\nEnter calculation (e.g. 12+34=) or 'exit': ")
        if prompt.lower() == 'exit':
            break
            
        if not prompt.endswith('='):
            prompt += '='
            
        try:
            ids = [dataset.char_to_id[c] for c in prompt]
            input_seq = torch.tensor([ids]).to(device)
            
            with torch.no_grad():
                logits, state = model(input_seq)
                
                # Predict next tokens
                generated = list(ids)
                curr_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated.append(curr_token.item())
                
                for _ in range(12): # Max output digits
                    logits, state = model(curr_token, state=state)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    tok_id = next_token.item()
                    
                    # Check for EOS (Standard model uses it)
                    if tok_id == dataset.char_to_id['<EOS>']:
                        break
                        
                    generated.append(tok_id)
                    curr_token = next_token.unsqueeze(0)
            
            print(f"Result: {dataset.decode(generated)}")
        except Exception as e:
            print(f"Error processing input: {e}")

if __name__ == "__main__":
    # Default to the latest checkpoint from current experiment
    # Users can override via CLI args
    path = r"D:\ASAS\projects\GFN\checkpoints\medium_fast\epoch_0.pt" 
    if len(sys.argv) > 1:
        path = sys.argv[1]
    chat_with_gfn(path)
