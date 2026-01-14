import torch
import time
from src.model import GFN

def run_memory_trial(model, vocab_size, context_len, device):
    """
    Mental test: "The secret number is X ... [random filler] ... What is the secret number?"
    Simplified for char/id level: [KEY_VAL] [RANDOM_IDS...] -> Predict KEY_VAL at the end.
    """
    # 0: KEY_DATA, 1-V: DISTRACTORS
    key_val = torch.randint(1, vocab_size, (1,)).to(device)
    filler = torch.randint(1, vocab_size, (context_len,)).to(device)
    
    # Sequence: [Key] [Filler...]
    input_seq = torch.cat([key_val, filler]).unsqueeze(0) # [1, L+1]
    
    model.eval()
    with torch.no_grad():
        # The model sees the key, then filler.
        # We want to see if at any point in the filler, or at the end, 
        # it can "retrieve" the key if prompted.
        # Let's simplify: Can it predict Key given the sequence? 
        # (This tests if the signal survives the manifold flow)
        logits, _ = model(input_seq)
        
        # We check if the logits at the LAST position favor the 'key_val'
        # even though it hasn't seen it for 'context_len' steps.
        # We'll use a specific "PROMPT" token (id 0) to ask for the key.
        prompt = torch.zeros(1, 1).long().to(device) # Token 0 = "Retrieve"
        input_with_prompt = torch.cat([input_seq, prompt], dim=1)
        
        logits, _ = model(input_with_prompt)
        prediction = torch.argmax(logits[0, -1, :]).item()
        
    return prediction == key_val.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 100
    dim = 256
    depth = 4
    rank = 32
    
    print(f"--- GFN Memory Stress Test (Device: {device}) ---")
    print(f"Testing ability to transport information via Geodesic Flow.")
    
    # Create a RANDOM model (not trained) to see if the structure 
    # even allows signal propagation. (Better results if trained, of course)
    model = GFN(vocab_size, dim, depth, rank).to(device)
    
    lengths = [32, 64, 128, 256, 512, 1024]
    trials = 20
    
    for length in lengths:
        hits = 0
        t0 = time.time()
        for _ in range(trials):
            if run_memory_trial(model, vocab_size, length, device):
                hits += 1
        dt = time.time() - t0
        acc = (hits / trials) * 100
        print(f"Context Length {length:.<5}: {acc:>6.1f}% Accuracy | Time: {dt:.2f}s")

    print("\nNote: On a random model, accuracy will be low (~1%), but we are checking for gradient/signal death.")
    print("If signal death occurs, it will be 0%. If flow works, it stays >0% or we see stable integration.")

if __name__ == "__main__":
    main()
