import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from gfn.model import Manifold

# === PHYSICS SIMULATION (Double Pendulum) ===
def double_pendulum_derivs(state, t, m1, m2, L1, L2, g):
    theta1, z1, theta2, z2 = state
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    
    # Lagrangian derivatives
    theta1_dot = z1
    theta2_dot = z2
    
    num1 = -g*(2*m1+m2)*np.sin(theta1) - m2*g*np.sin(theta1-2*theta2) - 2*s*m2*(z2**2*L2 + z1**2*L1*c)
    den1 = L1 * (2*m1 + m2 - m2*c*2*c) # Typo in standard formula? standard is -m2*cos(2*delta)
    # Using a simplified Hamiltonian form for cleaner data generation might be better
    # But let's stick to standard equations
    den1 = L1 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))
    
    z1_dot = num1 / den1
    
    num2 = 2*s*(z1**2*L1*(m1+m2) + g*(m1+m2)*np.cos(theta1) + z2**2*L2*m2*c)
    den2 = L2 * (2*m1 + m2 - m2*np.cos(2*(theta1-theta2)))
    z2_dot = num2 / den2
    
    return [theta1_dot, z1_dot, theta2_dot, z2_dot]

def generate_chaos_data(n_seq, seq_len):
    print("🌀 Generating Double Pendulum Chaos Data...")
    m1, m2, L1, L2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    dt = 0.05
    t = np.arange(0, seq_len*dt, dt)
    
    data = []
    
    for _ in range(n_seq):
        # Random initial state near unstable equilibrium
        init_state = np.random.uniform(-0.5, 0.5, 4) + np.array([np.pi, 0, np.pi, 0])
        trajectory = odeint(double_pendulum_derivs, init_state, t, args=(m1, m2, L1, L2, g))
        data.append(trajectory)
        
    return torch.tensor(np.array(data), dtype=torch.float32)

class BaselineLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)

def run_chaos_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEQ_LEN = 100
    PRED_LEN = 50
    
    # 1. Generate Data
    # 4 coords: theta1, w1, theta2, w2
    data = generate_chaos_data(1000, SEQ_LEN + PRED_LEN)
    train_data = data[:800].to(device)
    test_data = data[800:].to(device)
    
    # 2. Train Models
    # Manifold Setup
    # Manifold expects discrete tokens usually. For continuous regression:
    # We need a continuous Manifold or abuse the embedding.
    # Actually Manifold is designed for tokens in current `model.py`.
    # BUT `Manifold` can accept continuous inputs if we bypass embedding?
    # NO, `model.forward` takes `x` (long).
    # Hack: We will discretize the continuous values into 1000 bins for this test.
    # This proves it can learn physics from "Symbols".
    
    print("🔢 Discretizing Dynamics into Tokens (Symbolic Physics)...")
    BINS = 1000
    MIN_VAL, MAX_VAL = -10, 10
    
    def continuous_to_tokens(tensor):
        norm = (tensor - MIN_VAL) / (MAX_VAL - MIN_VAL)
        tokens = (norm * BINS).long().clamp(0, BINS-1)
        # Flatten 4 coords into sequence? Or parallel heads?
        # Let's flatten: [t1_theta, t1_w, ..., t2_theta...]
        # shape [B, L, 4] -> [B, L*4]
        b, l, d = tensor.shape
        return tokens.reshape(b, l*d)
        
    train_tokens = continuous_to_tokens(train_data)
    test_tokens = continuous_to_tokens(test_data)
    
    print(f"Dataset Shape: {train_tokens.shape}")
    
    # Model 1: Manifold
    manifold = Manifold(
        vocab_size=BINS,
        dim=128,
        depth=4,
        heads=4,
        integrator_type='symplectic', # Crucial
        physics_config={'active_inference': {'enabled': False}} 
    ).to(device)
    
    manifold_opt = optim.AdamW(manifold.parameters(), lr=1e-3)
    
    # Model 2: LSTM Baseline (Symbolic)
    lstm = BaselineLSTM(128, 128).to(device) # Embeddings handled implicitly?
    # Wait, LSTM needs embeddings for tokens too comparison fair
    lstm_embed = nn.Embedding(BINS, 128).to(device)
    lstm_head = nn.Linear(128, BINS).to(device)
    lstm_opt = optim.AdamW(list(lstm.parameters()) + list(lstm_embed.parameters()) + list(lstm_head.parameters()), lr=1e-3)
    
    criterion = nn.CrossEntropyLoss()
    
    # Train Loop (Short)
    EPOCHS = 500
    print("⚔️ Training Manifold vs LSTM on Chaos...")
    
    loss_m_hist, loss_l_hist = [], []
    
    for epoch in range(EPOCHS):
        # Batching
        idx = torch.randint(0, 800, (32,))
        batch = train_tokens[idx]
        
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        
        # Manifold Step
        manifold_opt.zero_grad()
        logits_m, _, _ = manifold(inp)
        loss_m = criterion(logits_m.reshape(-1, BINS), tgt.reshape(-1))
        loss_m.backward()
        manifold_opt.step()
        
        # LSTM Step
        lstm_opt.zero_grad()
        emb_l = lstm_embed(inp)
        out_l, _ = lstm.lstm(emb_l)
        logits_l = lstm_head(out_l)
        loss_l = criterion(logits_l.reshape(-1, BINS), tgt.reshape(-1))
        loss_l.backward()
        lstm_opt.step()
        
        loss_m_hist.append(loss_m.item())
        loss_l_hist.append(loss_l.item())
        
        if epoch % 50 == 0:
            print(f"Ep {epoch}: Manifold Loss={loss_m.item():.4f} | LSTM Loss={loss_l.item():.4f}")
            
    # 3. Evaluate Long-Term Prediction
    print("\n🔮 Evaluating Long-Term Prediction Standard...")
    
    # Pick a test sample
    sample_idx = 0
    seeds = test_tokens[sample_idx, :50].unsqueeze(0) # First 50 steps
    future = test_tokens[sample_idx, 50:] # Next steps
    
    def autoregress(model_fn, seed, length):
        curr = seed
        preds = []
        with torch.no_grad():
            for _ in range(length):
                logits = model_fn(curr)
                next_tok = torch.argmax(logits[:, -1:], dim=-1)
                preds.append(next_tok)
                curr = torch.cat([curr, next_tok], dim=1)
        return torch.cat(preds, dim=1)
        
    # Wrappers
    def pred_manifold(x):
        l, _, _ = manifold(x)
        return l
        
    def pred_lstm(x):
        e = lstm_embed(x)
        o, _ = lstm.lstm(e)
        return lstm_head(o)
        
    print("Generating Manifold Trajectory...")
    fut_m = autoregress(pred_manifold, seeds, len(future))
    
    print("Generating LSTM Trajectory...")
    fut_l = autoregress(pred_lstm, seeds, len(future))
    
    # Calculate Error
    err_m = (fut_m.float() - future.float()).abs().mean().item()
    err_l = (fut_l.float() - future.float()).abs().mean().item()
    
    print(f"\n🏆 Final Chaos Results:")
    print(f"Manifold MAE: {err_m:.4f}")
    print(f"LSTM MAE:     {err_l:.4f}")
    
    if err_m < err_l:
        print("✅ Manifold WINS! (Better prediction of chaos)")
    else:
        print("❌ Manifold lost (Needs tuning)")

if __name__ == "__main__":
    run_chaos_test()
