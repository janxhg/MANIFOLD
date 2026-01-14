import torch
import torch.nn as nn
from .layers import GLayer


class GFN(nn.Module):
    """
    Geodesic Flow Network (GFN)
    
    A sequence model that evolves hidden states as geodesic flows
    on a Riemannian manifold. Achieves O(1) memory complexity.
    
    Architecture:
        1. Embedding: Token -> Force impulse on manifold
        2. Dynamics: G-Layers evolve state (x, v) via geodesic flow
        3. Readout: Position x -> Logits via learned projection
    
    Args:
        vocab_size: Size of vocabulary
        dim: Hidden dimension (default: 256)
        depth: Number of G-Layers (default: 4)
        rank: Low-rank Christoffel approximation (default: 32)
        integrator_type: 'heun', 'rk4', or 'symplectic' (default: 'heun')
    
    Example:
        >>> model = GFN(vocab_size=16, dim=512, depth=12, integrator_type='heun')
        >>> logits, state = model(input_ids)
    """
    
    def __init__(self, vocab_size, dim=256, depth=4, rank=32, integrator_type='heun'):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.integrator_type = integrator_type
        
        # Token embedding (acts as "force" on the manifold)
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Stack of Geodesic Layers
        self.layers = nn.ModuleList([
            GLayer(dim, rank=rank, integrator_type=integrator_type) 
            for _ in range(depth)
        ])
        
        # Output projection
        self.readout_norm = nn.LayerNorm(dim)
        self.readout = nn.Linear(dim, vocab_size)
        
        # Learnable initial state (position and velocity)
        self.x0 = nn.Parameter(torch.zeros(1, dim))
        self.v0 = nn.Parameter(torch.zeros(1, dim))
    
    def forward(self, input_ids, attention_mask=None, state=None):
        """
        Forward pass through the geodesic flow.
        
        Args:
            input_ids: Token indices [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len] (1=valid, 0=pad)
            state: Optional tuple (x, v) to continue from previous state
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            state: Final state tuple (x, v) for continuation
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize state from learnable parameters or provided state
        if state is None:
            x = self.x0.expand(batch_size, -1)
            v = self.v0.expand(batch_size, -1)
        else:
            x, v = state
        
        # Pre-compute all token embeddings (forces)
        all_forces = self.embedding(input_ids)  # [batch, seq_len, dim]
        
        # Prepare attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        else:
            mask = torch.ones(batch_size, seq_len, 1, device=input_ids.device)
        
        # Process sequence token by token (recurrent dynamics)
        logits_list = []
        
        for t in range(seq_len):
            # Get force for current timestep
            force = all_forces[:, t]  # [batch, dim]
            
            # Apply mask (zero force for padding tokens)
            force = force * mask[:, t]
            
            # Evolve state through all G-Layers
            for layer in self.layers:
                x, v = layer(x, v, force)
            
            # Readout: project position to vocabulary logits
            out = self.readout_norm(x)
            logit = self.readout(out)  # [batch, vocab_size]
            logits_list.append(logit.unsqueeze(1))
        
        # Stack all logits
        logits = torch.cat(logits_list, dim=1)  # [batch, seq_len, vocab_size]
        
        return logits, (x, v)
    
    def generate(self, prompt_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        """
        Autoregressive generation with sampling.
        
        Args:
            prompt_ids: Prompt token indices [1, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Softmax temperature (1.0 = normal, <1 = sharper)
            top_k: Limit to top K tokens (e.g. 40)
            top_p: Nucleus sampling probability (e.g. 0.9)
            
        Returns:
            generated_ids: Full sequence including prompt
        """
        self.eval()
        device = prompt_ids.device
        
        # Process prompt
        logits, state = self(prompt_ids)
        
        # Start generation
        generated = prompt_ids.tolist()[0]
        
        def sample_next(logits, temp=1.0, k=None, p=None):
            # Last timestep logits
            next_logit = logits[:, -1, :] / temp
            probs = torch.softmax(next_logit, dim=-1)
            
            # Top-K
            if k is not None:
                v, _ = torch.topk(next_logit, k)
                next_logit[next_logit < v[:, [-1]]] = -float('Inf')
            
            # Top-P (Nucleus)
            if p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logit, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logit[indices_to_remove] = -float('Inf')
            
            # Sample
            if k is None and p is None:
                # Greedy
                return torch.argmax(next_logit, dim=-1, keepdim=True)
            else:
                # Multinomial
                probs = torch.softmax(next_logit, dim=-1)
                return torch.multinomial(probs, num_samples=1)

        # Initial sample
        curr_token = sample_next(logits, temperature, top_k, top_p)
        generated.append(curr_token.item())
        
        for _ in range(max_new_tokens - 1):
            logits, state = self(curr_token, state=state)
            curr_token = sample_next(logits, temperature, top_k, top_p)
            generated.append(curr_token.item())
        
        return generated
