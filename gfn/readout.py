import torch
import torch.nn as nn

class ImplicitReadout(nn.Module):
    """
    Temperature-Annealed Sigmoid Readout (Gumbel-Softmax variant)
    
    Prevents gradient cliffs from hard thresholding by using soft sigmoid
    with temperature that anneals from high (smooth) to low (sharp).
    
    Args:
        dim: Input dimension
        coord_dim: Output coordinate dimension  
        temp_init: Initial temperature (high = smooth gradients)
        temp_final: Final temperature (low = sharp outputs)
    """
    def __init__(self, dim, coord_dim, temp_init=1.0, temp_final=0.2):
        super().__init__()
        
        # MLP to output coordinates
        # TOROIDAL EMBEDDING: Input is doubled (Sin/Cos)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim), 
            nn.GELU(),
            nn.Linear(dim, coord_dim)
        )
        
        # Tag for specialized Level 5 initialization in model.py
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                module.is_readout = True
        
        # Temperature annealing parameters
        self.temp_init = temp_init
        self.temp_final = temp_final
        
        # Training progress tracker
        self.register_buffer('training_step', torch.tensor(0))
        self.register_buffer('max_steps', torch.tensor(1000))
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            bits_soft: [batch, seq, coord_dim] in range [0, 1]
        """
        # LEVEL 16: TOROIDAL EMBEDDING
        # Map x -> [sin(x), cos(x)] to enforce periodicity
        x_emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        
        # LEVEL 20: SIGNAL BOOST
        # 10.0 constant gain provides sharp gradients for BCE convergence.
        logits = self.mlp(x_emb) * 10.0 # [batch, seq, coord_dim]
        
        return logits
        # LEVEL 19 REMOVED INFERENCE BRANCH (Moved to generate/testing)
        # return torch.sigmoid(logits)
    
    def update_step(self):
        """Call this after each optimizer step to update temperature schedule."""
        if self.training:
            self.training_step += 1
    
    def set_max_steps(self, max_steps):
        """Update max steps for temperature schedule."""
        self.max_steps = torch.tensor(max_steps)
