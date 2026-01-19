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
    def __init__(self, dim, coord_dim, temp_init=5.0, temp_final=0.5):
        super().__init__()
        
        # MLP to output coordinates
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, coord_dim)
        )
        
        # Temperature annealing parameters
        self.temp_init = temp_init
        self.temp_final = temp_final
        
        # Training progress tracker
        self.register_buffer('training_step', torch.tensor(0))
        self.register_buffer('max_steps', torch.tensor(1500))
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            bits_soft: [batch, seq, coord_dim] in range [0, 1]
        """
        # Get continuous coordinates
        coords = self.mlp(x)  # [batch, seq, coord_dim]
        
        if self.training:
            # Temperature annealing: high temp early (smooth), low temp late (sharp)
            progress = min(1.0, self.training_step.float() / self.max_steps)
            temp = self.temp_init * (1.0 - progress) + self.temp_final * progress
            
            # Soft threshold with temperature
            bits_soft = torch.sigmoid(coords / temp)
        else:
            # Inference: use final temperature for sharp outputs
            bits_soft = torch.sigmoid(coords / self.temp_final)
        
        return bits_soft
    
    def update_step(self):
        """Call this after each optimizer step to update temperature schedule."""
        if self.training:
            self.training_step += 1
    
    def set_max_steps(self, max_steps):
        """Update max steps for temperature schedule."""
        self.max_steps = torch.tensor(max_steps)
