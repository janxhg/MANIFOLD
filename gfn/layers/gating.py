import torch
import torch.nn as nn

class RiemannianGating(nn.Module):
    """
    Computes a scalar curvature-based gating mechanism.
    If curvature is high, dt should be small (complex region).
    If curvature is low (flat), dt can be large (skip connection behavior).
    """
    def __init__(self, dim):
        super().__init__()
        self.curvature_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid() # Range [0, 1]
        )
        
    def forward(self, x):
        """
        Returns a scaling factor for dt.
        """
        # Try CUDA path
        try:
            from gfn.cuda.ops import dynamic_gating_fused, CUDA_AVAILABLE
            if CUDA_AVAILABLE and x.is_cuda:
                W1 = self.curvature_net[0].weight  # [dim/4, dim]
                b1 = self.curvature_net[0].bias    # [dim/4]
                W2 = self.curvature_net[2].weight  # [1, dim/4]
                b2 = self.curvature_net[2].bias    # [1]
                return dynamic_gating_fused(x, W1, b1, W2, b2)
        except Exception:
            pass
        
        # Fallback PyTorch
        return self.curvature_net(x)
