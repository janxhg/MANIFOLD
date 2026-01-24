"""
GFN Optimizers
==============

Riemannian-aware optimizers for training on curved manifolds.
"""

import torch
from torch.optim import Optimizer
import math


class RiemannianAdam(Optimizer):
    """
    Riemannian Adam Optimizer.
    
    Instead of Euclidean gradient descent (W = W - lr * grad), this optimizer
    uses exponential map retraction to ensure weight updates stay on the manifold.
    
    Update rule:
        W_new = Retract(W_old, -lr * corrected_grad)
    
    For most neural network weights, we use a simple retraction that includes
    normalization to prevent weight explosion/collapse.
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        betas: Adam momentum coefficients (default: (0.9, 0.999))
        eps: Numerical stability (default: 1e-8)
        weight_decay: L2 regularization (default: 0.01)
        retraction: Type of retraction ('euclidean', 'normalize', 'cayley')
        max_norm: Maximum weight norm for retraction (default: 10.0)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, retraction='normalize', max_norm=10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       retraction=retraction, max_norm=max_norm)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step with Riemannian retraction.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = float(group['lr'])
            beta1, beta2 = group['betas']
            eps = float(group['eps'])
            weight_decay = float(group['weight_decay'])
            retraction = group['retraction']
            max_norm = float(group['max_norm'])

            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply weight decay (decoupled, like AdamW)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                # Get state
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2
                
                # Compute step direction (standard Adam)
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                step_direction = corrected_exp_avg / denom
                
                # === RIEMANNIAN RETRACTION ===
                # Instead of p = p - lr * step, we use a retraction
                
                if retraction == 'euclidean':
                    # Standard Euclidean update (fallback)
                    p.data.add_(step_direction, alpha=-lr)
                    
                elif retraction == 'normalize':
                    # Normalize retraction: keep weight matrices bounded
                    p.data.add_(step_direction, alpha=-lr)
                    
                    # Project back to bounded manifold
                    norm = p.data.norm()
                    if norm > max_norm:
                        p.data.mul_(max_norm / norm)
                        
                elif retraction == 'cayley':
                    # Cayley retraction for orthogonal-ish manifold
                    # Useful for preserving structure in weight matrices
                    # W_new = (I - A/2)^{-1} (I + A/2) W
                    # Simplified: just apply update and re-orthogonalize
                    p.data.add_(step_direction, alpha=-lr)
                    
                    # For 2D weight matrices, optionally orthogonalize
                    if p.dim() == 2 and p.shape[0] == p.shape[1]:
                        # Approximate orthogonalization via SVD
                        try:
                            U, S, Vh = torch.linalg.svd(p.data, full_matrices=False)
                            p.data.copy_(U @ Vh)
                        except:
                            pass  # SVD can fail, fall back to no-op
                    else:
                        # For non-square, just normalize
                        norm = p.data.norm()
                        if norm > max_norm:
                            p.data.mul_(max_norm / norm)
                            
                elif retraction == 'torus':
                    # Toroidal retraction: periodic wrapping [0, 2*PI)
                    p.data.add_(step_direction, alpha=-lr)
                    TWO_PI = 2.0 * 3.14159265359
                    p.data.copy_(torch.remainder(p.data, TWO_PI))
                    
                else:
                    # Unknown retraction, use Euclidean
                    p.data.add_(step_direction, alpha=-lr)
        
        return loss


class ManifoldSGD(Optimizer):
    """
    Simple SGD with retraction for Riemannian manifolds.
    
    Uses the same retraction concept but without momentum.
    Useful for debugging or when Adam is unstable.
    """
    
    def __init__(self, params, lr=1e-2, weight_decay=0.0, max_norm=10.0):
        defaults = dict(lr=lr, weight_decay=weight_decay, max_norm=max_norm)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            max_norm = group['max_norm']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Update with retraction
                p.data.add_(grad, alpha=-lr)
                
                # Normalize if needed
                norm = p.data.norm()
                if norm > max_norm:
                    p.data.mul_(max_norm / norm)
        
        return loss
