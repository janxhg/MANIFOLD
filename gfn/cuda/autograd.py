
import torch
import sys
from torch.autograd import Function

# Import CUDA_AVAILABLE from ops.py to ensure consistency
try:
    from .ops import CUDA_AVAILABLE, gfn_cuda
except ImportError:
    try:
        from gfn.cuda.ops import CUDA_AVAILABLE, gfn_cuda
    except ImportError:
        CUDA_AVAILABLE = False
        gfn_cuda = None

class ChristoffelFusedFn(Function):
    @staticmethod
    def forward(ctx, v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0):
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity, ctx.sing_thresh, ctx.sing_strength = plasticity, sing_thresh, sing_strength
        ctx.topology = topology
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        return gfn_cuda.christoffel_fused(v, U, W, x_in, V_w_in, plasticity, sing_thresh, sing_strength, topology)

    @staticmethod
    def backward(ctx, grad_gamma):
        v, U, W, x, V_w = ctx.saved_tensors
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        grads = gfn_cuda.christoffel_backward(grad_gamma.contiguous(), v, U, W, x_in, V_w_in, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, ctx.topology)
        gv, gU, gW, gx, gV = grads
        return gv, gU, gW, (gx if x is not None else None), (gV if V_w is not None else None), None, None, None, None

def christoffel_fused_autograd(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0):
    if not CUDA_AVAILABLE or not v.is_cuda: return None
    return ChristoffelFusedFn.apply(v.contiguous(), U.contiguous(), W.contiguous(), x.contiguous() if x is not None else None, V_w.contiguous() if V_w is not None else None, plasticity, sing_thresh, sing_strength, topology)

class ReactiveChristoffelFn(Function):
    @staticmethod
    def forward(ctx, v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0):
        if v.dim() == 3: return None # Python fallback
        ctx.save_for_backward(v, U, W, x, V_w)
        ctx.plasticity, ctx.sing_thresh, ctx.sing_strength = plasticity, sing_thresh, sing_strength
        ctx.topology = topology
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        return gfn_cuda.reactive_christoffel_forward(v, U, W, x_in, V_w_in, plasticity, sing_thresh, sing_strength, topology)

    @staticmethod
    def backward(ctx, grad_gamma):
        v, U, W, x, V_w = ctx.saved_tensors
        x_in = x if x is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        V_w_in = V_w if V_w is not None else torch.empty(0, device=v.device, dtype=v.dtype)
        # Aliases christoffel_backward for now
        grads = gfn_cuda.christoffel_backward(grad_gamma.contiguous(), v, U, W, x_in, V_w_in, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, ctx.topology)
        gv, gU, gW, gx, gV = grads
        return gv, gU, gW, (gx if x is not None else None), (gV if V_w is not None else None), None, None, None, None

def reactive_christoffel_autograd(v, U, W, x=None, V_w=None, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, topology=0):
    if not CUDA_AVAILABLE or not v.is_cuda: return None
    return ReactiveChristoffelFn.apply(v.contiguous(), U.contiguous(), W.contiguous(), x.contiguous() if x is not None else None, V_w.contiguous() if V_w is not None else None, plasticity, sing_thresh, sing_strength, topology)

class RecurrentManifoldFusedFn(Function):
    @staticmethod
    def forward(ctx, x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x=None, mix_v=None, W_f_stack=None, W_i_stack=None, b_f_stack=None, topology=0):
        ctx.save_for_backward(x, v, f, U, W, dt_scales, forget_rates, W_f_stack, W_i_stack, b_f_stack)
        ctx.dt, ctx.num_heads, ctx.topology = dt, num_heads, topology
        ctx.plasticity, ctx.sing_thresh, ctx.sing_strength = plasticity, sing_thresh, sing_strength
        ctx.mix_x, ctx.mix_v = mix_x, mix_v
        mx = mix_x if mix_x is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        mv = mix_v if mix_v is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        res = gfn_cuda.recurrent_manifold_fused(x.contiguous(), v.contiguous(), f.contiguous(), U.contiguous(), W.contiguous(), dt, dt_scales.contiguous(), forget_rates.contiguous(), num_heads, plasticity, sing_thresh, sing_strength, mx.contiguous(), mv.contiguous(), W_f_stack, W_i_stack, b_f_stack, topology)
        return res[0], res[1], res[2], res[3]

    @staticmethod
    def backward(ctx, grad_xf, grad_vf, grad_seq, grad_reg):
        x0, v0, f_seq, U, W, dt_scales, forget, Wf, Wi, bf = ctx.saved_tensors
        mx = ctx.mix_x if ctx.mix_x is not None else torch.empty(0, device=x0.device)
        mv = ctx.mix_v if ctx.mix_v is not None else torch.empty(0, device=x0.device)
        # Recompute final state for adjoint
        with torch.no_grad():
             res = gfn_cuda.recurrent_manifold_fused(x0.contiguous(), v0.contiguous(), f_seq.contiguous(), U.contiguous(), W.contiguous(), ctx.dt, dt_scales.contiguous(), forget.contiguous(), ctx.num_heads, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, mx.contiguous(), mv.contiguous(), Wf, Wi, bf, ctx.topology)
             xf, vf = res[0], res[1]
        grads = gfn_cuda.recurrent_manifold_backward(grad_seq.contiguous() if grad_seq is not None else torch.empty(0, device=x0.device), grad_xf.contiguous() if grad_xf is not None else torch.empty(0, device=x0.device), grad_vf.contiguous() if grad_vf is not None else torch.empty(0, device=x0.device), xf, vf, f_seq.contiguous(), U.contiguous(), W.contiguous(), ctx.dt, dt_scales.contiguous(), forget.contiguous(), ctx.num_heads, ctx.plasticity, ctx.sing_thresh, ctx.sing_strength, mx.contiguous(), mv.contiguous(), Wf, Wi, bf, ctx.topology)
        # Return all gradients (match signature)
        return grads[0], grads[1], grads[2], grads[3], grads[4], None, None, grads[7], None, None, None, None, grads[5], grads[6], grads[8], grads[9], grads[10], None

def recurrent_manifold_fused_autograd(x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity=0.0, sing_thresh=1.0, sing_strength=1.0, mix_x=None, mix_v=None, W_f_stack=None, W_i_stack=None, b_f_stack=None, topology=0):
    if not CUDA_AVAILABLE or not x.is_cuda: return None
    return RecurrentManifoldFusedFn.apply(x, v, f, U, W, dt, dt_scales, forget_rates, num_heads, plasticity, sing_thresh, sing_strength, mix_x, mix_v, W_f_stack, W_i_stack, b_f_stack, topology)
