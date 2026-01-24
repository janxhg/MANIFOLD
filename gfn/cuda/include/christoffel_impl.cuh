
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_RANK 128

// Reduction Utilities
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Device function for Dynamic Friction (Clutch)
// friction = sigmoid(W_x @ x + W_u @ u + b) * 5.0
__device__ __forceinline__ void compute_friction_device(
    float* s_mu,          // [Dim] Output
    const float* s_x,     // [Dim] Input State
    const float* s_u,     // [Dim] Input Force
    const float* W_x,     // [Dim, Dim] OR [Dim, 2*Dim] if Torus
    const float* W_u,     // [Dim, Dim]
    const float* b_gate,  // [Dim]
    int dim,
    int tid,
    int topology          // 0: Euclidean, 1: Torus
) {
    for (int i = tid; i < dim; i += blockDim.x) {
        float sum = b_gate[i];
        if (topology == 1) { // TORUS
            for (int j = 0; j < dim; j++) {
                sum += W_x[i * (2 * dim) + j] * sinf(s_x[j]);
                sum += W_x[i * (2 * dim) + j + dim] * cosf(s_x[j]);
            }
        } else { // EUCLIDEAN
            for (int j = 0; j < dim; j++) {
                sum += W_x[i * dim + j] * s_x[j];
            }
        }
        for (int j = 0; j < dim; j++) {
             sum += W_u[i * dim + j] * s_u[j];
        }
        s_mu[i] = (1.0f / (1.0f + expf(-fminf(fmaxf(sum, -20.0f), 20.0f)))) * 5.0f;
    }
    __syncthreads();
}

// Optimized Christoffel Device Function
__device__ __forceinline__ void christoffel_device(
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ gamma_out, 
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
    // Clutch Parameters (Functional Manifold 2.0)
    const float* __restrict__ target_force,
    const float* __restrict__ W_forget,
    const float* __restrict__ W_input,
    const float* __restrict__ b_forget,
    // Shared Memory Pointers
    float* s_h,          // [rank]
    double* s_E,         // [1] - High Precision
    double* s_P,         // [1] - High Precision
    float* s_M           // [1] - Final Multiplier
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // 1. Reset Shared
    if (tid == 0) {
        *s_E = 0.0;
        *s_P = 0.0;
        *s_M = 1.0f;
    }
    for (int r = tid; r < rank; r += bdim) s_h[r] = 0.0f;
    __syncthreads();

    // 2. Active Inference Multipliers
    if (use_active) {
        float p_E = 0.0f, p_P = 0.0f;
        for (int i = tid; i < dim; i += bdim) {
            if (plasticity != 0.0f) p_E += v[i] * v[i];
            if (x != nullptr && V_w != nullptr) p_P += x[i] * V_w[i];
        }
        // Reduction
        p_E = warpReduceSum(p_E);
        p_P = warpReduceSum(p_P);
        if ((tid & 31) == 0) {
            if (plasticity != 0.0f) atomicAdd(s_E, p_E);
            if (x != nullptr && V_w != nullptr) atomicAdd(s_P, p_P);
        }
        __syncthreads();
        
        if (tid == 0) {
            float m = 1.0f;
            if (plasticity != 0.0f) m *= (1.0f + plasticity * tanhf(*s_E / (float)dim));
            if (x != nullptr && V_w != nullptr) {
                float pot = 1.0f / (1.0f + expf(-fminf(fmaxf(*s_P, -20.0), 20.0))); // Safe Exp
                if (pot > sing_thresh) m *= sing_strength;
            }
            // Absolute clamp for safety
            *s_M = fminf(fmaxf(m, 0.1f), 10.0f);
        }
    }

    // 3. Projection: h = U^T v
    // Coalesced Matrix-Vector Product
    for (int r = tid; r < rank; r += bdim) {
        float local_h = 0.0f;
        for (int i = 0; i < dim; i++) {
            local_h += v[i] * U[i * rank + r];
        }
        s_h[r] = local_h; // Direct assignment since 1 block per head/batch
    }
    __syncthreads();

    // 4. Reconstruction: gamma = W * h^2 * S
    // a) Norm S
    float local_nsq = 0.0f;
    for (int r = tid; r < rank; r += bdim) local_nsq += s_h[r] * s_h[r];
    local_nsq = warpReduceSum(local_nsq);
    
    if (tid == 0) *s_E = 0.0; 
    __syncthreads();
    
    // Proper shared memory block reduction for S
    if ((tid & 31) == 0) atomicAdd(s_E, (double)local_nsq);
    __syncthreads();
    
    // Safety: Clamp energy to non-negative to prevent NaN in sqrt
    float energy_val = fmaxf((float)*s_E, 0.0f);
    float S = 1.0f / (1.0f + sqrtf(energy_val) + 1e-6f);
    float final_m = *s_M;

    // b) Gamma Reconstruction
    for (int i = tid; i < dim; i += bdim) {
        float g = 0.0f;
        for (int r = 0; r < rank; r++) g += W[i * rank + r] * s_h[r] * s_h[r];
        
        float res = g * S * final_m;
        
        // LEVEL 25: THE CLUTCH (CUDA Port)
        // Dynamic Input-Dependent Friction: mu = sigmoid(W_x * x + W_f * F + b) * 5.0
        // (Handled by Leapfrog Integrator for stability, but we can compute it here if needed)

        // LEVEL 23: PURE CURVATURE (No redundant friction)
        gamma_out[i] = 20.0f * tanhf(res / 20.0f);
    }
}
// Backward of Christoffel w.r.t velocity v AND Clutch Parameters
// Computes s_gv += (dL/dGamma)^T * (dGamma/dv)
// Computes grad_W_forget, grad_b_forget, grad_W_input
__device__ __forceinline__ void christoffel_v_backward_device(
    const float* __restrict__ s_dGamma,  // [Dim]
    const float* __restrict__ U,         // [Dim, Rank]
    const float* __restrict__ W,         // [Dim, Rank]
    const float* __restrict__ s_h,       // [Rank]
    const float* __restrict__ s_v,       // [Dim]
    float* __restrict__ s_gv,            // [Dim] -> Cumulative update
    const int dim,
    const int rank,
    float plasticity,
    bool use_active,
    // Clutch Parameters
    const float* __restrict__ target_force,
    const float* __restrict__ W_forget, 
    const float* __restrict__ W_input, 
    const float* __restrict__ b_forget,
    const float* __restrict__ s_x,       // [Dim] Input to gate
    float* grad_W_forget,                // [Dim, Dim] Global Atomic
    float* grad_W_input,                 // [Dim, Dim] Global Atomic
    float* grad_b_forget,                // [Dim] Global Atomic
    // Shared Memory Pointers
    const float S,          // Pre-computed Norm Scaling
    const float M,          // Pre-computed Active Multiplier
    float* s_dL_dh_shared   // Scratch [rank]
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // 1. Friction Recomputation
    
    // We need gate_sum for EACH dimension i.
    // Friction is computed per dimension i.
    // friction[i] = sigmoid(gate_sum[i]) * 5.0
    // dL/d_friction[i] = s_dGamma[i] * v[i]
    // dL/d_gate[i] = dL/d_friction[i] * friction[i] * (1 - sigmoid[i]) * scale? 
    // No, friction = sig * 5. dF/dsig = 5. dF/dgate = 5 * sig * (1-sig).
    // So dL/d_gate[i] = s_dGamma[i] * v[i] * (friction[i] * (1.0f - friction[i]/5.0f));
    
    // We compute this on the fly to save memory.
    
    // 2. Compute dL/dh [Rank] (Geometric Part)
    // dL/dh_r approx = sum_i (dL/dGamma_i * W_ir) * 2 * h_r * S * M
    for (int r = tid; r < rank; r += bdim) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += s_dGamma[i] * W[i * rank + r];
        }
        s_dL_dh_shared[r] = sum * 2.0f * s_h[r] * S * M;
    }
    __syncthreads();

    // 3. Compute dL/dv = dL/dh @ U^T + Friction Term
    for (int i = tid; i < dim; i += bdim) {
        float grad_v_i = 0.0f;
        
        // Geometric
        for (int r = 0; r < rank; r++) {
            grad_v_i += s_dL_dh_shared[r] * U[i * rank + r];
        }
        
        // Clutch Logic
        if (W_forget != nullptr && b_forget != nullptr && s_x != nullptr) {
            float g_sum = b_forget[i];
            for (int k = 0; k < dim; k++) {
                 g_sum += s_x[k] * W_forget[i * dim + k];
                 if (target_force != nullptr && W_input != nullptr)
                     g_sum += target_force[k] * W_input[i * dim + k];
            }
            float sig = 1.0f / (1.0f + expf(-fminf(fmaxf(g_sum, -20.0f), 20.0f)));
            float f_val = sig * 5.0f;
            
            // dL/dv += dL/dGamma * dGamma/dv
            // Gamma now is PURE CURVATURE, friction is handled EXTERNALLY in integrator.
            // So dGamma/dv = 0 for the friction part here. 
            // We only keep the gate backprop if we want to learn it.
            
            // Backprop to Gate Weights
            // dL/d_gate_i = dL/d_mu * d_mu/d_gate
            // dL/d_mu is provided by the integrator adjoint.
            // THIS FUNCTION handles the geometric part + direct-dependent friction.
            // Since we removed friction from gamma_out, dGamma/dv = 0 here for friction.
        }
        
        s_gv[i] += grad_v_i;
    }
    __syncthreads();
}
