
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_RANK 128
#ifndef PI
#define PI 3.14159265358979323846f
#endif

// Reduction Utilities
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// RMSNorm Utility
__device__ __forceinline__ void rmsnorm_device(float* s_buf, int dim, int tid, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) sum_sq += s_buf[i] * s_buf[i];
    sum_sq = warpReduceSum(sum_sq);
    
    __shared__ float s_rms;
    if (tid == 0) s_rms = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(&s_rms, sum_sq);
    __syncthreads();
    
    float rms = rsqrtf(s_rms / (float)dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) s_buf[i] *= rms;
    __syncthreads();
}

// Tanh Bounding (Velocity Control)
__device__ __forceinline__ void tanh_bounding_device(float* s_buf, int dim, int tid, float bound = 10.0f) {
    for (int i = tid; i < dim; i += blockDim.x) {
        s_buf[i] = bound * tanhf(s_buf[i] / bound);
    }
    __syncthreads();
}

// Device function for Dynamic Friction (Clutch)
__device__ __forceinline__ void compute_friction_device(
    float* s_mu,          
    const float* s_x,     
    const float* s_u,     
    const float* W_x,     
    const float* W_u,     
    const float* b_gate,  
    int dim,
    int tid,
    int topology          
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
        if (s_u) {
            for (int j = 0; j < dim; j++) sum += W_u[i * dim + j] * s_u[j];
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
    int topology,
    // Clutch Parameters
    const float* __restrict__ target_force,
    const float* __restrict__ W_forget,
    const float* __restrict__ W_input,
    const float* __restrict__ b_forget,
    // Shared Memory Pointers
    float* s_h,          
    double* s_E,         
    double* s_P,         
    float* s_M           
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    if (tid == 0) { *s_E = 0.0; *s_P = 0.0; *s_M = 1.0f; }
    for (int r = tid; r < rank; r += bdim) s_h[r] = 0.0f;
    __syncthreads();

    if (use_active) {
        float p_E = 0.0f, p_P = 0.0f;
        for (int i = tid; i < dim; i += bdim) {
            if (plasticity != 0.0f) p_E += v[i] * v[i];
            if (x != nullptr && V_w != nullptr) {
                if (topology == 1) p_P += sinf(x[i]) * V_w[i]; // Periodic Bias
                else p_P += x[i] * V_w[i];
            }
        }
        p_E = warpReduceSum(p_E); p_P = warpReduceSum(p_P);
        if ((tid & 31) == 0) {
            if (plasticity != 0.0f) atomicAdd(s_E, p_E);
            if (x != nullptr && V_w != nullptr) atomicAdd(s_P, p_P);
        }
        __syncthreads();
        if (tid == 0) {
            float m = 1.0f;
            if (plasticity != 0.0f) m *= (1.0f + plasticity * tanhf(*s_E / (float)dim));
            if (x != nullptr && V_w != nullptr) {
                float pot = 1.0f / (1.0f + expf(-fminf(fmaxf((float)*s_P, -20.0f), 20.0f)));
                // Soft Singularity: Use sigmoid-based scaling instead of hard if
                // m_sing = 1.0 + (strength - 1.0) * sigmoid(10.0 * (pot - thresh))
                float soft_m = 1.0f / (1.0f + expf(-10.0f * (pot - sing_thresh)));
                m *= (1.0f + (sing_strength - 1.0f) * soft_m);
            }
            *s_M = fminf(fmaxf(m, 0.1f), 10.0f);
        }
    }

    // h = U^T v
    for (int r = tid; r < rank; r += bdim) {
        float local_h = 0.0f;
        for (int i = 0; i < dim; i++) local_h += v[i] * U[i * rank + r];
        s_h[r] = local_h;
    }
    __syncthreads();

    // S = 1 / (1 + ||h||)
    float local_nsq = 0.0f;
    for (int r = tid; r < rank; r += bdim) local_nsq += s_h[r] * s_h[r];
    local_nsq = warpReduceSum(local_nsq);
    if (tid == 0) *s_E = 0.0; __syncthreads();
    if ((tid & 31) == 0) atomicAdd(s_E, (double)local_nsq);
    __syncthreads();
    float S = 1.0f / (1.0f + sqrtf(fmaxf((float)*s_E, 0.0f)) + 1e-6f);
    float final_m = *s_M;

    // gamma = bound * tanh( (W * h^2 * S * m) / bound )
    for (int i = tid; i < dim; i += bdim) {
        float g = 0.0f;
        for (int r = 0; r < rank; r++) g += W[i * rank + r] * s_h[r] * s_h[r];
        float res = g * S * final_m;
        gamma_out[i] = 20.0f * tanhf(res / 20.0f);
    }
}

// Utility to compute U and W gradients
__device__ __forceinline__ void christoffel_grads_device(
    const float* s_dG, const float* s_v, const float* s_h, const float* U, const float* W, float* g_U, float* g_W, int dim, int rank, int tid, float* s_tmp
) {
    float local_nsq = 0.0f;
    for (int r = tid; r < rank; r += blockDim.x) local_nsq += s_h[r] * s_h[r];
    local_nsq = warpReduceSum(local_nsq);
    if (tid == 0) *s_tmp = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(s_tmp, local_nsq);
    __syncthreads();
    float S = 1.0f / (1.0f + sqrtf(*s_tmp) + 1e-6f);
    for (int i = tid; i < dim * rank; i += blockDim.x) atomicAdd(&g_W[i], s_dG[i/rank] * s_h[i%rank] * s_h[i%rank] * S);
    __syncthreads();
    for (int r = tid; r < rank; r += blockDim.x) {
        float sum_dw = 0.0f;
        for (int i = 0; i < dim; i++) sum_dw += s_dG[i] * W[i * rank + r];
        float dL_dh = sum_dw * 2.0f * s_h[r] * S;
        for (int j = 0; j < dim; j++) atomicAdd(&g_U[j * rank + r], dL_dh * s_v[j]);
    }
}

// Backward of Christoffel w.r.t velocity v
__device__ __forceinline__ void christoffel_v_backward_device(
    const float* __restrict__ s_dGamma,  
    const float* __restrict__ U,         
    const float* __restrict__ W,         
    const float* __restrict__ s_h,       
    const float* __restrict__ s_v,       
    float* __restrict__ s_gv,            
    const int dim,
    const int rank,
    float plasticity,
    bool use_active,
    int topology,
    // Clutch Parameters
    const float* __restrict__ target_force,
    const float* __restrict__ W_forget, 
    const float* __restrict__ W_input, 
    const float* __restrict__ b_forget,
    const float* __restrict__ s_x,       
    float* grad_W_forget,                
    float* grad_W_input,                 
    float* grad_b_forget,                
    // Shared Memory Pointers (pre-computed scale is safer)
    const float S,          
    const float M,          
    float* s_dL_dh_shared   
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // 2. dL/dv = dL/dh @ U^T
    for (int i = tid; i < dim; i += bdim) {
        float gv_i = 0.0f;
        for (int r = 0; r < rank; r++) gv_i += s_dL_dh_shared[r] * U[i * rank + r];
        s_gv[i] += gv_i;
        
        // 3. dL/dM contribution (Active Inference Gradient)
        if (use_active && plasticity != 0.0f) {
             // Derivative of tanh scaling (M) w.r.t energy... logic matches backward dispatch
        }
    }
    __syncthreads();
}

// Full Geometric Backward (including dL/dx and dL/dV_w)
__device__ __forceinline__ void christoffel_full_backward_device(
    const float* __restrict__ s_dGamma,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ s_h,
    const float* __restrict__ s_v,
    const float* __restrict__ s_x,
    const float* __restrict__ V_w,
    float* __restrict__ s_gx,
    float* __restrict__ s_gv,
    float* __restrict__ g_V_w,
    float* __restrict__ g_U,
    float* __restrict__ g_W,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    bool use_active,
    int topology,
    const float S,
    const float M,
    double* s_P_shared,
    float* s_dL_dh_shared
) {
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // 1. Compute dL/dGamma contribution (Adjoint)
    float local_g_out = 0.0f;
    for (int i = tid; i < dim; i += bdim) {
        // Derivative of 20*tanh(res/20) is sech^2(res/20)
        float res_i = 0.0f;
        for (int r = 0; r < rank; r++) res_i += W[i * rank + r] * s_h[r] * s_h[r];
        res_i = (res_i * S * M) / 20.0f;
        float th = tanhf(res_i);
        local_g_out = s_dGamma[i] * (1.0f - th * th);
        s_dL_dh_shared[i] = local_g_out; // Use as temp storage for dL/dBaseG
    }
    __syncthreads();

    // 2. dL/dv (Standard Christoffel) + Weights
    christoffel_v_backward_device(s_dL_dh_shared, U, W, s_h, s_v, s_gv, dim, rank, plasticity, use_active, topology, nullptr, nullptr, nullptr, nullptr, s_x, nullptr, nullptr, nullptr, S, M, s_dL_dh_shared);
    christoffel_grads_device(s_dGamma, s_v, s_h, U, W, g_U, g_W, dim, rank, tid, (float*)s_P_shared); // s_P_shared as temp for grads
    __syncthreads();

    // 3. dL/dx (Geodesic Curvature Gradient)
    if (use_active && s_x != nullptr && V_w != nullptr) {
        // dL/dM
        float dL_dm = 0.0f;
        for (int i = tid; i < dim; i += bdim) {
            float g_i = 0.0f;
            for (int r = 0; r < rank; r++) g_i += W[i * rank + r] * s_h[r] * s_h[r];
            dL_dm += s_dGamma[i] * (g_i * S); // approx
        }
        dL_dm = warpReduceSum(dL_dm);
        __shared__ float s_dLdm;
        if (tid == 0) s_dLdm = 0.0f; __syncthreads();
        if ((tid & 31) == 0) atomicAdd(&s_dLdm, dL_dm);
        __syncthreads();

        // Singularity contribution via Soft Sigmoid
        float pot = 1.0f / (1.0f + expf(-fminf(fmaxf((float)*s_P_shared, -20.0f), 20.0f)));
        float sig_m = 1.0f / (1.0f + expf(-10.0f * (pot - sing_thresh)));
        float dM_dpot = (sing_strength - 1.0f) * 10.0f * sig_m * (1.0f - sig_m);
        float dL_dpot = s_dLdm * dM_dpot;

        for (int i = tid; i < dim; i += bdim) {
            float dpot_dxi = (topology == 1) ? cosf(s_x[i]) * V_w[i] : V_w[i];
            s_gx[i] += dL_dpot * dpot_dxi;
            if (g_V_w) {
                float dpot_dV = (topology == 1) ? sinf(s_x[i]) : s_x[i];
                atomicAdd(&g_V_w[i], dL_dpot * dpot_dV);
            }
        }
    }
    __syncthreads();
}
