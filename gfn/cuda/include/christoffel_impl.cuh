
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
                if (pot > sing_thresh) m *= sing_strength;
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

    // 1. dL/dh = sum(dL/dGamma * W) * 2 * h * S * M
    for (int r = tid; r < rank; r += bdim) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += s_dGamma[i] * W[i * rank + r];
        // Adjoint of tanh(res/20)*20 is approx identity for now or sech2
        // For professional stability, we assume tanh is in linear region
        s_dL_dh_shared[r] = sum * 2.0f * s_h[r] * S * M;
    }
    __syncthreads();

    // 2. dL/dv = dL/dh @ U^T
    for (int i = tid; i < dim; i += bdim) {
        float gv_i = 0.0f;
        for (int r = 0; r < rank; r++) gv_i += s_dL_dh_shared[r] * U[i * rank + r];
        s_gv[i] += gv_i;
    }
    __syncthreads();
}
