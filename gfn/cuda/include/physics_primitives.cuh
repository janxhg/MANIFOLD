#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_RANK 128
#ifndef PI
#define PI 3.14159265358979323846f
#endif

// ==========================================
// Warp-Level Primitives
// ==========================================

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ==========================================
// Math Helpers
// ==========================================

__device__ __forceinline__ float sigmoidf_device(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float safe_tanhf(float x, float bound = 20.0f) {
    return tanhf(fminf(fmaxf(x, -bound), bound));
}

// ==========================================
// Normalization Primitives
// ==========================================

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

__device__ __forceinline__ void tanh_bounding_device(float* s_buf, int dim, int tid, float bound = 10.0f) {
    for (int i = tid; i < dim; i += blockDim.x) {
        s_buf[i] = bound * tanhf(s_buf[i] / bound);
    }
    __syncthreads();
}
