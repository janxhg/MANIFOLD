
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

__global__ void reactive_christoffel_forward_kernel(
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ gamma,
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    int topology,
    float R_val,
    float r_val
) {
    __shared__ float s_h[MAX_RANK];
    __shared__ double s_E;
    __shared__ double s_P;
    __shared__ float s_M;

    const int b = blockIdx.x;
    if (b >= batch) return;

    christoffel_device(
        v + b * dim, U, W, gamma + b * dim, 
        (x != nullptr) ? (x + b * dim) : nullptr, V_w, 
        dim, rank, 
        plasticity, sing_thresh, sing_strength, true, 
        topology,
        s_h, &s_E, &s_P, &s_M,
        R_val, r_val
    );
}

extern "C" void launch_reactive_christoffel_forward(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    int topology,
    float R_val, float r_val,
    cudaStream_t stream
) {
    reactive_christoffel_forward_kernel<<<batch, BLOCK_SIZE, 0, stream>>>(
        v, U, W, gamma, x, V_w, batch, dim, rank, plasticity, sing_thresh, sing_strength, topology, R_val, r_val
    );
}

__global__ void reactive_christoffel_backward_kernel(
    const float* __restrict__ grad_gamma,
    const float* __restrict__ v,
    const float* __restrict__ U,
    const float* __restrict__ W,
    const float* __restrict__ x,
    const float* __restrict__ V_w,
    float* __restrict__ grad_v,
    float* __restrict__ grad_U,
    float* __restrict__ grad_W,
    float* __restrict__ grad_x,
    float* __restrict__ grad_V_w,
    const int batch,
    const int dim,
    const int rank,
    float plasticity,
    float sing_thresh,
    float sing_strength,
    int topology,
    float R_val,
    float r_val
) {
    // For simplicity, we reuse the robust logic from christoffel_backward_kernel
    // in christoffel_fused.cu which already handles everything.
    // In a professional production environment, these would be aliased.
}

extern "C" void launch_reactive_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    int topology,
    float R_val, float r_val,
    cudaStream_t stream
) {
    // Forward to launch_christoffel_backward but with use_active=true
    extern void launch_christoffel_backward(
        const float*, const float*, const float*, const float*, 
        const float*, const float*,
        float*, float*, float*,
        float*, float*,
        int, int, int,
        float, float, float,
        bool, int, float, float, cudaStream_t
    );
    
    launch_christoffel_backward(
        grad_gamma, v, U, W, x, V_w, grad_v, grad_U, grad_W, grad_x, grad_V_w,
        batch, dim, rank, plasticity, sing_thresh, sing_strength, true, topology, R_val, r_val, stream
    );
}
