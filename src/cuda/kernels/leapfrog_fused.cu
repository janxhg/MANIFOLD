/*
 * Fused Leapfrog Integrator Kernel
 * =================================
 * 
 * Computes a complete symplectic Leapfrog step with inline Christoffel:
 * 
 *   v_half = v + 0.5 * dt * dt_scale * (f - Γ(v))
 *   x_new = x + dt * dt_scale * v_half
 *   v_new = v_half + 0.5 * dt * dt_scale * (f - Γ(v_half))
 * 
 * Performance:
 *   - Eliminates 8+ kernel launches per layer
 *   - All intermediate values kept in registers
 *   - Expected speedup: 4-5x
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_RANK 128

// Helper: Compute Christoffel symbol for a single velocity vector
__device__ float compute_christoffel_component(
    const float* v_local,
    const float* U,
    const float* W,
    int d,
    int dim,
    int rank
) {
    float result = 0.0f;
    
    for (int r = 0; r < rank; r++) {
        // Compute proj[r] = sum_i U[i,r] * v[i]
        float proj = 0.0f;
        for (int i = 0; i < dim; i++) {
            proj += U[i * rank + r] * v_local[i];
        }
        
        // Accumulate W[d,r] * proj^2
        result += W[d * rank + r] * proj * proj;
    }
    
    // Clamp to [-5, 5]
    return fminf(fmaxf(result, -5.0f), 5.0f);
}

__global__ void leapfrog_fused_kernel(
    const float* __restrict__ x,      // [batch, dim]
    const float* __restrict__ v,      // [batch, dim]
    const float* __restrict__ f,      // [batch, dim]
    const float* __restrict__ U,      // [dim, rank]
    const float* __restrict__ W,      // [dim, rank]
    float* __restrict__ x_new,        // [batch, dim]
    float* __restrict__ v_new,        // [batch, dim]
    const float dt,
    const float dt_scale,
    const int batch,
    const int dim,
    const int rank
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / dim;  // batch index
    const int d = idx % dim;  // dimension index
    
    if (b >= batch || d >= dim) return;
    
    // Load local copies
    extern __shared__ float shared_v[];
    float* v_local = &shared_v[threadIdx.x * dim];
    
    // Load current velocity into shared memory
    for (int i = 0; i < dim; i++) {
        v_local[i] = v[b * dim + i];
    }
    
    const float effective_dt = dt * dt_scale;
    const float x_curr = x[b * dim + d];
    const float v_curr = v[b * dim + d];
    const float f_curr = f[b * dim + d];
    
    // Step 1: Half-step velocity with Γ(v)
    float gamma_v = compute_christoffel_component(v_local, U, W, d, dim, rank);
    float v_half = v_curr + 0.5f * effective_dt * (f_curr - gamma_v);
    
    // Step 2: Full-step position
    float x_next = x_curr + effective_dt * v_half;
    
    // Update shared memory with v_half for second Christoffel computation
    v_local[d] = v_half;
    __syncthreads();
    
    // Step 3: Half-step velocity with Γ(v_half)
    float gamma_v_half = compute_christoffel_component(v_local, U, W, d, dim, rank);
    float v_next = v_half + 0.5f * effective_dt * (f_curr - gamma_v_half);
    
    // Write outputs
    x_new[b * dim + d] = x_next;
    v_new[b * dim + d] = v_next;
}

// Host function
std::vector<torch::Tensor> leapfrog_fused_cuda(
    torch::Tensor x,          // [batch, dim]
    torch::Tensor v,          // [batch, dim]
    torch::Tensor f,          // [batch, dim]
    torch::Tensor U,          // [dim, rank]
    torch::Tensor W,          // [dim, rank]
    float dt,
    float dt_scale
) {
    const int batch = x.size(0);
    const int dim = x.size(1);
    const int rank = U.size(1);
    
    auto x_new = torch::empty_like(x);
    auto v_new = torch::empty_like(v);
    
    const int threads = BLOCK_SIZE;
    const int total = batch * dim;
    const int blocks = (total + threads - 1) / threads;
    
    const int shared_mem = threads * dim * sizeof(float);
    
    leapfrog_fused_kernel<<<blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        v.data_ptr<float>(),
        f.data_ptr<float>(),
        U.data_ptr<float>(),
        W.data_ptr<float>(),
        x_new.data_ptr<float>(),
        v_new.data_ptr<float>(),
        dt, dt_scale,
        batch, dim, rank
    );
    
    return {x_new, v_new};
}
