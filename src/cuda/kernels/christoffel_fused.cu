/*
 * Fused Christoffel Symbol Kernel
 * ================================
 * 
 * Computes Γ(v,v) = W * (U^T v)^2 in a single fused kernel.
 * 
 * Mathematical Operation:
 *   1. proj = U^T * v      [rank]
 *   2. sq = proj^2         [rank]
 *   3. gamma = W * sq      [dim]
 * 
 * Performance:
 *   - Fuses 3 operations into 1 kernel launch
 *   - Keeps intermediate results in registers/shared memory
 *   - Expected speedup: 2-3x over PyTorch
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define MAX_RANK 128  // Max rank for shared memory strategy

// Fused kernel: computes gamma[b, d] = sum_r W[d,r] * (sum_i U[i,r] * v[b,i])^2
__global__ void christoffel_fused_kernel(
    const float* __restrict__ v,      // [batch, dim]
    const float* __restrict__ U,      // [dim, rank]
    const float* __restrict__ W,      // [dim, rank]
    float* __restrict__ gamma,        // [batch, dim]
    const int batch,
    const int dim,
    const int rank
) {
    // Shared memory for U and W tiles
    __shared__ float s_U[MAX_RANK];
    __shared__ float s_W[MAX_RANK];
    
    const int b = blockIdx.x;  // batch index
    const int d = threadIdx.x; // output dimension index
    
    if (b >= batch || d >= dim) return;
    
    float result = 0.0f;
    
    // Load W[d, :] into shared memory (coalesced)
    for (int r = 0; r < rank; r++) {
        if (d == 0 && r < rank) {
            s_W[r] = W[d * rank + r];
        }
    }
    __syncthreads();
    
    // Main computation loop over rank dimension
    for (int r = 0; r < rank; r++) {
        // Compute U^T * v for this rank
        float proj = 0.0f;
        
        // This is the critical inner product: sum_i U[i,r] * v[b,i]
        // Each thread computes partial sum, then we reduce
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            proj += U[i * rank + r] * v[b * dim + i];
        }
        
        // Warp-level reduction
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            proj += __shfl_down_sync(0xffffffff, proj, offset);
        }
        
        // First thread in warp now has complete proj for this rank
        if (threadIdx.x % warpSize == 0) {
            s_U[r] = proj;
        }
        __syncthreads();
        
        // Now compute contribution: W[d,r] * proj^2
        if (d < dim) {
            float proj_val = s_U[r];
            result += s_W[r] * proj_val * proj_val;
        }
    }
    
    // Write result with clamping (matches PyTorch version)
    if (d < dim) {
        result = fminf(fmaxf(result, -5.0f), 5.0f);
        gamma[b * dim + d] = result;
    }
}

// Host function
torch::Tensor christoffel_fused_cuda(
    torch::Tensor v,  // [batch, dim]
    torch::Tensor U,  // [dim, rank]
    torch::Tensor W   // [dim, rank]
) {
    const int batch = v.size(0);
    const int dim = v.size(1);
    const int rank = U.size(1);
    
    TORCH_CHECK(rank <= MAX_RANK, "Rank exceeds MAX_RANK");
    
    auto gamma = torch::empty({batch, dim}, v.options());
    
    const int threads = BLOCK_SIZE;
    const int blocks = batch;
    
    christoffel_fused_kernel<<<blocks, threads>>>(
        v.data_ptr<float>(),
        U.data_ptr<float>(),
        W.data_ptr<float>(),
        gamma.data_ptr<float>(),
        batch, dim, rank
    );
    
    return gamma;
}
