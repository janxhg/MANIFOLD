
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 512

/**
 * ULTRA-PERFORMANCE Recurrent Manifold Fused Kernel (v2.3 - MULTI-HEAD + REGULARIZATION)
 * -----------------------------------------------------------------------
 * Consolidates the loop over sequence length AND the loop over layers.
 * SUPPORTS MULTIPLE ATTENTION HEADS processed in parallel!
 * NOW FEATURING: Fused Hamiltonian (Geodesic) Regularization Loss.
 * 
 * Reg Loss: Sum of ||Gamma||^2 across all heads, all layers, all timesteps.
 */
__global__ void recurrent_manifold_fused_kernel(
    float* __restrict__ x_inout,      // [B, D] - Initial/Final state
    float* __restrict__ v_inout,      // [B, D] - Initial/Final state
    const float* __restrict__ forces,  // [B, T, D] - External forces
    const float* __restrict__ U_stack, // [L*H, D/H, R] - Stacked projection matrices
    const float* __restrict__ W_stack, // [L*H, D/H, R] - Stacked weighting matrices
    float* __restrict__ x_out_seq,    // [B, T, D] - Trajectory output (Optional)
    float* __restrict__ reg_loss_out, // [B] - Cumulative ||Gamma||^2 for regularization
    const int batch,
    const int seq_len,
    const int dim,
    const int rank,
    const int num_layers,
    const int num_heads,
    const float dt,
    const float dt_scale,
    const float plasticity,
    const float sing_thresh,
    const float sing_strength
) {
    extern __shared__ float s_mem_f[];
    
    const int dim_per_head = dim / num_heads;
    
    // Memory Budget per head
    float* s_x     = s_mem_f;
    float* s_v     = s_x + dim_per_head;
    float* s_gamma = s_v + dim_per_head;
    float* s_force = s_gamma + dim_per_head;
    float* s_h     = s_force + dim_per_head;
    
    // Aligned double storage for active inference reductions
    double* s_double = (double*)(s_h + rank + (rank % 2));
    double* s_E = s_double;
    double* s_P = s_E + 1;
    float* s_M  = (float*)(s_P + 1);

    // Decode block index: batch and head
    const int b = blockIdx.x / num_heads;  // batch index
    const int h = blockIdx.x % num_heads;  // head index
    const int tid = threadIdx.x;

    if (b >= batch) return;

    // Local accumulation for regularization loss (per block/head)
    float block_reg_loss = 0.0f;

    // Offset for this head in global memory
    const int head_offset = h * dim_per_head;

    // 1. Initial Load of State (x0, v0) for this head
    for (int i = tid; i < dim_per_head; i += blockDim.x) {
        s_x[i] = x_inout[b * dim + head_offset + i];
        s_v[i] = v_inout[b * dim + head_offset + i];
    }
    __syncthreads();

    const float eff_dt = dt * dt_scale;

    // === OUTER LOOP: SEQUENCE TIME ===
    for (int t = 0; t < seq_len; t++) {
        
        // a) Load force for this token and head ONCE into shared memory
        for (int i = tid; i < dim_per_head; i += blockDim.x) {
            s_force[i] = forces[(b * seq_len + t) * dim + head_offset + i];
        }
        __syncthreads();

        // b) INNER LOOP: LAYERS (for this head)
        for (int l = 0; l < num_layers; l++) {
            
            // Index into U/W stack: layer_head_idx = l * num_heads + h
            const int layer_head_idx = l * num_heads + h;
            const float* U_l = U_stack + (layer_head_idx * dim_per_head * rank);
            const float* W_l = W_stack + (layer_head_idx * dim_per_head * rank);

            // Compute Manifold Geometry (Γ) for this head
            christoffel_device(
                s_v, U_l, W_l, s_gamma, s_x, nullptr, 
                dim_per_head, rank, plasticity, sing_thresh, sing_strength, false, 
                s_h, s_E, s_P, s_M
            );
            __syncthreads();

            // Unified State Update (Recurrent Evolution)
            float local_gamma_sq = 0.0f;
            for (int i = tid; i < dim_per_head; i += blockDim.x) {
                float g = s_gamma[i];
                local_gamma_sq += g * g;

                // v[t+1] = v[t] + dt * (F[t] - Gamma[t])
                s_v[i] += eff_dt * (s_force[i] - g);
                // x[t+1] = x[t] + dt * v[t+1]
                s_x[i] += eff_dt * s_v[i];
            }
            
            // Accumulate ||Gamma||^2 for this layer/timestep
            block_reg_loss += warpReduceSum(local_gamma_sq);
            __syncthreads();
        }

        // c) Stream back frame to trajectory buffer if requested (for Readout)
        if (x_out_seq != nullptr) {
            for (int i = tid; i < dim_per_head; i += blockDim.x) {
                x_out_seq[(b * seq_len + t) * dim + head_offset + i] = s_x[i];
            }
        }
        __syncthreads();
    }

    // 2. Final Write Back of sequence-updated state for this head
    for (int i = tid; i < dim_per_head; i += blockDim.x) {
        x_inout[b * dim + head_offset + i] = s_x[i];
        v_inout[b * dim + head_offset + i] = s_v[i];
    }

    // 3. Final Write Back of Regularization Loss
    // Each warp returns its local sum, thread 0 of the block adds to global
    if (tid == 0) {
        atomicAdd(&reg_loss_out[b], block_reg_loss);
    }
}

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state,
    const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* reg_loss,
    int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, float dt_scale, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    cudaStream_t stream
) {
    const int dim_per_head = dim / num_heads;
    const int shared_bytes = (4 * dim_per_head + rank + 16) * sizeof(float) + 2 * sizeof(double);
    
    // Launch batch * num_heads blocks
    const int num_blocks = batch * num_heads;
    
    // Zero out reg_loss before kernel
    cudaMemsetAsync(reg_loss, 0, batch * sizeof(float), stream);
    
    recurrent_manifold_fused_kernel<<<num_blocks, BLOCK_SIZE, shared_bytes, stream>>>(
        x_state, v_state, forces, U_stack, W_stack, x_out_seq, reg_loss,
        batch, seq_len, dim, rank, num_layers, num_heads, dt, dt_scale,
        plasticity, sing_thresh, sing_strength
    );
}
