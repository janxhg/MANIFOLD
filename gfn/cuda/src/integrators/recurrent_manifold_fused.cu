
#include "../../include/christoffel_impl.cuh"
#include "../../include/boundaries.cuh"

#define BLOCK_SIZE 512

/**
 * RECURRENT MANIFOLD FUSED KERNEL v5.0 (MULTI-HEAD MIXING)
 */

// Device function for Head Mixing (Linear Projection with Periodic Support)
__device__ void head_mixing_device(
    float* s_x,           
    float* s_v,           
    const float* W_x,     
    const float* W_v,     
    float* s_temp_x,      
    float* s_temp_v,      
    int dim,
    int tid,
    int topology
) {
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) {
        float sum_x = 0.0f, sum_v = 0.0f;
        if (topology == 1) { // TORUS
            for (int j = 0; j < dim; j++) {
                sum_x += sinf(s_x[j]) * W_x[i * (3 * dim) + j];
                sum_x += cosf(s_x[j]) * W_x[i * (3 * dim) + j + dim];
                sum_x += s_v[j]       * W_x[i * (3 * dim) + j + 2 * dim];
            }
        } else { // EUCLIDEAN
            for (int j = 0; j < dim; j++) sum_x += s_x[j] * W_x[i * dim + j];
        }
        for (int j = 0; j < dim; j++) sum_v += s_v[j] * W_v[i * dim + j];
        s_temp_x[i] = sum_x; s_temp_v[i] = sum_v;
    }
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) { s_x[i] = s_temp_x[i]; s_v[i] = s_temp_v[i]; }
    __syncthreads();
    rmsnorm_device(s_x, dim, tid); rmsnorm_device(s_v, dim, tid);
}

__global__ void recurrent_manifold_fused_kernel(
    float* __restrict__ x_state, float* __restrict__ v_state,
    const float* __restrict__ forces, const float* __restrict__ U_stack, const float* __restrict__ W_stack,
    float* __restrict__ x_out_seq, float* __restrict__ reg_loss_out,
    const float* __restrict__ W_mix_x, const float* __restrict__ W_mix_v,
    const float* __restrict__ W_forget_stack, const float* __restrict__ W_input_stack, const float* __restrict__ b_forget_stack,
    const int batch, const int seq_len, const int dim, const int rank, const int num_layers, const int num_heads,
    const float dt, const float* __restrict__ dt_scales, const float* __restrict__ forget_rates,
    const float plasticity, const float sing_thresh, const float sing_strength, const int topology
) {
    extern __shared__ float s_mem_f[];
    const int dim_per_head = dim / num_heads;
    const int head_rank = rank / num_heads;
    
    float* s_x = s_mem_f, *s_v = s_x + dim, *s_gamma = s_v + dim, *s_force = s_gamma + dim, *s_temp = s_force + dim, *s_temp2 = s_temp + dim, *s_h = s_temp2 + dim;
    size_t float_offset = (6 * dim + num_heads * head_rank) * sizeof(float);
    size_t align_padding = (8 - (float_offset % 8)) % 8;
    double* s_double_base = (double*)((char*)s_mem_f + float_offset + align_padding);
    
    const int b = blockIdx.x, tid = threadIdx.x;
    if (b >= batch) return;

    for (int i = tid; i < dim; i += blockDim.x) { s_x[i] = x_state[b * dim + i]; s_v[i] = v_state[b * dim + i]; }
    __syncthreads();
    
    const float depth_scale = 1.0f / sqrtf((float)num_layers);

    for (int t = 0; t < seq_len; t++) {
        for (int i = tid; i < dim; i += blockDim.x) s_force[i] = forces[(b * seq_len + t) * dim + i];
        __syncthreads();
        
        for (int l = 0; l < num_layers; l++) {
            for (int h = 0; h < num_heads; h++) {
                int head_offset = h * dim_per_head, layer_head_idx = l * num_heads + h;
                const float h_dt_s = dt_scales ? dt_scales[h] : 1.0f;
                const float eff_dt = dt * h_dt_s * depth_scale, half_dt = 0.5f * eff_dt;
                
                float* s_v_h = s_v + head_offset, *s_x_h = s_x + head_offset, *s_gamma_h = s_gamma + head_offset;
                const float* U_h = U_stack + (layer_head_idx * dim_per_head * head_rank);
                const float* W_h = W_stack + (layer_head_idx * dim_per_head * head_rank);
                int w_x_s = (topology == 1) ? (2 * dim_per_head) : dim_per_head;
                const float* W_f_h = W_forget_stack + (layer_head_idx * dim_per_head * w_x_s);
                const float* W_i_h = W_input_stack + (layer_head_idx * dim_per_head * dim_per_head);
                const float* b_f_h = b_forget_stack + (layer_head_idx * dim_per_head);
                const float* s_f_h = s_force + head_offset;
                float* s_h_s = s_h + h * head_rank; 
                double* s_E_s = s_double_base, *s_P_s = s_E_s + 1;
                float* s_M_s = (float*)(s_P_s + 1), *s_fric_h = s_temp + head_offset;

                compute_friction_device(s_fric_h, s_x_h, s_f_h, W_f_h, W_i_h, b_f_h, dim_per_head, tid, topology);
                christoffel_device(s_v_h, U_h, W_h, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, topology, s_f_h, W_f_h, W_i_h, b_f_h, s_h_s, s_E_s, s_P_s, s_M_s);
                __syncthreads();
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_v_h[i] = (s_v_h[i] + half_dt * (s_f_h[i] - s_gamma_h[i])) / (1.0f + half_dt * s_fric_h[i]);
                __syncthreads();
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_x_h[i] = apply_boundary(s_x_h[i] + eff_dt * s_v_h[i], topology);
                __syncthreads();
                compute_friction_device(s_fric_h, s_x_h, s_f_h, W_f_h, W_i_h, b_f_h, dim_per_head, tid, topology);
                christoffel_device(s_v_h, U_h, W_h, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, topology, s_f_h, W_f_h, W_i_h, b_f_h, s_h_s, s_E_s, s_P_s, s_M_s);
                __syncthreads();
                for (int i = tid; i < dim_per_head; i += blockDim.x) s_v_h[i] = (s_v_h[i] + half_dt * (s_f_h[i] - s_gamma_h[i])) / (1.0f + half_dt * s_fric_h[i]);
                __syncthreads();
            } 
            if (num_heads > 1 && W_mix_x != nullptr && l < num_layers - 1) head_mixing_device(s_x, s_v, W_mix_x, W_mix_v, s_temp, s_temp2, dim, tid, topology);
            tanh_bounding_device(s_v, dim, tid, 10.0f);
        } 
        __syncthreads();
        if (x_out_seq) for (int i = tid; i < dim; i += blockDim.x) x_out_seq[(b * seq_len + t) * dim + i] = s_x[i];
        __syncthreads();
    }
    for (int i = tid; i < dim; i += blockDim.x) { x_state[b * dim + i] = s_x[i]; v_state[b * dim + i] = s_v[i]; }
}

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state, const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* reg_loss, int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, const float* dt_scales, const float* forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    const float* W_mix_x, const float* W_mix_v, 
    const float* W_forget_stack, const float* W_input_stack, const float* b_forget_stack,
    int topology, cudaStream_t stream
) {
    const int shared_bytes = (6 * dim + rank + 128) * sizeof(float) + 8 * sizeof(double) + 32;
    cudaMemsetAsync(reg_loss, 0, batch * sizeof(float), stream);
    recurrent_manifold_fused_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        x_state, v_state, forces, U_stack, W_stack, x_out_seq, reg_loss, W_mix_x, W_mix_v,
        W_forget_stack, W_input_stack, b_forget_stack, batch, seq_len, dim, rank, num_layers, num_heads,
        dt, dt_scales, forget_rates, plasticity, sing_thresh, sing_strength, topology
    );
}
