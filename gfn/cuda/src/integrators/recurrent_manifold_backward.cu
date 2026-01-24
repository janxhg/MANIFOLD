#include "../../include/christoffel_impl.cuh"
#include "../../include/boundaries.cuh"

#define BLOCK_SIZE 512

// --- HELPER FUNCTIONS ---

// Device function: Backward of Head Mixing
__device__ void head_mixing_backward_device(
    float* s_gx,          
    float* s_gv,          
    const float* s_x,     
    const float* s_v,     
    const float* W_x,     
    const float* W_v,     
    float* grad_W_mix_x,  
    float* grad_W_mix_v,  
    float* s_temp_x,      
    float* s_temp_v,      
    int dim,
    int tid
) {
    __syncthreads();
    for (int i = tid; i < dim * dim; i += blockDim.x) {
        int r = i / dim;
        int c = i % dim;
        if (grad_W_mix_x) atomicAdd(&grad_W_mix_x[i], s_gx[r] * s_x[c]);
        if (grad_W_mix_v) atomicAdd(&grad_W_mix_v[i], s_gv[r] * s_v[c]);
    }
    __syncthreads();
    for (int j = tid; j < dim; j += blockDim.x) {
        float sum_x = 0.0f, sum_v = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_x += s_gx[i] * W_x[i * dim + j];
            sum_v += s_gv[i] * W_v[i * dim + j];
        }
        s_temp_x[j] = sum_x;
        s_temp_v[j] = sum_v;
    }
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) {
        s_gx[i] = s_temp_x[i];
        s_gv[i] = s_temp_v[i];
    }
    __syncthreads();
}

// Device function to accumulate gradients for U and W
__device__ void christoffel_grads_device(
    const float* s_dGamma,  
    const float* s_v,       
    const float* s_h,       
    const float* U,         
    const float* W,         
    float* grad_U,          
    float* grad_W,          
    int dim,
    int rank,
    int tid,
    float* s_temp_float   
) {
    float local_nsq = 0.0f;
    for (int r = tid; r < rank; r += blockDim.x) local_nsq += s_h[r] * s_h[r];
    for (int offset = 16; offset > 0; offset /= 2) local_nsq += __shfl_down_sync(0xffffffff, local_nsq, offset);
    if (tid == 0) *s_temp_float = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(s_temp_float, local_nsq);
    __syncthreads();
    
    float S = 1.0f / (1.0f + sqrtf(*s_temp_float) + 1e-6f);
    
    for (int i = tid; i < dim * rank; i += blockDim.x) {
        int r = i % rank;
        int d = i / rank;
        atomicAdd(&grad_W[i], s_dGamma[d] * s_h[r] * s_h[r] * S);
    }
    __syncthreads();
    
    for (int r = tid; r < rank; r += blockDim.x) {
        float sum_dw = 0.0f;
        for (int i = 0; i < dim; i++) sum_dw += s_dGamma[i] * W[i * rank + r];
        float dL_dh = sum_dw * 2.0f * s_h[r] * S;
        for (int j = 0; j < dim; j++) atomicAdd(&grad_U[j * rank + r], dL_dh * s_v[j]);
    }
    __syncthreads();
}

// Backward of Friction Gate (Periodic Mapping Aware)
__device__ void compute_friction_backward_device(
    float* s_dmu,         
    const float* s_x,     
    const float* s_u,     
    const float* W_x,     // [Dim, Dim] OR [Dim, 2*Dim]
    const float* W_u,     
    const float* b_gate,  
    float* s_gx,          
    float* s_gu,          
    float* g_W_x,         
    float* g_W_u,         
    float* g_b,           
    int dim,
    int tid,
    int topology
) {
    for (int i = tid; i < dim; i += blockDim.x) {
        // 1. Forward reconstruction of sum for Sigmoid derivative
        float sum = b_gate[i];
        if (topology == 1) { // TORUS
            for (int k = 0; k < dim; k++) {
                sum += W_x[i * (2 * dim) + k] * sinf(s_x[k]);
                sum += W_x[i * (2 * dim) + k + dim] * cosf(s_x[k]);
            }
        } else {
            for (int k = 0; k < dim; k++) sum += W_x[i * dim + k] * s_x[k];
        }
        for (int k = 0; k < dim; k++) sum += W_u[i * dim + k] * s_u[k];
        
        float sig = 1.0f / (1.0f + expf(-fminf(fmaxf(sum, -20.0f), 20.0f)));
        float d_sum = s_dmu[i] * 5.0f * sig * (1.0f - sig);
        
        // 2. Gradients w.r.t Bias and W_u (Static part)
        if (g_b) atomicAdd(&g_b[i], d_sum);
        for (int k = 0; k < dim; k++) {
            if (g_W_u) atomicAdd(&g_W_u[i * dim + k], d_sum * s_u[k]);
            atomicAdd(&s_gu[k], d_sum * W_u[i * dim + k]);
        }
        
        // 3. Gradients w.r.t W_x and s_x (Periodic Adjoint)
        if (topology == 1) { // TORUS
            for (int k = 0; k < dim; k++) {
                float sk = sinf(s_x[k]);
                float ck = cosf(s_x[k]);
                if (g_W_x) {
                    atomicAdd(&g_W_x[i * (2 * dim) + k], d_sum * sk);
                    atomicAdd(&g_W_x[i * (2 * dim) + k + dim], d_sum * ck);
                }
                // dSum/dx[k] = W_sin * cos(x) - W_cos * sin(x)
                float d_x_k = d_sum * (W_x[i * (2 * dim) + k] * ck - W_x[i * (2 * dim) + k + dim] * sk);
                atomicAdd(&s_gx[k], d_x_k);
            }
        } else { // EUCLIDEAN
            for (int k = 0; k < dim; k++) {
                if (g_W_x) atomicAdd(&g_W_x[i * dim + k], d_sum * s_x[k]);
                atomicAdd(&s_gx[k], d_sum * W_x[i * dim + k]);
            }
        }
    }
    __syncthreads();
}

// --- MAIN KERNEL ---

__global__ void recurrent_manifold_backward_kernel(
    const float* __restrict__ grad_x_seq,    
    const float* __restrict__ grad_x_final,  
    const float* __restrict__ grad_v_final,  
    const float* __restrict__ x_final,       
    const float* __restrict__ v_final,       
    const float* __restrict__ forces,        
    const float* __restrict__ U_stack,       
    const float* __restrict__ W_stack,       
    float* __restrict__ grad_x_init,         
    float* __restrict__ grad_v_init,         
    float* __restrict__ grad_forces,         
    float* __restrict__ grad_U,              
    float* __restrict__ grad_W,
    const float* __restrict__ W_mix_x,
    const float* __restrict__ W_mix_v,
    float* __restrict__ grad_W_mix_x,
    float* __restrict__ grad_W_mix_v,
    const float* __restrict__ W_forget_stack,
    const float* __restrict__ W_input_stack,
    const float* __restrict__ b_forget_stack,
    float* __restrict__ grad_W_forget,
    float* __restrict__ grad_W_input,
    float* __restrict__ grad_b_forget,
    const int batch, 
    const int seq_len,
    const int dim,         
    const int rank,
    const int num_layers,
    const int num_heads,
    const float dt,
    const float* __restrict__ dt_scales,
    const float* __restrict__ forget_rates,
    float* __restrict__ grad_forget_rates,
    const float plasticity,
    const float sing_thresh,
    const float sing_strength,
    const int topology
) {
    extern __shared__ float s_mem[];
    const int dim_per_head = dim / num_heads;
    const int head_rank = rank / num_heads;
    
    float* s_x      = s_mem;          
    float* s_v      = s_x + dim;      
    float* s_gx     = s_v + dim;      
    float* s_gv     = s_gx + dim;     
    float* s_gamma  = s_gv + dim;     
    float* s_v_prev = s_gamma + dim;  
    float* s_mu     = s_v_prev + dim; 
    float* s_dmu    = s_mu + dim;    
    float* s_gu     = s_dmu + dim;   
    float* s_temp1  = s_gu + dim;    
    float* s_temp2  = s_temp1 + dim; 
    float* s_h      = s_temp2 + dim; 
    
    float* s_temp_float = s_h + rank;

    const int b = blockIdx.x; 
    const int tid = threadIdx.x;
    if (b >= batch) return;

    const float depth_scale = 1.0f / sqrtf((float)num_layers);
    
    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_final[b * dim + i];
        s_v[i] = v_final[b * dim + i];
        s_gx[i] = grad_x_final ? grad_x_final[b * dim + i] : 0.0f;
        s_gv[i] = grad_v_final ? grad_v_final[b * dim + i] : 0.0f;
    }
    __syncthreads();
    
    for (int t = seq_len - 1; t >= 0; t--) {
        if (grad_x_seq != nullptr) {
             for (int i = tid; i < dim; i += blockDim.x) s_gx[i] += grad_x_seq[(b * seq_len + t) * dim + i];
             __syncthreads();
        }
        
        for (int l = num_layers - 1; l >= 0; l--) {
            if (num_heads > 1 && W_mix_x != nullptr && l < num_layers - 1) {
                head_mixing_backward_device(s_gx, s_gv, s_x, s_v, W_mix_x, W_mix_v, grad_W_mix_x, grad_W_mix_v, s_temp1, s_temp2, dim, tid);
            }
            
            for (int h = num_heads - 1; h >= 0; h--) {
                const int head_offset = h * dim_per_head;
                const long long layer_head_idx = l * num_heads + h;
                const float h_dt_scale = dt_scales ? dt_scales[h] : 1.0f;
                const float eff_dt = dt * h_dt_scale * depth_scale;
                const float half_dt = 0.5f * eff_dt;
                
                float* s_x_h = s_x + head_offset;
                float* s_v_h = s_v + head_offset;
                float* s_gx_h = s_gx + head_offset;
                float* s_gv_h = s_gv + head_offset;
                float* s_gamma_h = s_gamma + head_offset;
                float* s_v_prev_h = s_v_prev + head_offset;
                float* s_mu_h = s_mu + head_offset;
                float* s_dmu_h = s_dmu + head_offset;
                float* s_gu_h = s_gu + head_offset;
                float* s_h_scr = s_h + h * head_rank; 

                const float* U_h = U_stack + (layer_head_idx * dim_per_head * head_rank);
                const float* W_h = W_stack + (layer_head_idx * dim_per_head * head_rank);
                const float* f_t = &forces[(b * seq_len + t) * dim + head_offset];

                // W_forget_stack is [L*H, D, 2D] if torus
                int w_x_stride = (topology == 1) ? (2 * dim_per_head) : dim_per_head;
                const float* W_f_h = W_forget_stack + (layer_head_idx * dim_per_head * w_x_stride);
                const float* W_i_h = W_input_stack + (layer_head_idx * dim_per_head * dim_per_head);
                const float* b_f_h = b_forget_stack + (layer_head_idx * dim_per_head);
                
                float* g_W_f = grad_W_forget ? grad_W_forget + (layer_head_idx * dim_per_head * w_x_stride) : nullptr;
                float* g_W_i = grad_W_input ? grad_W_input + (layer_head_idx * dim_per_head * dim_per_head) : nullptr;
                float* g_b_f = grad_b_forget ? grad_b_forget + (layer_head_idx * dim_per_head) : nullptr;
                float* g_U_h = grad_U + (layer_head_idx * dim_per_head * head_rank);
                float* g_W_h = grad_W + (layer_head_idx * dim_per_head * head_rank);

                // --- 1. REVERSE KICK 2 ---
                compute_friction_device(s_mu_h, s_x_h, f_t, W_f_h, W_i_h, b_f_h, dim_per_head, tid, topology);
                christoffel_device(s_v_h, U_h, W_h, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, f_t, W_f_h, W_i_h, b_f_h, s_h_scr, (double*)s_temp2, (double*)s_temp2+1, (float*)s_temp2+4);
                __syncthreads();
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_v_prev_h[i] = s_v_h[i] * (1.0f + half_dt * s_mu_h[i]) - half_dt * (f_t[i] - s_gamma_h[i]);
                }
                __syncthreads();

                // ADJOINT (Kick 2)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    float gv_next = s_gv_h[i];
                    float den = 1.0f + half_dt * s_mu_h[i];
                    float gv_prev = gv_next / den;
                    s_dmu_h[i] = -gv_prev * s_v_h[i] * half_dt;
                    s_gamma_h[i] = -gv_prev * half_dt;
                    if (grad_forces) atomicAdd(&grad_forces[(b*seq_len+t)*dim+head_offset+i], gv_prev * half_dt);
                    s_gv_h[i] = gv_prev;
                }
                __syncthreads();
                
                for(int i=tid; i<dim_per_head; i+=blockDim.x) s_gu_h[i] = 0.0f;
                __syncthreads();
                compute_friction_backward_device(s_dmu_h, s_x_h, f_t, W_f_h, W_i_h, b_f_h, s_gx_h, s_gu_h, g_W_f, g_W_i, g_b_f, dim_per_head, tid, topology);
                for (int i = tid; i < dim_per_head; i += blockDim.x) if (grad_forces) atomicAdd(&grad_forces[(b*seq_len+t)*dim+head_offset+i], s_gu_h[i]);
                
                christoffel_grads_device(s_gamma_h, s_v_prev_h, s_h_scr, U_h, W_h, g_U_h, g_W_h, dim_per_head, head_rank, tid, s_temp_float);
                christoffel_v_backward_device(s_gamma_h, U_h, W_h, s_h_scr, s_v_prev_h, s_gv_h, dim_per_head, head_rank, plasticity, false, f_t, W_f_h, W_i_h, b_f_h, s_x_h, g_W_f, g_W_i, g_b_f, 1.0f, 1.0f, s_temp1);
                
                for(int i=tid; i<dim_per_head; i+=blockDim.x) s_v_h[i] = s_v_prev_h[i];
                __syncthreads();

                // --- 2. REVERSE DRIFT ---
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_x_h[i] = apply_boundary(s_x_h[i] - eff_dt * s_v_h[i], topology);
                    s_gv_h[i] += s_gx_h[i] * eff_dt;
                }
                __syncthreads();

                // --- 3. REVERSE KICK 1 ---
                compute_friction_device(s_mu_h, s_x_h, f_t, W_f_h, W_i_h, b_f_h, dim_per_head, tid, topology);
                christoffel_device(s_v_h, U_h, W_h, s_gamma_h, s_x_h, nullptr, dim_per_head, head_rank, plasticity, sing_thresh, sing_strength, false, f_t, W_f_h, W_i_h, b_f_h, s_h_scr, (double*)s_temp2, (double*)s_temp2+1, (float*)s_temp2+4);
                __syncthreads();
                
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    s_v_prev_h[i] = s_v_h[i] * (1.0f + half_dt * s_mu_h[i]) - half_dt * (f_t[i] - s_gamma_h[i]);
                }
                __syncthreads();

                // ADJOINT (Kick 1)
                for (int i = tid; i < dim_per_head; i += blockDim.x) {
                    float gv_next = s_gv_h[i];
                    float den = 1.0f + half_dt * s_mu_h[i];
                    float gv_prev = gv_next / den;
                    s_dmu_h[i] = -gv_prev * s_v_h[i] * half_dt;
                    s_gamma_h[i] = -gv_prev * half_dt;
                    if (grad_forces) atomicAdd(&grad_forces[(b*seq_len+t)*dim+head_offset+i], gv_prev * half_dt);
                    s_gv_h[i] = gv_prev;
                }
                __syncthreads();
                
                for(int i=tid; i<dim_per_head; i+=blockDim.x) s_gu_h[i] = 0.0f;
                __syncthreads();
                compute_friction_backward_device(s_dmu_h, s_x_h, f_t, W_f_h, W_i_h, b_f_h, s_gx_h, s_gu_h, g_W_f, g_W_i, g_b_f, dim_per_head, tid, topology);
                for (int i = tid; i < dim_per_head; i += blockDim.x) if (grad_forces) atomicAdd(&grad_forces[(b*seq_len+t)*dim+head_offset+i], s_gu_h[i]);
                
                christoffel_grads_device(s_gamma_h, s_v_prev_h, s_h_scr, U_h, W_h, g_U_h, g_W_h, dim_per_head, head_rank, tid, s_temp_float);
                christoffel_v_backward_device(s_gamma_h, U_h, W_h, s_h_scr, s_v_prev_h, s_gv_h, dim_per_head, head_rank, plasticity, false, f_t, W_f_h, W_i_h, b_f_h, s_x_h, g_W_f, g_W_i, g_b_f, 1.0f, 1.0f, s_temp1);
                
                for(int i=tid; i<dim_per_head; i+=blockDim.x) s_v_h[i] = s_v_prev_h[i];
                __syncthreads();
            }
        }
    }
    
    for (int i = tid; i < dim; i += blockDim.x) {
        grad_x_init[b * dim + i] = s_gx[i];
        grad_v_init[b * dim + i] = s_gv[i];
    }
}

extern "C" void launch_recurrent_manifold_backward(
    const float* grad_x_seq, const float* grad_x_final, const float* grad_v_final,
    const float* x_final, const float* v_final,
    const float* forces, const float* U_stack, const float* W_stack,
    float* grad_x_init, float* grad_v_init, float* grad_forces,
    float* grad_U, float* grad_W,
    const float* W_mix_x, const float* W_mix_v,
    float* grad_W_mix_x, float* grad_W_mix_v,
    const float* W_forget_stack, const float* W_input_stack, const float* b_forget_stack,
    float* grad_W_forget, float* grad_W_input, float* grad_b_forget,
    int batch_total, int seq_len, int dim, int rank, int num_layers, int num_heads,
    float dt, const float* dt_scales, const float* forget_rates, float* grad_forget_rates,
    float plasticity, float sing_thresh, float sing_strength,
    int topology,
    cudaStream_t stream
) {
    const int shared_bytes = (11 * dim + rank + 256) * sizeof(float);
    recurrent_manifold_backward_kernel<<<batch_total, BLOCK_SIZE, shared_bytes, stream>>>(
        grad_x_seq, grad_x_final, grad_v_final, x_final, v_final, forces, U_stack, W_stack,
        grad_x_init, grad_v_init, grad_forces, grad_U, grad_W,
        W_mix_x, W_mix_v, grad_W_mix_x, grad_W_mix_v,
        W_forget_stack, W_input_stack, b_forget_stack,
        grad_W_forget, grad_W_input, grad_b_forget,
        batch_total, seq_len, dim, rank, num_layers, num_heads,
        dt, dt_scales, forget_rates, grad_forget_rates,
        plasticity, sing_thresh, sing_strength, topology
    );
}
