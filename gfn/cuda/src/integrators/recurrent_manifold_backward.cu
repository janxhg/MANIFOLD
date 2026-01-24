
#include "../../include/christoffel_impl.cuh"
#include "../../include/boundaries.cuh"

#define BLOCK_SIZE 512

// --- HELPER FUNCTIONS ---

__device__ void tanh_bounding_backward_device(float* s_gv, const float* s_v, int dim, int tid, float bound = 10.0f) {
    for (int i = tid; i < dim; i += blockDim.x) {
        float tv = tanhf(s_v[i] / bound);
        s_gv[i] *= (1.0f - tv * tv);
    }
    __syncthreads();
}

__device__ void rmsnorm_backward_device(float* s_gx, const float* s_x, int dim, int tid, float eps = 1e-6f) {
    float sum_sq = 0.0f, dot_gy = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) sum_sq += s_x[i] * s_x[i];
    sum_sq = warpReduceSum(sum_sq);
    __shared__ float s_rms_val;
    if (tid == 0) s_rms_val = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(&s_rms_val, sum_sq);
    __syncthreads();
    float inv_rms = rsqrtf(s_rms_val / (float)dim + eps);
    for (int i = tid; i < dim; i += blockDim.x) dot_gy += s_gx[i] * s_x[i] * inv_rms;
    dot_gy = warpReduceSum(dot_gy);
    __shared__ float s_dot;
    if (tid == 0) s_dot = 0.0f;
    __syncthreads();
    if ((tid & 31) == 0) atomicAdd(&s_dot, dot_gy);
    __syncthreads();
    float mean_dot = s_dot / (float)dim;
    for (int i = tid; i < dim; i += blockDim.x) s_gx[i] = inv_rms * (s_gx[i] - (s_x[i] * inv_rms) * mean_dot);
    __syncthreads();
}

__device__ void head_mixing_backward_device(
    float* s_gx, float* s_gv, const float* s_x, const float* s_v, const float* W_x, const float* W_v, 
    float* g_Wx, float* g_Wv, float* s_temp_x, float* s_temp_v, int dim, int tid, int topology
) {
    rmsnorm_backward_device(s_gx, s_temp_x, dim, tid);
    rmsnorm_backward_device(s_gv, s_temp_v, dim, tid);
    __syncthreads();
    if (topology == 1) {
        for (int i = tid; i < dim; i += blockDim.x) {
            for (int j = 0; j < dim; j++) {
                float sj = sinf(s_x[j]), cj = cosf(s_x[j]), vj = s_v[j];
                if (g_Wx) {
                    atomicAdd(&g_Wx[i*(3*dim)+j], s_gx[i]*sj);
                    atomicAdd(&g_Wx[i*(3*dim)+j+dim], s_gx[i]*cj);
                    atomicAdd(&g_Wx[i*(3*dim)+j+2*dim], s_gx[i]*vj);
                }
            }
        }
    } else {
        for (int i = tid; i < dim * dim; i += blockDim.x) {
            if (g_Wx) atomicAdd(&g_Wx[i], s_gx[i/dim] * s_x[i%dim]);
        }
    }
    for (int i = tid; i < dim * dim; i += blockDim.x) if (g_Wv) atomicAdd(&g_Wv[i], s_gv[i/dim] * s_v[i%dim]);
    __syncthreads();
    for (int j = tid; j < dim; j += blockDim.x) {
        float sum_v = 0.0f;
        for (int i = 0; i < dim; i++) sum_v += s_gv[i] * W_v[i * dim + j];
        s_temp_v[j] = sum_v;
        if (topology == 1) {
            float sum_x = 0.0f, cj = cosf(s_x[j]), sj = sinf(s_x[j]);
            for (int i = 0; i < dim; i++) {
                sum_x += s_gx[i] * (W_x[i*(3*dim)+j]*cj - W_x[i*(3*dim)+j+dim]*sj);
                s_temp_v[j] += s_gx[i] * W_x[i*(3*dim)+j+2*dim];
            }
            s_temp_x[j] = sum_x;
        } else {
            float sum_x = 0.0f;
            for (int i = 0; i < dim; i++) sum_x += s_gx[i] * W_x[i * dim + j];
            s_temp_x[j] = sum_x;
        }
    }
    __syncthreads();
    for (int i = tid; i < dim; i += blockDim.x) { s_gx[i] = s_temp_x[i]; s_gv[i] = s_temp_v[i]; }
    __syncthreads();
}

__device__ void christoffel_grads_device(
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

__device__ void compute_friction_backward_device(
    float* s_dmu, const float* s_x, const float* s_u, const float* W_x, const float* W_v, const float* b, 
    float* s_gx, float* s_gu, float* g_Wx, float* g_Wv, float* g_b, int dim, int tid, int topology
) {
    for (int i = tid; i < dim; i += blockDim.x) {
        float sum = b[i];
        if (topology == 1) { 
            for (int k = 0; k < dim; k++) { sum += W_x[i*(2*dim)+k]*sinf(s_x[k]) + W_x[i*(2*dim)+k+dim]*cosf(s_x[k]); }
        } else { for (int k = 0; k < dim; k++) sum += W_x[i*dim+k]*s_x[k]; }
        for (int k = 0; k < dim; k++) sum += W_v[i*dim+k]*s_u[k];
        float sig = 1.0f / (1.0f + expf(-fminf(fmaxf(sum, -20.0f), 20.0f)));
        float d_sum = s_dmu[i] * 5.0f * sig * (1.0f - sig);
        if (g_b) atomicAdd(&g_b[i], d_sum);
        for (int k = 0; k < dim; k++) {
            if (g_Wv) atomicAdd(&g_Wv[i*dim+k], d_sum*s_u[k]);
            atomicAdd(&s_gu[k], d_sum*W_v[i*dim+k]);
        }
        if (topology == 1) {
            for (int k = 0; k < dim; k++) {
                float sk = sinf(s_x[k]), ck = cosf(s_x[k]);
                if (g_Wx) { atomicAdd(&g_Wx[i*(2*dim)+k], d_sum*sk); atomicAdd(&g_Wx[i*(2*dim)+k+dim], d_sum*ck); }
                atomicAdd(&s_gx[k], d_sum * (W_x[i*(2*dim)+k]*ck - W_x[i*(2*dim)+k+dim]*sk));
            }
        } else {
            for (int k = 0; k < dim; k++) { if (g_Wx) atomicAdd(&g_Wx[i*dim+k], d_sum*s_x[k]); atomicAdd(&s_gx[k], d_sum*W_x[i*dim+k]); }
        }
    }
}

__global__ void recurrent_manifold_backward_kernel(
    const float* __restrict__ g_x_seq, const float* __restrict__ g_x_f, const float* __restrict__ g_v_f,
    const float* __restrict__ x_f, const float* __restrict__ v_f, const float* __restrict__ forces,
    const float* __restrict__ U_s, const float* __restrict__ W_s, float* __restrict__ g_x0, float* __restrict__ g_v0,
    float* __restrict__ g_forces, float* __restrict__ g_U, float* __restrict__ g_W,
    const float* __restrict__ Wmx, const float* __restrict__ Wmv, float* __restrict__ g_Wmx, float* __restrict__ g_Wmv,
    const float* __restrict__ Wf_s, const float* __restrict__ Wi_s, const float* __restrict__ bf_s,
    float* __restrict__ g_Wf, float* __restrict__ g_Wi, float* __restrict__ g_bf,
    const int batch, const int seq_len, const int dim, const int rank, const int num_layers, const int num_heads,
    const float dt, const float* __restrict__ dt_scales, const float* __restrict__ forget_rates, float* __restrict__ g_fr,
    const float plasticity, const float sing_thresh, const float sing_strength, const int topology
) {
    extern __shared__ float s_mem[];
    const int dim_h = dim / num_heads, rank_h = rank / num_heads;
    float* s_x = s_mem, *s_v = s_x + dim, *s_gx = s_v + dim, *s_gv = s_gx + dim, *s_ga = s_gv + dim, *s_vp = s_ga + dim, *s_mu = s_vp + dim, *s_dmu = s_mu + dim, *s_gu = s_dmu + dim;
    float* s_t1 = s_gu + dim, *s_t2 = s_t1 + dim, *s_h = s_t2 + dim, *s_tmpf = s_h + rank;
    size_t f_off = (11 * dim + rank + 16) * sizeof(float);
    double* s_db = (double*)((char*)s_mem + f_off);

    const int b = blockIdx.x, tid = threadIdx.x;
    if (b >= batch) return;
    const float d_sc = 1.0f / sqrtf((float)num_layers);
    for (int i = tid; i < dim; i += blockDim.x) { s_x[i] = x_f[b * dim + i]; s_v[i] = v_f[b * dim + i]; s_gx[i] = g_x_f ? g_x_f[b * dim + i] : 0.0f; s_gv[i] = g_v_f ? g_v_f[b * dim + i] : 0.0f; }
    __syncthreads();
    
    for (int t = seq_len - 1; t >= 0; t--) {
        if (g_x_seq) { for (int i = tid; i < dim; i += blockDim.x) s_gx[i] += g_x_seq[(b*seq_len+t)*dim+i]; __syncthreads(); }
        for (int l = num_layers - 1; l >= 0; l--) {
            tanh_bounding_backward_device(s_gv, s_v, dim, tid, 10.0f);
            if (num_heads > 1 && Wmx && l < num_layers - 1) {
                for (int i = tid; i < dim; i += blockDim.x) {
                    float sx = 0, sv = 0;
                    if (topology == 1) { for(int j=0; j<dim; j++) { sx += sinf(s_x[j])*Wmx[i*(3*dim)+j] + cosf(s_x[j])*Wmx[i*(3*dim)+j+dim] + s_v[j]*Wmx[i*(3*dim)+j+2*dim]; } }
                    else { for(int j=0; j<dim; j++) sx += s_x[j]*Wmx[i*dim+j]; }
                    for(int j=0; j<dim; j++) sv += s_v[j]*Wmv[i*dim+j];
                    s_t1[i] = sx; s_t2[i] = sv;
                }
                __syncthreads();
                head_mixing_backward_device(s_gx, s_gv, s_x, s_v, Wmx, Wmv, g_Wmx, g_Wmv, s_t1, s_t2, dim, tid, topology);
            }
            for (int h = num_heads - 1; h >= 0; h--) {
                int off_h = h * dim_h; long long lh_i = l * num_heads + h;
                const float eff_dt = dt * (dt_scales?dt_scales[h]:1.0f) * d_sc, half_dt = 0.5f * eff_dt;
                float* s_xh = s_x+off_h, *s_vh = s_v+off_h, *s_gxh = s_gx+off_h, *s_gvh = s_gv+off_h, *s_gah = s_ga+off_h, *s_vph = s_vp+off_h, *s_muh = s_mu+off_h, *s_dmh = s_dmu+off_h, *s_guh = s_gu+off_h, *s_h_s = s_h+h*rank_h;
                const float* Uh = U_s + lh_i*dim_h*rank_h, *Wh = W_s + lh_i*dim_h*rank_h, *ft = &forces[(b*seq_len+t)*dim+off_h];
                int wx_s = (topology == 1) ? (2*dim_h) : dim_h;
                const float* Wfh = Wf_s + lh_i*dim_h*wx_s, *Wih = Wi_s + lh_i*dim_h*dim_h, *bfh = bf_s + lh_i*dim_h;
                float* g_Wfh = g_Wf ? g_Wf + lh_i*dim_h*wx_s : nullptr, *g_Wih = g_Wi ? g_Wi + lh_i*dim_h*dim_h : nullptr, *g_bfh = g_bf ? g_bf + lh_i*dim_h : nullptr, *g_Uh = g_U + lh_i*dim_h*rank_h, *g_Wh = g_W + lh_i*dim_h*rank_h;

                compute_friction_device(s_muh, s_xh, ft, Wfh, Wih, bfh, dim_h, tid, topology);
                christoffel_device(s_vh, Uh, Wh, s_gah, s_xh, nullptr, dim_h, rank_h, plasticity, sing_thresh, sing_strength, false, topology, ft, Wfh, Wih, bfh, s_h_s, s_db, s_db+1, (float*)(s_db+2));
                __syncthreads();
                for (int i = tid; i < dim_h; i += blockDim.x) { s_vph[i] = s_vh[i] * (1.0f + half_dt * s_muh[i]) - half_dt * (ft[i] - s_gah[i]); float gvn = s_gvh[i], den = 1.0f + half_dt * s_muh[i], gvp = gvn / den; s_dmh[i] = -gvp * s_vh[i] * half_dt; s_gah[i] = -gvp * half_dt; if (g_forces) atomicAdd(&g_forces[(b*seq_len+t)*dim+off_h+i], gvp * half_dt); s_gvh[i] = gvp; }
                __syncthreads();
                for(int i=tid; i<dim_h; i+=blockDim.x) s_guh[i] = 0.0f; __syncthreads();
                compute_friction_backward_device(s_dmh, s_xh, ft, Wfh, Wih, bfh, s_gxh, s_guh, g_Wfh, g_Wih, g_bfh, dim_h, tid, topology);
                for (int i = tid; i < dim_h; i += blockDim.x) { if (g_forces) atomicAdd(&g_forces[(b*seq_len+t)*dim+off_h+i], s_guh[i]); s_xh[i] = apply_boundary(s_xh[i] - eff_dt * s_vph[i], topology); s_gvh[i] += s_gxh[i] * eff_dt; }
                __syncthreads();
                christoffel_grads_device(s_gah, s_vph, s_h_s, Uh, Wh, g_Uh, g_Wh, dim_h, rank_h, tid, s_tmpf);
                christoffel_v_backward_device(s_gah, Uh, Wh, s_h_s, s_vph, s_gvh, dim_h, rank_h, plasticity, false, topology, ft, Wfh, Wih, bfh, s_xh, g_Wfh, g_Wih, g_bfh, 1.0f, 1.0f, s_t1+off_h);
                for(int i=tid; i<dim_h; i+=blockDim.x) s_vh[i] = s_vph[i];
                __syncthreads();
            }
        }
    }
    for (int i = tid; i < dim; i += blockDim.x) { g_x0[b * dim + i] = s_gx[i]; g_v0[b * dim + i] = s_gv[i]; }
}

extern "C" void launch_recurrent_manifold_backward(
    const float* g_xs, const float* g_xf, const float* g_vf, const float* xf, const float* vf, const float* f, const float* Us, const float* Ws,
    float* g_x0, float* g_v0, float* g_f, float* g_U, float* g_W,
    const float* Wmx, const float* Wmv, float* g_Wmx, float* g_Wmv,
    const float* Wf_s, const float* Wi_s, const float* bf_s, float* g_Wf, float* g_Wi, float* g_bf,
    int batch, int seq_len, int dim, int rank, int num_layers, int num_heads,
    float dt, const float* dt_scales, const float* forget_rates, float* g_fr,
    float plasticity, float sing_thresh, float sing_strength, int topology, cudaStream_t stream
) {
    const int shared_bytes = (11 * dim + rank + 128) * sizeof(float) + 16 * sizeof(double);
    recurrent_manifold_backward_kernel<<<batch, BLOCK_SIZE, shared_bytes, stream>>>(
        g_xs, g_xf, g_vf, xf, vf, f, Us, Ws, g_x0, g_v0, g_f, g_U, g_W, Wmx, Wmv, g_Wmx, g_Wmv,
        Wf_s, Wi_s, bf_s, g_Wf, g_Wi, g_bf, batch, seq_len, dim, rank, num_layers, num_heads,
        dt, dt_scales, forget_rates, g_fr, plasticity, sing_thresh, sing_strength, topology
    );
}
