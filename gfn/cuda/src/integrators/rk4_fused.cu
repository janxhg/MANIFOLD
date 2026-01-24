
#include "../../include/christoffel_impl.cuh"

#define BLOCK_SIZE 256

__global__ void rk4_fused_kernel(
    const float* __restrict__ x_in,
    const float* __restrict__ v_in,
    const float* __restrict__ f,
    const float* __restrict__ U,
    const float* __restrict__ W,
    float* __restrict__ x_out,
    float* __restrict__ v_out,
    float dt,
    float dt_scale,
    const int batch,
    const int dim,
    const int rank,
    const int steps,
    int topology
) {
    extern __shared__ float s_mem_f[];
    float* s_x = s_mem_f;
    float* s_v = s_x + dim;
    float* s_k1 = s_v + dim;
    float* s_k2 = s_k1 + dim;
    float* s_k3 = s_k2 + dim;
    float* s_k4 = s_k3 + dim;
    float* s_gamma = s_k4 + dim;
    float* s_h = s_gamma + dim;
    
    size_t float_offset_bytes = (7 * dim + rank) * sizeof(float);
    size_t align_pad = (8 - (float_offset_bytes % 8)) % 8;
    double* s_mem_d = (double*)((char*)s_mem_f + float_offset_bytes + align_pad);
    double* s_E = s_mem_d;
    double* s_P = s_E + 1;
    float* s_M = (float*)(s_P + 1);

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= batch) return;

    for (int i = tid; i < dim; i += blockDim.x) {
        s_x[i] = x_in[b * dim + i];
        s_v[i] = v_in[b * dim + i];
    }
    __syncthreads();

    const float eff_dt = dt * dt_scale;

    for (int s = 0; s < steps; s++) {
        christoffel_device(s_v, U, W, s_gamma, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, topology, nullptr, nullptr, nullptr, nullptr, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k1[i] = eff_dt * (f_val - s_gamma[i]);
            s_k2[i] = s_v[i] + 0.5f * s_k1[i];
        }
        __syncthreads();
        
        christoffel_device(s_k2, U, W, s_gamma, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, topology, nullptr, nullptr, nullptr, nullptr, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k2[i] = eff_dt * (f_val - s_gamma[i]);
            s_k3[i] = s_v[i] + 0.5f * s_k2[i];
        }
        __syncthreads();
        
        christoffel_device(s_k3, U, W, s_gamma, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, topology, nullptr, nullptr, nullptr, nullptr, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k3[i] = eff_dt * (f_val - s_gamma[i]);
            s_k4[i] = s_v[i] + s_k3[i];
        }
        __syncthreads();
        
        christoffel_device(s_k4, U, W, s_gamma, s_x, nullptr, dim, rank, 0.0f, 1.0f, 1.0f, false, topology, nullptr, nullptr, nullptr, nullptr, s_h, s_E, s_P, s_M);
        __syncthreads();
        for (int i = tid; i < dim; i += blockDim.x) {
            float f_val = (f != nullptr) ? f[b * dim + i] : 0.0f;
            s_k4[i] = eff_dt * (f_val - s_gamma[i]);
            s_v[i] = s_v[i] + (s_k1[i] + 2.0f * s_k2[i] + 2.0f * s_k3[i] + s_k4[i]) / 6.0f;
            s_x[i] = s_x[i] + eff_dt * s_v[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < dim; i += blockDim.x) {
        x_out[b * dim + i] = s_x[i];
        v_out[b * dim + i] = s_v[i];
    }
}

extern "C" void launch_rk4_fused(
    const float* x, const float* v, const float* f,
    const float* U, const float* W,
    float* x_new, float* v_new,
    float dt, float dt_scale,
    int batch, int dim, int rank,
    int steps, int topology,
    cudaStream_t stream
) {
    int shared = (7 * dim + rank + 16) * sizeof(float) + 2 * sizeof(double) + 16;
    rk4_fused_kernel<<<batch, BLOCK_SIZE, shared, stream>>>(
        x, v, f, U, W, x_new, v_new, dt, dt_scale, batch, dim, rank, steps, topology
    );
}
