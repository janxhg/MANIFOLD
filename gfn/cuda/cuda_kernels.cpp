
#ifdef _WIN32
#define NOMINMAX
#endif

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CRITICAL FIX for CUDA + PyTorch conflict:
#include <ATen/cuda/CUDAContext.h>

// Forward declarations of CUDA launchers (Raw C interface)
extern "C" void launch_christoffel_fused(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, int topology, cudaStream_t stream
);

extern "C" void launch_christoffel_backward(
    const float* grad_gamma, const float* v, const float* U, const float* W, 
    const float* x, const float* V_w,
    float* grad_v, float* grad_U, float* grad_W,
    float* grad_x, float* grad_V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    bool use_active, int topology, cudaStream_t stream
);

extern "C" void launch_reactive_christoffel_forward(
    const float* v, const float* U, const float* W, float* gamma,
    const float* x, const float* V_w,
    int batch, int dim, int rank,
    float plasticity, float sing_thresh, float sing_strength,
    int topology, cudaStream_t stream
);

extern "C" void launch_recurrent_manifold_fused(
    float* x_state, float* v_state,
    const float* forces, const float* U_stack, const float* W_stack,
    float* x_out_seq, float* reg_loss,
    int batch, int seq_len, int dim, int rank, int num_layers,
    float dt, const float* dt_scales, const float* forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    const float* W_mix_x, const float* W_mix_v,
    const float* W_forget_stack, const float* W_input_stack, const float* b_forget_stack,
    int topology,
    cudaStream_t stream
);

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
);

extern "C" void launch_leapfrog_fused(const float* x, const float* v, const float* f, const float* U, const float* W, float* x_new, float* v_new, float dt, float dt_scale, int batch, int dim, int rank, int steps, int topology, cudaStream_t stream);
extern "C" void launch_leapfrog_backward(const float* gx_n, const float* gv_n, const float* x, const float* v, const float* f, const float* U, const float* W, float* gx, float* gv, float* gf, float* gU, float* gW, int batch, int dim, int rank, float dt, float dt_s, int steps, int topology, cudaStream_t stream);
extern "C" void launch_euler_fused(const float* x, const float* v, const float* f, const float* U, const float* W, float* x_new, float* v_new, float dt, float dt_scale, int batch, int dim, int rank, int steps, int topology, cudaStream_t stream);

// --- PyTorch Wrappers (Tensor safe) ---

torch::Tensor christoffel_fused_cuda(torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength, int topology) {
    auto gamma = torch::empty_like(v);
    const float* x_ptr = (x.numel() > 0) ? x.data_ptr<float>() : nullptr;
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    bool use_active = (plasticity != 0.0f) || (x_ptr != nullptr && V_ptr != nullptr);
    launch_christoffel_fused(v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(), x_ptr, V_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, topology, at::cuda::getCurrentCUDAStream());
    return gamma;
}

std::vector<torch::Tensor> christoffel_backward_cuda(torch::Tensor grad_gamma, torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength, int topology) {
    auto grad_v = torch::zeros_like(v); auto grad_U = torch::zeros_like(U); auto grad_W = torch::zeros_like(W);
    torch::Tensor grad_x, grad_V_w; const float *x_ptr = nullptr, *V_ptr = nullptr; float *gx_ptr = nullptr, *gV_ptr = nullptr;
    bool use_active = false;
    if (x.numel() > 0) { grad_x = torch::zeros_like(x); x_ptr = x.data_ptr<float>(); gx_ptr = grad_x.data_ptr<float>(); use_active = true; }
    if (V_w.numel() > 0) { grad_V_w = torch::zeros_like(V_w); V_ptr = V_w.data_ptr<float>(); gV_ptr = grad_V_w.data_ptr<float>(); use_active = true; }
    if (plasticity != 0.0f) use_active = true;
    launch_christoffel_backward(grad_gamma.data_ptr<float>(), v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), x_ptr, V_ptr, grad_v.data_ptr<float>(), grad_U.data_ptr<float>(), grad_W.data_ptr<float>(), gx_ptr, gV_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, use_active, topology, at::cuda::getCurrentCUDAStream());
    return {grad_v, grad_U, grad_W, grad_x, grad_V_w};
}

std::vector<torch::Tensor> leapfrog_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps, int topology) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    const float* f_ptr = (f.numel() > 0) ? f.data_ptr<float>() : nullptr;
    launch_leapfrog_fused(x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, topology, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

std::vector<torch::Tensor> leapfrog_backward_cuda(torch::Tensor g_xn, torch::Tensor g_vn, torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps, int topology) {
    auto g_x = torch::zeros_like(x); auto g_v = torch::zeros_like(v); auto g_f = torch::zeros_like(f); auto g_U = torch::zeros_like(U); auto g_W = torch::zeros_like(W);
    const float* f_ptr = (f.numel() > 0) ? f.data_ptr<float>() : nullptr;
    float* gf_ptr = (f.numel() > 0) ? g_f.data_ptr<float>() : nullptr;
    launch_leapfrog_backward(g_xn.data_ptr<float>(), g_vn.data_ptr<float>(), x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), g_x.data_ptr<float>(), g_v.data_ptr<float>(), gf_ptr, g_U.data_ptr<float>(), g_W.data_ptr<float>(), x.size(0), x.size(1), U.size(1), dt, dt_scale, steps, topology, at::cuda::getCurrentCUDAStream());
    return {g_x, g_v, g_f, g_U, g_W};
}

std::vector<torch::Tensor> euler_fused_cuda(torch::Tensor x, torch::Tensor v, torch::Tensor f, torch::Tensor U, torch::Tensor W, float dt, float dt_scale, int steps, int topology) {
    auto x_new = torch::empty_like(x); auto v_new = torch::empty_like(v);
    const float* f_ptr = (f.numel() > 0) ? f.data_ptr<float>() : nullptr;
    launch_euler_fused(x.data_ptr<float>(), v.data_ptr<float>(), f_ptr, U.data_ptr<float>(), W.data_ptr<float>(), x_new.data_ptr<float>(), v_new.data_ptr<float>(), dt, dt_scale, x.size(0), x.size(1), U.size(1), steps, topology, at::cuda::getCurrentCUDAStream());
    return {x_new, v_new};
}

torch::Tensor reactive_christoffel_forward_cuda(torch::Tensor v, torch::Tensor U, torch::Tensor W, torch::Tensor x, torch::Tensor V_w, float plasticity, float sing_thresh, float sing_strength, int topology) {
    auto gamma = torch::empty_like(v);
    const float* x_ptr = (x.numel() > 0) ? x.data_ptr<float>() : nullptr;
    const float* V_ptr = (V_w.numel() > 0) ? V_w.data_ptr<float>() : nullptr;
    launch_reactive_christoffel_forward(v.data_ptr<float>(), U.data_ptr<float>(), W.data_ptr<float>(), gamma.data_ptr<float>(), x_ptr, V_ptr, v.size(0), v.size(1), U.size(1), plasticity, sing_thresh, sing_strength, topology, at::cuda::getCurrentCUDAStream());
    return gamma;
}

std::vector<torch::Tensor> recurrent_manifold_fused_cuda(
    torch::Tensor x_state, torch::Tensor v_state, torch::Tensor forces, torch::Tensor U_stack, torch::Tensor W_stack,
    float dt, torch::Tensor dt_scales, torch::Tensor forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    torch::Tensor mix_x, torch::Tensor mix_v,
    torch::Tensor W_forget_stack, torch::Tensor W_input_stack, torch::Tensor b_forget_stack,
    int topology
) {
    int batch = x_state.size(0); int dim = x_state.size(1); int seq_len = forces.size(1);
    int rank = U_stack.size(2); int num_layers = U_stack.size(0) / num_heads;
    auto x_out_seq = torch::empty({batch, seq_len, dim}, x_state.options());
    auto reg_loss = torch::zeros({batch}, x_state.options());
    launch_recurrent_manifold_fused(x_state.data_ptr<float>(), v_state.data_ptr<float>(), forces.data_ptr<float>(), U_stack.data_ptr<float>(), W_stack.data_ptr<float>(), x_out_seq.data_ptr<float>(), reg_loss.data_ptr<float>(), batch, seq_len, dim, rank, num_layers, dt, dt_scales.data_ptr<float>(), forget_rates.data_ptr<float>(), num_heads, plasticity, sing_thresh, sing_strength, mix_x.data_ptr<float>(), mix_v.data_ptr<float>(), W_forget_stack.data_ptr<float>(), W_input_stack.data_ptr<float>(), b_forget_stack.data_ptr<float>(), topology, at::cuda::getCurrentCUDAStream());
    return {x_state, v_state, x_out_seq, reg_loss};
}

std::vector<torch::Tensor> recurrent_manifold_backward_cuda(
    torch::Tensor grad_x_seq, torch::Tensor grad_x_final, torch::Tensor grad_v_final,
    torch::Tensor x_final, torch::Tensor v_final,
    torch::Tensor forces, torch::Tensor U_stack, torch::Tensor W_stack,
    float dt, torch::Tensor dt_scales, torch::Tensor forget_rates, int num_heads,
    float plasticity, float sing_thresh, float sing_strength,
    torch::Tensor mix_x, torch::Tensor mix_v,
    torch::Tensor W_forget_stack, torch::Tensor W_input_stack, torch::Tensor b_forget_stack,
    int topology
) {
    int batch_total = x_final.size(0); int dim = x_final.size(1); int seq_len = forces.size(1);
    int rank = U_stack.size(2); int num_layers = U_stack.size(0) / num_heads;
    auto g_x0 = torch::zeros_like(x_final); auto g_v0 = torch::zeros_like(v_final);
    auto g_f = torch::zeros_like(forces); auto g_U = torch::zeros_like(U_stack); auto g_W = torch::zeros_like(W_stack);
    auto g_mx = mix_x.defined() ? torch::zeros_like(mix_x) : torch::empty({0}, x_final.options());
    auto g_mv = mix_v.defined() ? torch::zeros_like(mix_v) : torch::empty({0}, x_final.options());
    auto g_fr = torch::zeros_like(forget_rates);
    auto g_wf = W_forget_stack.defined() ? torch::zeros_like(W_forget_stack) : torch::empty({0}, x_final.options());
    auto g_wi = W_input_stack.defined() ? torch::zeros_like(W_input_stack) : torch::empty({0}, x_final.options());
    auto g_bf = b_forget_stack.defined() ? torch::zeros_like(b_forget_stack) : torch::empty({0}, x_final.options());
    
    launch_recurrent_manifold_backward(grad_x_seq.data_ptr<float>(), grad_x_final.data_ptr<float>(), grad_v_final.data_ptr<float>(), x_final.data_ptr<float>(), v_final.data_ptr<float>(), forces.data_ptr<float>(), U_stack.data_ptr<float>(), W_stack.data_ptr<float>(), g_x0.data_ptr<float>(), g_v0.data_ptr<float>(), g_f.data_ptr<float>(), g_U.data_ptr<float>(), g_W.data_ptr<float>(), mix_x.data_ptr<float>(), mix_v.data_ptr<float>(), g_mx.data_ptr<float>(), g_mv.data_ptr<float>(), W_forget_stack.data_ptr<float>(), W_input_stack.data_ptr<float>(), b_forget_stack.data_ptr<float>(), g_wf.data_ptr<float>(), g_wi.data_ptr<float>(), g_bf.data_ptr<float>(), batch_total, seq_len, dim, rank, num_layers, num_heads, dt, dt_scales.data_ptr<float>(), forget_rates.data_ptr<float>(), g_fr.data_ptr<float>(), plasticity, sing_thresh, sing_strength, topology, at::cuda::getCurrentCUDAStream());
    return {g_x0, g_v0, g_f, g_U, g_W, g_mx, g_mv, g_fr, g_wf, g_wi, g_bf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("christoffel_fused", &christoffel_fused_cuda, py::arg("v"), py::arg("U"), py::arg("W"), py::arg("x"), py::arg("V_w"), py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"), py::arg("topology")=0);
    m.def("christoffel_backward", &christoffel_backward_cuda, py::arg("grad_gamma"), py::arg("v"), py::arg("U"), py::arg("W"), py::arg("x"), py::arg("V_w"), py::arg("plasticity"), py::arg("sing_thresh"), py::arg("sing_strength"), py::arg("topology")=0);
    m.def("reactive_christoffel_forward", &reactive_christoffel_forward_cuda);
    m.def("recurrent_manifold_fused", &recurrent_manifold_fused_cuda);
    m.def("recurrent_manifold_backward", &recurrent_manifold_backward_cuda);
    m.def("leapfrog_fused", &leapfrog_fused_cuda);
    m.def("leapfrog_backward", &leapfrog_backward_cuda);
    m.def("euler_fused", &euler_fused_cuda);
}
