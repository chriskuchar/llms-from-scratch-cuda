#include "kernels.cuh"
#include <math.h>

// --- swiglu.cu ---
// SwiGLU MLP: two parallel projections (gate, up); gate goes through SiLU, then
// multiplied element-wise with up. SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
//   gate   = inp @ W_gate          [B*T, ffn_hidden]
//   up     = inp @ W_up            [B*T, ffn_hidden]
//   hidden = SiLU(gate) * up       [B*T, ffn_hidden]
//   out    = hidden @ W_down       [B*T, C]

__global__ void swiglu_forward_kernel(float* hidden,        // output [N], SiLU(gate) * up
                                      const float* gate,    // gate projection [N]
                                      const float* up,      // up projection [N]
                                      int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = gate[idx];
    float sig = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sig;

    hidden[idx] = silu_g * up[idx];
}

// Backward (element-wise): SiLU'(x) = sig * (1 + x*(1 - sig)); dup = dhidden*SiLU(gate); dgate = dhidden*up*SiLU'(gate).
__global__ void swiglu_backward_kernel(float* dgate,            // gate gradient [N]
                                       float* dup,              // up gradient [N]
                                       const float* dhidden,    // upstream gradient [N]
                                       const float* gate,       // saved gate values from forward [N]
                                       const float* up,         // saved up values from forward [N]
                                       int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = gate[idx];
    float u = up[idx];
    float dh = dhidden[idx];

    float sig = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sig;

    dup[idx] = dh * silu_g;

    float silu_deriv = sig * (1.0f + g * (1.0f - sig));
    dgate[idx] = dh * u * silu_deriv;
}

void swiglu_forward(float* out,              // final MLP output [B*T, C]
                    float* gate,             // gate buffer [B*T, ffn_hidden]
                    float* up,               // up buffer [B*T, ffn_hidden]
                    float* hidden,           // hidden buffer [B*T, ffn_hidden]
                    const float* inp,        // input from residual stream [B*T, C]
                    const float* w_gate,     // gate weight [C, ffn_hidden]
                    const float* w_up,       // up weight [C, ffn_hidden]
                    const float* w_down,     // down weight [ffn_hidden, C]
                    int B_T,
                    int C,
                    int ffn_hidden,
                    cublasHandle_t handle) {
    matmul_forward(handle, gate, inp, w_gate, B_T, ffn_hidden, C);    // gate = inp @ W_gate
    matmul_forward(handle, up, inp, w_up, B_T, ffn_hidden, C);      // up = inp @ W_up

    int N = B_T * ffn_hidden;
    int block = 256;
    int grid = ceil_div(N, block);
    swiglu_forward_kernel<<<grid, block>>>(hidden, gate, up, N);     // hidden = SiLU(gate) * up

    matmul_forward(handle, out, hidden, w_down, B_T, C, ffn_hidden); // out = hidden @ W_down
}

void swiglu_backward(float* dinp,              // input gradient [B*T, C], written then accumulated
                     float* dw_gate,           // gate weight gradient [C, ffn_hidden]
                     float* dw_up,             // up weight gradient [C, ffn_hidden]
                     float* dw_down,           // down weight gradient [ffn_hidden, C]
                     const float* dout,        // upstream gradient [B*T, C]
                     const float* inp,         // saved input from forward [B*T, C]
                     const float* gate,        // saved gate from forward [B*T, ffn_hidden]
                     const float* up,          // saved up from forward [B*T, ffn_hidden]
                     const float* w_gate,      // gate weight [C, ffn_hidden]
                     const float* w_up,        // up weight [C, ffn_hidden]
                     const float* w_down,      // down weight [ffn_hidden, C]
                     int B_T,
                     int C,
                     int ffn_hidden,
                     cublasHandle_t handle) {
    int N = B_T * ffn_hidden;
    int block = 256;
    int grid = ceil_div(N, block);

    // recompute hidden = SiLU(gate) * up (cheaper than storing it)
    float* hidden;
    CUDA_CHECK(cudaMalloc(&hidden, N * sizeof(float)));
    swiglu_forward_kernel<<<grid, block>>>(hidden, gate, up, N);

    float* dhidden;
    float* dgate;
    float* dup;
    CUDA_CHECK(cudaMalloc(&dhidden, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dgate, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dup, N * sizeof(float)));

    matmul_backward(handle, dhidden, dw_down, dout, (const float*)hidden, w_down,
                    B_T, C, ffn_hidden);                             // dhidden = dout @ W_down^T, dW_down = hidden^T @ dout

    swiglu_backward_kernel<<<grid, block>>>(dgate, dup, dhidden, gate, up, N);

    matmul_backward(handle, dinp, dw_gate, dgate, inp, w_gate,
                    B_T, ffn_hidden, C);                             // dinp = dgate @ W_gate^T, dW_gate = inp^T @ dgate

    float* dinp_temp;
    CUDA_CHECK(cudaMalloc(&dinp_temp, B_T * C * sizeof(float)));
    matmul_backward(handle, dinp_temp, dw_up, dup, inp, w_up,
                    B_T, ffn_hidden, C);                             // dinp_temp = dup @ W_up^T, dW_up = inp^T @ dup

    float* dinp_sum;
    CUDA_CHECK(cudaMalloc(&dinp_sum, B_T * C * sizeof(float)));
    residual_forward(dinp_sum, dinp, dinp_temp, B_T * C);           // dinp += dinp_temp
    CUDA_CHECK(cudaMemcpy(dinp, dinp_sum, B_T * C * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(hidden));
    CUDA_CHECK(cudaFree(dhidden));
    CUDA_CHECK(cudaFree(dgate));
    CUDA_CHECK(cudaFree(dup));
    CUDA_CHECK(cudaFree(dinp_temp));
    CUDA_CHECK(cudaFree(dinp_sum));
}

// ======================== fp16 overloads ========================
// Same SwiGLU math, fp16 I/O, fp32 internal activation math.

__global__ void swiglu_forward_kernel_bf16(__nv_bfloat16* hidden, const __nv_bfloat16* gate,
                                            const __nv_bfloat16* up, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    hidden[idx] = __float2bfloat16(g * sig * u);
}

__global__ void swiglu_backward_kernel_bf16(__nv_bfloat16* dgate, __nv_bfloat16* dup,
                                             const __nv_bfloat16* dhidden, const __nv_bfloat16* gate,
                                             const __nv_bfloat16* up, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float g  = __bfloat162float(gate[idx]);
    float u  = __bfloat162float(up[idx]);
    float dh = __bfloat162float(dhidden[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sig;
    dup[idx] = __float2bfloat16(dh * silu_g);
    float silu_deriv = sig * (1.0f + g * (1.0f - sig));
    dgate[idx] = __float2bfloat16(dh * u * silu_deriv);
}

void swiglu_forward(__nv_bfloat16* out, __nv_bfloat16* gate, __nv_bfloat16* up, __nv_bfloat16* hidden,
                    const __nv_bfloat16* inp, const __nv_bfloat16* w_gate, const __nv_bfloat16* w_up,
                    const __nv_bfloat16* w_down, int B_T, int C, int ffn_hidden,
                    cublasHandle_t handle) {
    matmul_forward(handle, gate, inp, w_gate, B_T, ffn_hidden, C);
    matmul_forward(handle, up, inp, w_up, B_T, ffn_hidden, C);

    int N = B_T * ffn_hidden;
    int block = 256;
    int grid = ceil_div(N, block);
    swiglu_forward_kernel_bf16<<<grid, block>>>(hidden, gate, up, N);

    matmul_forward(handle, out, hidden, w_down, B_T, C, ffn_hidden);
}

void swiglu_backward(__nv_bfloat16* dinp, __nv_bfloat16* dw_gate, __nv_bfloat16* dw_up, __nv_bfloat16* dw_down,
                     const __nv_bfloat16* dout, const __nv_bfloat16* inp, const __nv_bfloat16* gate,
                     const __nv_bfloat16* up, const __nv_bfloat16* w_gate, const __nv_bfloat16* w_up,
                     const __nv_bfloat16* w_down, int B_T, int C, int ffn_hidden,
                     cublasHandle_t handle) {
    int N = B_T * ffn_hidden;
    int block = 256;
    int grid = ceil_div(N, block);

    __nv_bfloat16* hidden;
    CUDA_CHECK(cudaMalloc(&hidden, N * sizeof(__nv_bfloat16)));
    swiglu_forward_kernel_bf16<<<grid, block>>>(hidden, gate, up, N);

    __nv_bfloat16* dhidden;
    __nv_bfloat16* dgate;
    __nv_bfloat16* dup;
    CUDA_CHECK(cudaMalloc(&dhidden, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dgate, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dup, N * sizeof(__nv_bfloat16)));

    matmul_backward(handle, dhidden, dw_down, dout, (const __nv_bfloat16*)hidden, w_down,
                    B_T, C, ffn_hidden);

    swiglu_backward_kernel_bf16<<<grid, block>>>(dgate, dup, dhidden, gate, up, N);

    matmul_backward(handle, dinp, dw_gate, dgate, inp, w_gate,
                    B_T, ffn_hidden, C);

    __nv_bfloat16* dinp_temp;
    CUDA_CHECK(cudaMalloc(&dinp_temp, B_T * C * sizeof(__nv_bfloat16)));
    matmul_backward(handle, dinp_temp, dw_up, dup, inp, w_up,
                    B_T, ffn_hidden, C);

    __nv_bfloat16* dinp_sum;
    CUDA_CHECK(cudaMalloc(&dinp_sum, B_T * C * sizeof(__nv_bfloat16)));
    residual_forward(dinp_sum, dinp, dinp_temp, B_T * C);
    CUDA_CHECK(cudaMemcpy(dinp, dinp_sum, B_T * C * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(hidden));
    CUDA_CHECK(cudaFree(dhidden));
    CUDA_CHECK(cudaFree(dgate));
    CUDA_CHECK(cudaFree(dup));
    CUDA_CHECK(cudaFree(dinp_temp));
    CUDA_CHECK(cudaFree(dinp_sum));
}
