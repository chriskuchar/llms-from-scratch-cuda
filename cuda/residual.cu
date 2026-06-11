#include "kernels.cuh"

// --- residual.cu ---
// Residual (skip) connections — lets gradients flow straight through.

// Forward:
// out[i] = a[i] + b[i]

__global__ void residual_forward_kernel(float* out,        // output [N]
                                        const float* a,    // first input [N]
                                        const float* b,    // second input [N]
                                        int N) {           // total elements (B*T*C)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = a[idx] + b[idx];
}

// Backward:
// da[i] += dout[i]
// db[i] += dout[i]

__global__ void residual_backward_kernel(float* da,          // gradient for first input [N], accumulated
                                         float* db,          // gradient for second input [N], accumulated
                                         const float* dout,  // upstream gradient [N]
                                         int N) {            // total elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    da[idx] += dout[idx];
    db[idx] += dout[idx];
}

void residual_forward(float* out,        // output [N]
                      const float* a,    // first input [N]
                      const float* b,    // second input [N]
                      int N) {           // total elements
    int block = 256;
    int grid = ceil_div(N, block);
    residual_forward_kernel<<<grid, block>>>(out, a, b, N);
}

void residual_backward(float* da,          // gradient for first input [N]
                       float* db,          // gradient for second input [N]
                       const float* dout,  // upstream gradient [N]
                       int N) {            // total elements
    int block = 256;
    int grid = ceil_div(N, block);
    residual_backward_kernel<<<grid, block>>>(da, db, dout, N);
}

// ======================== fp16 overloads ========================

__global__ void residual_forward_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* a,
                                              const __nv_bfloat16* b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = __float2bfloat16(__bfloat162float(a[idx]) + __bfloat162float(b[idx]));
}

__global__ void residual_backward_kernel_bf16(__nv_bfloat16* da, __nv_bfloat16* db,
                                               const __nv_bfloat16* dout, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float d = __bfloat162float(dout[idx]);
    da[idx] = __float2bfloat16(__bfloat162float(da[idx]) + d);
    db[idx] = __float2bfloat16(__bfloat162float(db[idx]) + d);
}

void residual_forward(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int N) {
    int block = 256;
    int grid = ceil_div(N, block);
    residual_forward_kernel_bf16<<<grid, block>>>(out, a, b, N);
}

void residual_backward(__nv_bfloat16* da, __nv_bfloat16* db, const __nv_bfloat16* dout, int N) {
    int block = 256;
    int grid = ceil_div(N, block);
    residual_backward_kernel_bf16<<<grid, block>>>(da, db, dout, N);
}
