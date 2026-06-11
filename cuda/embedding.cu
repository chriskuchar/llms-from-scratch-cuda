#include "kernels.cuh"

// --- embedding.cu ---
// Embedding lookup — converts token IDs into dense vectors.
// Each token ID indexes one row of the embedding table wte[V, C].

// Forward:
// out[bt, c] = wte[input_ids[bt], c]

__global__ void embedding_forward_kernel(float* out,              // output [B*T, C]
                                         const float* wte,        // embedding table [V, C]
                                         const int* input_ids,    // token IDs [B*T]
                                         int B,
                                         int T,
                                         int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;

    int bt = idx / C;
    int c  = idx % C;
    int token_id = input_ids[bt];

    out[idx] = wte[token_id * C + c];
}

// Backward:
// dwte[input_ids[bt], c] += dout[bt, c]

__global__ void embedding_backward_kernel(float* dwte,            // embedding gradient [V, C], accumulated
                                          const float* dout,      // upstream gradient [B*T, C]
                                          const int* input_ids,   // token IDs [B*T]
                                          int B,
                                          int T,
                                          int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;

    int bt = idx / C;
    int c  = idx % C;
    int token_id = input_ids[bt];

    atomicAdd(&dwte[token_id * C + c], dout[idx]);
}

void embedding_forward(float* out,              // output [B*T, C]
                       const float* wte,        // embedding table [V, C]
                       const int* input_ids,    // token IDs [B*T]
                       int B,
                       int T,
                       int C) {
    int N = B * T * C;
    int block = 256;
    int grid = ceil_div(N, block);
    embedding_forward_kernel<<<grid, block>>>(out, wte, input_ids, B, T, C);
}

void embedding_backward(float* dwte,            // embedding gradient [V, C]
                        const float* dout,      // upstream gradient [B*T, C]
                        const int* input_ids,   // token IDs [B*T]
                        int B,
                        int T,
                        int C) {
    int N = B * T * C;
    int block = 256;
    int grid = ceil_div(N, block);
    embedding_backward_kernel<<<grid, block>>>(dwte, dout, input_ids, B, T, C);
}

// ======================== fp16 overloads ========================

__global__ void embedding_forward_kernel_bf16(__nv_bfloat16* out, const __nv_bfloat16* wte,
                                               const int* input_ids,
                                               int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;
    int bt = idx / C;
    int c  = idx % C;
    int token_id = input_ids[bt];
    out[idx] = wte[token_id * C + c];
}

__global__ void embedding_backward_kernel_bf16(__nv_bfloat16* dwte, const __nv_bfloat16* dout,
                                                const int* input_ids,
                                                int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;
    int bt = idx / C;
    int c  = idx % C;
    int token_id = input_ids[bt];
    atomicAddBf16(&dwte[token_id * C + c], __bfloat162float(dout[idx]));
}

void embedding_forward(__nv_bfloat16* out, const __nv_bfloat16* wte, const int* input_ids,
                       int B, int T, int C) {
    int N = B * T * C;
    int block = 256;
    int grid = ceil_div(N, block);
    embedding_forward_kernel_bf16<<<grid, block>>>(out, wte, input_ids, B, T, C);
}

void embedding_backward(__nv_bfloat16* dwte, const __nv_bfloat16* dout, const int* input_ids,
                        int B, int T, int C) {
    int N = B * T * C;
    int block = 256;
    int grid = ceil_div(N, block);
    embedding_backward_kernel_bf16<<<grid, block>>>(dwte, dout, input_ids, B, T, C);
}

// Mixed: fp32 gradient output, fp16 activation input.
// Prevents scatter-add overflow for high-frequency tokens.
__global__ void embedding_backward_kernel_mixed(float* dwte, const __nv_bfloat16* dout,
                                                const int* input_ids,
                                                int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;
    int bt = idx / C;
    int c  = idx % C;
    int token_id = input_ids[bt];
    atomicAdd(&dwte[token_id * C + c], __bfloat162float(dout[idx]));
}

void embedding_backward(float* dwte, const __nv_bfloat16* dout, const int* input_ids,
                        int B, int T, int C) {
    int N = B * T * C;
    int block = 256;
    int grid = ceil_div(N, block);
    embedding_backward_kernel_mixed<<<grid, block>>>(dwte, dout, input_ids, B, T, C);
}