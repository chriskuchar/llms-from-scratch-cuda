#include "kernels.cuh"
#include <math.h>

// --- attention.cu ---
// Grouped-Query Attention (GQA): 12 query heads, 4 KV heads — every 3 query heads share 1 KV head.

// scores[b,h,t1,t2] = (Q · K) * scale, causal-masked.
__global__ void attention_score_kernel(float* att,          // attention scores [B, n_head, T, T]
                                       const float* q,      // query [B*T, n_head * head_dim]
                                       const float* k,      // key [B*T, n_kv_head * head_dim]
                                       int B,
                                       int T,
                                       int n_head,
                                       int n_kv_head,
                                       int head_dim,
                                       float scale) {       // 1/sqrt(head_dim)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * T;
    if (idx >= total) return;

    int t2 = idx % T;
    int t1 = (idx / T) % T;
    int h  = (idx / (T * T)) % n_head;
    int b  = idx / (T * T * n_head);

    if (t2 > t1) { att[idx] = -INFINITY; return; }              // causal mask

    int kv_h = h / (n_head / n_kv_head);                        // GQA: map query head → shared KV head

    int q_offset = (b * T + t1) * (n_head * head_dim) + h * head_dim;
    int k_offset = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim;

    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        score += q[q_offset + d] * k[k_offset + d];
    }

    att[idx] = score * scale;
}

// context[b,h,t1,d] = sum_t2 probs[b,h,t1,t2] * V[b,kv_h,t2,d].
__global__ void attention_value_kernel(float* context,      // output [B*T, C]
                                       const float* att,    // softmax probs [B, n_head, T, T]
                                       const float* v,      // value [B*T, n_kv_head * head_dim]
                                       int B,
                                       int T,
                                       int n_head,
                                       int n_kv_head,
                                       int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * head_dim;
    if (idx >= total) return;

    int d  = idx % head_dim;
    int t1 = (idx / head_dim) % T;
    int h  = (idx / (head_dim * T)) % n_head;
    int b  = idx / (head_dim * T * n_head);

    int kv_h = h / (n_head / n_kv_head);                        // GQA: map query head → KV head

    float val = 0.0f;
    for (int t2 = 0; t2 <= t1; t2++) {
        int att_idx  = ((b * n_head + h) * T + t1) * T + t2;
        int v_offset = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
        val += att[att_idx] * v[v_offset];
    }

    int out_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
    context[out_idx] = val;
}

// datt[b,h,t1,t2] = sum_d dcontext[b,h,t1,d] * V[b,kv_h,t2,d], zero where causal-masked.
__global__ void attention_value_backward_datt(float* datt,            // gradient of probs [B, n_head, T, T]
                                               const float* dcontext, // upstream gradient [B*T, C]
                                               const float* v,        // value from forward [B*T, kv_dim]
                                               int B,
                                               int T,
                                               int n_head,
                                               int n_kv_head,
                                               int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * T;
    if (idx >= total) return;

    int t2 = idx % T;
    int t1 = (idx / T) % T;
    int h  = (idx / (T * T)) % n_head;
    int b  = idx / (T * T * n_head);

    if (t2 > t1) { datt[idx] = 0.0f; return; }                 // causal mask

    int kv_h = h / (n_head / n_kv_head);                        // GQA: map query head → KV head

    float val = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        int ctx_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
        int v_idx   = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
        val += dcontext[ctx_idx] * v[v_idx];
    }
    datt[idx] = val;
}

// dV[b,kv_h,t2,d] += sum over query heads in group, t1>=t2: probs[b,h,t1,t2] * dcontext[b,h,t1,d].
__global__ void attention_value_backward_dv(float* dv,              // value gradient [B*T, kv_dim], accumulated
                                             const float* att,       // softmax probs from forward [B, n_head, T, T]
                                             const float* dcontext,  // upstream gradient [B*T, C]
                                             int B,
                                             int T,
                                             int n_head,
                                             int n_kv_head,
                                             int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;             // one thread per (b, kv_h, t2, d)
    int total = B * n_kv_head * T * head_dim;
    if (idx >= total) return;

    int d    = idx % head_dim;
    int t2   = (idx / head_dim) % T;
    int kv_h = (idx / (head_dim * T)) % n_kv_head;
    int b    = idx / (head_dim * T * n_kv_head);

    int gqa_ratio = n_head / n_kv_head;                          // query heads per KV head (3)

    float val = 0.0f;
    for (int hg = 0; hg < gqa_ratio; hg++) {
        int h = kv_h * gqa_ratio + hg;
        for (int t1 = t2; t1 < T; t1++) {
            int att_idx = ((b * n_head + h) * T + t1) * T + t2;
            int ctx_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
            val += att[att_idx] * dcontext[ctx_idx];
        }
    }

    int v_idx = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
    atomicAdd(&dv[v_idx], val);                                 // multiple query heads write the same KV slot
}

// Softmax backward: dscores = att * (datt - sum_t2(datt * att)), zero where causal-masked.
__global__ void attention_softmax_backward_kernel(float* dscores,      // score gradients [B, n_head, T, T]
                                                   const float* datt,   // gradient of probs [B, n_head, T, T]
                                                   const float* att,    // softmax probs [B, n_head, T, T]
                                                   int B,
                                                   int T,
                                                   int n_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;             // one thread per (b, h, t1) softmax row
    int total = B * n_head * T;
    if (idx >= total) return;

    int t1 = idx % T;
    int h  = (idx / T) % n_head;
    int b  = idx / (T * n_head);

    int row_start = ((b * n_head + h) * T + t1) * T;

    float dot = 0.0f;
    for (int t2 = 0; t2 <= t1; t2++) {
        dot += datt[row_start + t2] * att[row_start + t2];
    }

    for (int t2 = 0; t2 <= t1; t2++) {
        dscores[row_start + t2] = att[row_start + t2] * (datt[row_start + t2] - dot);
    }
    for (int t2 = t1 + 1; t2 < T; t2++) {
        dscores[row_start + t2] = 0.0f;                        // causal mask: no gradient to future positions
    }
}

// dQ[b,h,t1,d] = scale * sum_t2 dscores[b,h,t1,t2] * K[b,kv_h,t2,d].
__global__ void attention_score_backward_dq(float* dq,              // query gradient [B*T, C]
                                             const float* dscores,   // score gradients [B, n_head, T, T]
                                             const float* k,         // key from forward [B*T, kv_dim]
                                             int B,
                                             int T,
                                             int n_head,
                                             int n_kv_head,
                                             int head_dim,
                                             float scale) {          // 1/sqrt(head_dim)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * head_dim;
    if (idx >= total) return;

    int d  = idx % head_dim;
    int t1 = (idx / head_dim) % T;
    int h  = (idx / (head_dim * T)) % n_head;
    int b  = idx / (head_dim * T * n_head);

    int kv_h = h / (n_head / n_kv_head);                        // GQA: map query head → KV head

    float val = 0.0f;
    for (int t2 = 0; t2 <= t1; t2++) {
        int score_idx = ((b * n_head + h) * T + t1) * T + t2;
        int k_idx     = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
        val += dscores[score_idx] * k[k_idx];
    }

    int q_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
    dq[q_idx] = val * scale;
}

// dK[b,kv_h,t2,d] += scale * sum over query heads in group, t1>=t2: dscores[b,h,t1,t2] * Q[b,h,t1,d].
__global__ void attention_score_backward_dk(float* dk,              // key gradient [B*T, kv_dim], accumulated
                                             const float* dscores,   // score gradients [B, n_head, T, T]
                                             const float* q,         // query from forward [B*T, C]
                                             int B,
                                             int T,
                                             int n_head,
                                             int n_kv_head,
                                             int head_dim,
                                             float scale) {          // 1/sqrt(head_dim)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;             // one thread per (b, kv_h, t2, d)
    int total = B * n_kv_head * T * head_dim;
    if (idx >= total) return;

    int d    = idx % head_dim;
    int t2   = (idx / head_dim) % T;
    int kv_h = (idx / (head_dim * T)) % n_kv_head;
    int b    = idx / (head_dim * T * n_kv_head);

    int gqa_ratio = n_head / n_kv_head;                          // query heads per KV head (3)

    float val = 0.0f;
    for (int hg = 0; hg < gqa_ratio; hg++) {
        int h = kv_h * gqa_ratio + hg;
        for (int t1 = t2; t1 < T; t1++) {
            int score_idx = ((b * n_head + h) * T + t1) * T + t2;
            int q_idx     = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
            val += dscores[score_idx] * q[q_idx];
        }
    }

    int k_idx = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
    atomicAdd(&dk[k_idx], val * scale);                         // multiple query heads write the same KV slot
}

// ========== Top-level forward ==========
void attention_forward(float* out,            // final output [B*T, C]
                       float* q,              // query buffer [B*T, C]
                       float* k,              // key buffer [B*T, kv_dim]
                       float* v,              // value buffer [B*T, kv_dim]
                       float* att_or_lse,     // standard: scores [B, n_head, T, T] | flash: lse [B, n_head, T]
                       const float* inp,      // input from residual stream [B*T, C]
                       const float* wq,       // query weight [C, C]
                       const float* wk,       // key weight [C, kv_dim]
                       const float* wv,       // value weight [C, kv_dim]
                       const float* wo,       // output weight [C, C]
                       int B,
                       int T,
                       int C,
                       int n_head,
                       int n_kv_head,
                       float rope_theta,
                       cublasHandle_t handle) {
    int head_dim = C / n_head;
    int kv_dim = n_kv_head * head_dim;
    int B_T = B * T;
    float scale = 1.0f / sqrtf((float)head_dim);

    matmul_forward(handle, q, inp, wq, B_T, C, C);              // Q = inp @ W_q
    matmul_forward(handle, k, inp, wk, B_T, kv_dim, C);         // K = inp @ W_k
    matmul_forward(handle, v, inp, wv, B_T, kv_dim, C);         // V = inp @ W_v
                    
    rope_forward(q, k, B, T, n_head, n_kv_head, head_dim, rope_theta);
    float* context;
    CUDA_CHECK(cudaMalloc(&context, B_T * C * sizeof(float)));

    #ifdef USE_FLASH_ATTN
        flash_attention_forward(context, att_or_lse, q, k, v, // fused scores→softmax→@V; writes context + lse (no T×T att matrix)
                                B, T, n_head, n_kv_head, head_dim);
    #else 
        {                                                            // scores with causal mask
            int N = B * n_head * T * T;
            int block = 256;
            int grid = ceil_div(N, block);
            attention_score_kernel<<<grid, block>>>(att_or_lse, q, k, B, T, n_head, n_kv_head,
                                                    head_dim, scale);
        }

    softmax_forward(att_or_lse, att_or_lse, B * n_head * T, T);               // softmax over last dim (T)

    {                                                            // context = probs @ V
        int N = B * n_head * T * head_dim;
        int block = 256;
        int grid = ceil_div(N, block);
        attention_value_kernel<<<grid, block>>>(context, att_or_lse, v, B, T, n_head, n_kv_head,
                                                head_dim);
    }
    #endif

    matmul_forward(handle, out, context, wo, B_T, C, C);        // out = context @ W_o

    CUDA_CHECK(cudaFree(context));
}

// ========== Top-level backward ==========
void attention_backward(float* dinp,            // input gradient [B*T, C]
                        float* dwq,             // query weight gradient [C, C]
                        float* dwk,             // key weight gradient [C, kv_dim]
                        float* dwv,             // value weight gradient [C, kv_dim]
                        float* dwo,             // output weight gradient [C, C]
                        const float* dout,      // upstream gradient [B*T, C]
                        const float* inp,       // saved input [B*T, C]
                        const float* q,         // saved query [B*T, C]
                        const float* k,         // saved key [B*T, kv_dim]
                        const float* v,         // saved value [B*T, kv_dim]
                        const float* att_or_lse,// standard: saved softmax probs [B, n_head, T, T] | flash: saved lse [B, n_head, T]
                        const float* wq,        // query weight [C, C]
                        const float* wk,        // key weight [C, kv_dim]
                        const float* wv,        // value weight [C, kv_dim]
                        const float* wo,        // output weight [C, C]
                        int B,
                        int T,
                        int C,
                        int n_head,
                        int n_kv_head,
                        float rope_theta,
                        cublasHandle_t handle) {
    int head_dim = C / n_head;
    int kv_dim = n_kv_head * head_dim;
    int B_T = B * T;
    float scale = 1.0f / sqrtf((float)head_dim);
    int block = 256;

    float* context;                                              // recomputed attention output (cheaper than storing)
    CUDA_CHECK(cudaMalloc(&context, B_T * C * sizeof(float)));
    {
        int N = B * n_head * T * head_dim;
        int grid = ceil_div(N, block);
        attention_value_kernel<<<grid, block>>>(context, att_or_lse, v, B, T, n_head, n_kv_head,
                                                head_dim);        // recompute context = att @ V
    }

    float* dcontext;
    CUDA_CHECK(cudaMalloc(&dcontext, B_T * C * sizeof(float)));
    matmul_backward(handle, dcontext, dwo, dout, context, wo, B_T, C, C);  // dcontext, dW_o

    float* datt;
    CUDA_CHECK(cudaMalloc(&datt, B * n_head * T * T * sizeof(float)));
    {
        int N_att = B * n_head * T * T;
        int grid = ceil_div(N_att, block);
        attention_value_backward_datt<<<grid, block>>>(datt, dcontext, v,
                                                        B, T, n_head, n_kv_head, head_dim);
    }

    float* dv_buf;
    CUDA_CHECK(cudaMalloc(&dv_buf, B_T * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(dv_buf, 0, B_T * kv_dim * sizeof(float))); // must be zeroed before atomicAdd accumulation
    {
        int N_v = B * n_kv_head * T * head_dim;
        int grid = ceil_div(N_v, block);
        attention_value_backward_dv<<<grid, block>>>(dv_buf, att_or_lse, dcontext,
                                                      B, T, n_head, n_kv_head, head_dim);
    }

    float* dscores;
    CUDA_CHECK(cudaMalloc(&dscores, B * n_head * T * T * sizeof(float)));
    {
        int N_rows = B * n_head * T;
        int grid = ceil_div(N_rows, block);
        attention_softmax_backward_kernel<<<grid, block>>>(dscores, datt, att_or_lse,
                                                            B, T, n_head);
    }

    float* dq_buf;
    CUDA_CHECK(cudaMalloc(&dq_buf, B_T * C * sizeof(float)));
    {
        int N_q = B * n_head * T * head_dim;
        int grid = ceil_div(N_q, block);
        attention_score_backward_dq<<<grid, block>>>(dq_buf, dscores, k,
                                                      B, T, n_head, n_kv_head, head_dim, scale);
    }

    float* dk_buf;
    CUDA_CHECK(cudaMalloc(&dk_buf, B_T * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(dk_buf, 0, B_T * kv_dim * sizeof(float))); // must be zeroed before atomicAdd accumulation
    {
        int N_k = B * n_kv_head * T * head_dim;
        int grid = ceil_div(N_k, block);
        attention_score_backward_dk<<<grid, block>>>(dk_buf, dscores, q,
                                                      B, T, n_head, n_kv_head, head_dim, scale);
    }

    rope_backward(dq_buf, dk_buf, B, T, n_head, n_kv_head, head_dim, rope_theta);

    matmul_backward(handle, dinp, dwq, dq_buf, inp, wq, B_T, C, C);  // dinp = dQ @ W_q^T, dW_q = inp^T @ dQ

    float* dinp_k;
    CUDA_CHECK(cudaMalloc(&dinp_k, B_T * C * sizeof(float)));
    matmul_backward(handle, dinp_k, dwk, dk_buf, inp, wk, B_T, kv_dim, C);

    float* dinp_v;
    CUDA_CHECK(cudaMalloc(&dinp_v, B_T * C * sizeof(float)));
    matmul_backward(handle, dinp_v, dwv, dv_buf, inp, wv, B_T, kv_dim, C);

    int N_inp = B_T * C;
    float* dinp_tmp;
    CUDA_CHECK(cudaMalloc(&dinp_tmp, N_inp * sizeof(float)));
    residual_forward(dinp_tmp, dinp, dinp_k, N_inp);            // dinp += dinp_k
    float* dinp_tmp2;
    CUDA_CHECK(cudaMalloc(&dinp_tmp2, N_inp * sizeof(float)));
    residual_forward(dinp_tmp2, dinp_tmp, dinp_v, N_inp);       // dinp += dinp_v
    CUDA_CHECK(cudaMemcpy(dinp, dinp_tmp2, N_inp * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(context));
    CUDA_CHECK(cudaFree(dcontext));
    CUDA_CHECK(cudaFree(datt));
    CUDA_CHECK(cudaFree(dscores));
    CUDA_CHECK(cudaFree(dq_buf));
    CUDA_CHECK(cudaFree(dk_buf));
    CUDA_CHECK(cudaFree(dv_buf));
    CUDA_CHECK(cudaFree(dinp_k));
    CUDA_CHECK(cudaFree(dinp_v));
    CUDA_CHECK(cudaFree(dinp_tmp));
    CUDA_CHECK(cudaFree(dinp_tmp2));
}

// ======================== bf16 overloads ========================
// bf16 I/O, fp32 internal accumulation (att/softmax also accumulate in fp32 for stability).

__global__ void attention_score_kernel_bf16(__nv_bfloat16* att, const __nv_bfloat16* q, const __nv_bfloat16* k,
    int B, int T, int n_head, int n_kv_head, int head_dim, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * T;
    if (idx >= total) return;

    int t2 = idx % T;
    int t1 = (idx / T) % T;
    int h  = (idx / (T * T)) % n_head;
    int b  = idx / (T * T * n_head);

    if (t2 > t1) { att[idx] = __float2bfloat16(-INFINITY); return; }

    int kv_h = h / (n_head / n_kv_head);
    int q_offset = (b * T + t1) * (n_head * head_dim) + h * head_dim;
    int k_offset = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim;

    float score = 0.0f;
    for (int d = 0; d < head_dim; d++)
        score += __bfloat162float(q[q_offset + d]) * __bfloat162float(k[k_offset + d]);
    att[idx] = __float2bfloat16(score * scale);
}

__global__ void attention_value_kernel_bf16(__nv_bfloat16* context, const __nv_bfloat16* att,
    const __nv_bfloat16* v, int B, int T, int n_head, int n_kv_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * head_dim;
    if (idx >= total) return;

    int d  = idx % head_dim;
    int t1 = (idx / head_dim) % T;
    int h  = (idx / (head_dim * T)) % n_head;
    int b  = idx / (head_dim * T * n_head);
    int kv_h = h / (n_head / n_kv_head);

    float val = 0.0f;
    for (int t2 = 0; t2 <= t1; t2++) {
        int att_idx  = ((b * n_head + h) * T + t1) * T + t2;
        int v_offset = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
        val += __bfloat162float(att[att_idx]) * __bfloat162float(v[v_offset]);
    }
    int out_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
    context[out_idx] = __float2bfloat16(val);
}

__global__ void attention_value_backward_datt_bf16(__nv_bfloat16* datt, const __nv_bfloat16* dcontext,
    const __nv_bfloat16* v, int B, int T, int n_head, int n_kv_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * T;
    if (idx >= total) return;

    int t2 = idx % T;
    int t1 = (idx / T) % T;
    int h  = (idx / (T * T)) % n_head;
    int b  = idx / (T * T * n_head);

    if (t2 > t1) { datt[idx] = __float2bfloat16(0.0f); return; }
    int kv_h = h / (n_head / n_kv_head);

    float val = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        int ctx_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
        int v_idx   = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
        val += __bfloat162float(dcontext[ctx_idx]) * __bfloat162float(v[v_idx]);
    }
    datt[idx] = __float2bfloat16(val);
}

__global__ void attention_value_backward_dv_bf16(__nv_bfloat16* dv, const __nv_bfloat16* att,
    const __nv_bfloat16* dcontext, int B, int T, int n_head, int n_kv_head, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_kv_head * T * head_dim;
    if (idx >= total) return;

    int d    = idx % head_dim;
    int t2   = (idx / head_dim) % T;
    int kv_h = (idx / (head_dim * T)) % n_kv_head;
    int b    = idx / (head_dim * T * n_kv_head);
    int gqa_ratio = n_head / n_kv_head;

    float val = 0.0f;
    for (int hg = 0; hg < gqa_ratio; hg++) {
        int h = kv_h * gqa_ratio + hg;
        for (int t1 = t2; t1 < T; t1++) {
            int att_idx = ((b * n_head + h) * T + t1) * T + t2;
            int ctx_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
            val += __bfloat162float(att[att_idx]) * __bfloat162float(dcontext[ctx_idx]);
        }
    }
    int v_idx = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
    atomicAddBf16(&dv[v_idx], val);
}

__global__ void attention_softmax_backward_kernel_bf16(__nv_bfloat16* dscores,
    const __nv_bfloat16* datt, const __nv_bfloat16* att, int B, int T, int n_head) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T;
    if (idx >= total) return;

    int t1 = idx % T;
    int h  = (idx / T) % n_head;
    int b  = idx / (T * n_head);
    int row_start = ((b * n_head + h) * T + t1) * T;

    float dot = 0.0f;
    for (int t2 = 0; t2 <= t1; t2++)
        dot += __bfloat162float(datt[row_start + t2]) * __bfloat162float(att[row_start + t2]);
    for (int t2 = 0; t2 <= t1; t2++) {
        float a = __bfloat162float(att[row_start + t2]);
        dscores[row_start + t2] = __float2bfloat16(a * (__bfloat162float(datt[row_start + t2]) - dot));
    }
    for (int t2 = t1 + 1; t2 < T; t2++)
        dscores[row_start + t2] = __float2bfloat16(0.0f);
}

__global__ void attention_score_backward_dq_bf16(__nv_bfloat16* dq, const __nv_bfloat16* dscores,
    const __nv_bfloat16* k, int B, int T, int n_head, int n_kv_head, int head_dim, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_head * T * head_dim;
    if (idx >= total) return;

    int d  = idx % head_dim;
    int t1 = (idx / head_dim) % T;
    int h  = (idx / (head_dim * T)) % n_head;
    int b  = idx / (head_dim * T * n_head);
    int kv_h = h / (n_head / n_kv_head);

    float val = 0.0f;
    for (int t2 = 0; t2 <= t1; t2++) {
        int score_idx = ((b * n_head + h) * T + t1) * T + t2;
        int k_idx     = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
        val += __bfloat162float(dscores[score_idx]) * __bfloat162float(k[k_idx]);
    }
    int q_idx = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
    dq[q_idx] = __float2bfloat16(val * scale);
}

__global__ void attention_score_backward_dk_bf16(__nv_bfloat16* dk, const __nv_bfloat16* dscores,
    const __nv_bfloat16* q, int B, int T, int n_head, int n_kv_head, int head_dim, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * n_kv_head * T * head_dim;
    if (idx >= total) return;

    int d    = idx % head_dim;
    int t2   = (idx / head_dim) % T;
    int kv_h = (idx / (head_dim * T)) % n_kv_head;
    int b    = idx / (head_dim * T * n_kv_head);
    int gqa_ratio = n_head / n_kv_head;

    float val = 0.0f;
    for (int hg = 0; hg < gqa_ratio; hg++) {
        int h = kv_h * gqa_ratio + hg;
        for (int t1 = t2; t1 < T; t1++) {
            int score_idx = ((b * n_head + h) * T + t1) * T + t2;
            int q_idx     = (b * T + t1) * (n_head * head_dim) + h * head_dim + d;
            val += __bfloat162float(dscores[score_idx]) * __bfloat162float(q[q_idx]);
        }
    }
    int k_idx = (b * T + t2) * (n_kv_head * head_dim) + kv_h * head_dim + d;
    atomicAddBf16(&dk[k_idx], val * scale);
}

// Top-level bf16 attention forward
void attention_forward(__nv_bfloat16* out, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
                        #ifdef USE_FLASH_ATTN
                            float* lse,
                        #else
                            __nv_bfloat16* att,
                        #endif
                        const __nv_bfloat16* inp, const __nv_bfloat16* wq, const __nv_bfloat16* wk,
                        const __nv_bfloat16* wv, const __nv_bfloat16* wo,
                        int B, int T, int C, int n_head, int n_kv_head,
                        float rope_theta, cublasHandle_t handle) {
    int head_dim = C / n_head;
    int kv_dim = n_kv_head * head_dim;
    int B_T = B * T;
    float scale = 1.0f / sqrtf((float)head_dim);

    matmul_forward(handle, q, inp, wq, B_T, C, C);
    matmul_forward(handle, k, inp, wk, B_T, kv_dim, C);
    matmul_forward(handle, v, inp, wv, B_T, kv_dim, C);

    rope_forward(q, k, B, T, n_head, n_kv_head, head_dim, rope_theta);
    __nv_bfloat16* context;
    CUDA_CHECK(cudaMalloc(&context, B_T * C * sizeof(__nv_bfloat16)));

    #ifdef USE_FLASH_ATTN
        flash_attention_forward_bf16(context, lse, q, k, v, B, T, n_head, n_kv_head, head_dim);                
    #else 
        { int N = B * n_head * T * T; int block = 256; int grid = ceil_div(N, block);
        attention_score_kernel_bf16<<<grid, block>>>(att, q, k, B, T, n_head, n_kv_head, head_dim, scale); }

        softmax_forward(att, att, B * n_head * T, T);

        { int N = B * n_head * T * head_dim; int block = 256; int grid = ceil_div(N, block);
        attention_value_kernel_bf16<<<grid, block>>>(context, att, v, B, T, n_head, n_kv_head, head_dim); }
    #endif
    matmul_forward(handle, out, context, wo, B_T, C, C);
    CUDA_CHECK(cudaFree(context));
}

// Top-level bf16 attention backward
void attention_backward(__nv_bfloat16* dinp, __nv_bfloat16* dwq, __nv_bfloat16* dwk, __nv_bfloat16* dwv, __nv_bfloat16* dwo,
                        const __nv_bfloat16* dout, const __nv_bfloat16* inp, const __nv_bfloat16* q,
                        const __nv_bfloat16* k, const __nv_bfloat16* v, const __nv_bfloat16* att,
                        const __nv_bfloat16* wq, const __nv_bfloat16* wk, const __nv_bfloat16* wv, const __nv_bfloat16* wo,
                        int B, int T, int C, int n_head, int n_kv_head,
                        float rope_theta, cublasHandle_t handle) {
    int head_dim = C / n_head;
    int kv_dim = n_kv_head * head_dim;
    int B_T = B * T;
    float scale = 1.0f / sqrtf((float)head_dim);
    int block = 256;

    __nv_bfloat16* context;
    CUDA_CHECK(cudaMalloc(&context, B_T * C * sizeof(__nv_bfloat16)));
    { int N = B * n_head * T * head_dim; int grid = ceil_div(N, block);
      attention_value_kernel_bf16<<<grid, block>>>(context, att, v, B, T, n_head, n_kv_head, head_dim); }

    __nv_bfloat16* dcontext;
    CUDA_CHECK(cudaMalloc(&dcontext, B_T * C * sizeof(__nv_bfloat16)));
    matmul_backward(handle, dcontext, dwo, dout, context, wo, B_T, C, C);

    __nv_bfloat16* datt;
    CUDA_CHECK(cudaMalloc(&datt, B * n_head * T * T * sizeof(__nv_bfloat16)));
    { int N_att = B * n_head * T * T; int grid = ceil_div(N_att, block);
      attention_value_backward_datt_bf16<<<grid, block>>>(datt, dcontext, v, B, T, n_head, n_kv_head, head_dim); }

    __nv_bfloat16* dv_buf;
    CUDA_CHECK(cudaMalloc(&dv_buf, B_T * kv_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dv_buf, 0, B_T * kv_dim * sizeof(__nv_bfloat16)));
    { int N_v = B * n_kv_head * T * head_dim; int grid = ceil_div(N_v, block);
      attention_value_backward_dv_bf16<<<grid, block>>>(dv_buf, att, dcontext, B, T, n_head, n_kv_head, head_dim); }

    __nv_bfloat16* dscores;
    CUDA_CHECK(cudaMalloc(&dscores, B * n_head * T * T * sizeof(__nv_bfloat16)));
    { int N_rows = B * n_head * T; int grid = ceil_div(N_rows, block);
      attention_softmax_backward_kernel_bf16<<<grid, block>>>(dscores, datt, att, B, T, n_head); }

    __nv_bfloat16* dq_buf;
    CUDA_CHECK(cudaMalloc(&dq_buf, B_T * C * sizeof(__nv_bfloat16)));
    { int N_q = B * n_head * T * head_dim; int grid = ceil_div(N_q, block);
      attention_score_backward_dq_bf16<<<grid, block>>>(dq_buf, dscores, k, B, T, n_head, n_kv_head, head_dim, scale); }

    __nv_bfloat16* dk_buf;
    CUDA_CHECK(cudaMalloc(&dk_buf, B_T * kv_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(dk_buf, 0, B_T * kv_dim * sizeof(__nv_bfloat16)));
    { int N_k = B * n_kv_head * T * head_dim; int grid = ceil_div(N_k, block);
      attention_score_backward_dk_bf16<<<grid, block>>>(dk_buf, dscores, q, B, T, n_head, n_kv_head, head_dim, scale); }

    rope_backward(dq_buf, dk_buf, B, T, n_head, n_kv_head, head_dim, rope_theta);

    matmul_backward(handle, dinp, dwq, dq_buf, inp, wq, B_T, C, C);

    __nv_bfloat16* dinp_k;
    CUDA_CHECK(cudaMalloc(&dinp_k, B_T * C * sizeof(__nv_bfloat16)));
    matmul_backward(handle, dinp_k, dwk, dk_buf, inp, wk, B_T, kv_dim, C);

    __nv_bfloat16* dinp_v;
    CUDA_CHECK(cudaMalloc(&dinp_v, B_T * C * sizeof(__nv_bfloat16)));
    matmul_backward(handle, dinp_v, dwv, dv_buf, inp, wv, B_T, kv_dim, C);

    int N_inp = B_T * C;
    __nv_bfloat16* dinp_tmp;
    CUDA_CHECK(cudaMalloc(&dinp_tmp, N_inp * sizeof(__nv_bfloat16)));
    residual_forward(dinp_tmp, dinp, dinp_k, N_inp);
    __nv_bfloat16* dinp_tmp2;
    CUDA_CHECK(cudaMalloc(&dinp_tmp2, N_inp * sizeof(__nv_bfloat16)));
    residual_forward(dinp_tmp2, dinp_tmp, dinp_v, N_inp);
    CUDA_CHECK(cudaMemcpy(dinp, dinp_tmp2, N_inp * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(context));
    CUDA_CHECK(cudaFree(dcontext));
    CUDA_CHECK(cudaFree(datt));
    CUDA_CHECK(cudaFree(dscores));
    CUDA_CHECK(cudaFree(dq_buf));
    CUDA_CHECK(cudaFree(dk_buf));
    CUDA_CHECK(cudaFree(dv_buf));
    CUDA_CHECK(cudaFree(dinp_k));
    CUDA_CHECK(cudaFree(dinp_v));
    CUDA_CHECK(cudaFree(dinp_tmp));
    CUDA_CHECK(cudaFree(dinp_tmp2));
}
