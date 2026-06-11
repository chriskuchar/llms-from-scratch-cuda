#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "tensor.h"

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;

#ifdef __CUDACC__
// Portable atomicAdd for __nv_bfloat16 (CAS-based, works on all SM >= 8.0)
__device__ inline void atomicAddBf16(
    __nv_bfloat16* addr,
    float val) {

    unsigned int* addr_as_uint = (unsigned int*)((size_t)addr & ~3ULL);
    unsigned int old_val = *addr_as_uint;
    unsigned int assumed;

    bool is_high = ((size_t)addr & 2) != 0;

    do {
        assumed = old_val;

        __nv_bfloat16 h;
        if (is_high) h = __ushort_as_bfloat16((unsigned short)(assumed >> 16));
        else         h = __ushort_as_bfloat16((unsigned short)(assumed & 0xFFFF));

        float updated = __bfloat162float(h) + val;
        unsigned short new_bits = __bfloat16_as_ushort(__float2bfloat16(updated));

        unsigned int new_val;
        if (is_high) new_val = (assumed & 0x0000FFFF) | ((unsigned int)new_bits << 16);
        else         new_val = (assumed & 0xFFFF0000) | (unsigned int)new_bits;

        old_val = atomicCAS(addr_as_uint, assumed, new_val);
    } while (old_val != assumed);
}
#endif

// integer ceiling division (rounds up)
inline int ceil_div(int a, int b) { return (a + b -1) / b; }  

// matmul.cu
//C[M,N] = A[M,K] @ B[K,N]
void matmul_forward(cublasHandle_t handle,  // cuBLAS handle
    float* C,               // output [M, N]
    const float* A,         // left input [M, K]
    const float* B,         // right input [K, N]
    int M,                  // rows of A and C
    int N,                  // cols of B and C
    int K);                 // inner dimension


//dA[M,K] = dC[M,N] @ B^T, dB[K,N] = A^T @ dC[M,N]
void matmul_backward(cublasHandle_t handle, // cuBLAS handle
    float* dA,              // left input gradient [M, K]
    float* dB,              // right input gradient [K, N]
    const float* dC,        // upstream gradient [M, N]
    const float* A,         // saved left input [M, K]
    const float* B,         // saved right input [K, N]
    int M,                  // rows
    int N,                  // cols
    int K);                 // inner dimension


// --- embedding.cu ---
// out[B,T,C] = wte[input_ids[B,T]]   (no position embedding — RoPE handles it)
void embedding_forward(float* out,              // output [B*T, C]
    const float* wte,                           // embedding table [V, C]
    const int* input_ids,                       // token IDs [B*T]
    int B,                                      // batch size
    int T,                                      // sequence length
    int C);                                     // embedding dim (768)

// dwte[input_ids[B,T]] += dout[B,T,C]   (scatter-add gradients back into embedding table)
void embedding_backward(float* dwte,            // embedding gradient [V, C], accumulated
    const float* dout,                         // upstream gradient [B*T, C]
    const int* input_ids,                      // token IDs [B*T]
    int B,                                     // batch size
    int T,                                     // sequence length
    int C);                                    // embedding dim

// --- rmsnorm.cu ---
// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
void rmsnorm_forward(float* out,                // normalized output [B*T, C]
    float* rrms,                                // saved 1/rms per row [B*T]
    const float* inp,                           // input tensor [B*T, C]
    const float* weight,                        // learnable scale [C]
    float eps,                                  // stability constant (1e-5)
    int B_T,                                    // total rows (batch * seq_len)
    int C);                                     // embedding dim (768)

// Backward through RMSNorm. Given dout, compute:
//   dweight[c] += sum over rows of: dout[row][c] * inp[row][c] * rrms[row]
//   dinp[row][c] = weight[c] * rrms * (dout[row][c] - inp[row][c] * rrms^2 * (1/C) * sum(dout * weight * inp))    
void rmsnorm_backward(float* dinp,              // input gradient [B*T, C]
    float* dweight,                            // weight gradient [C], accumulated
    const float* dout,                         // upstream gradient [B*T, C]
    const float* inp,                          // saved input [B*T, C]
    const float* weight,                       // learnable scale [C]
    const float* rrms,                         // saved 1/rms from forward [B*T]
    int B_T,                                   // total rows
    int C);                                    // embedding dim
// dinp is coupled across the row: the RMS depends on every element, which is why the
// sum(dout * weight * inp) term appears (not a simple element-wise backward).

// --- rope.cu ---
// Forward: For each dimension pair (2i, 2i+1) at sequence position t:
//            freq = 1 / (theta ^ (2i / head_dim))
//            angle = t * freq
//            q[2i]_new   = q[2i]*cos(angle) - q[2i+1]*sin(angle)
//            q[2i+1]_new = q[2i]*sin(angle) + q[2i+1]*cos(angle)
//          Applied in-place to Q (n_head heads) and K (n_kv_head heads).
//          This is a 2D rotation that encodes position without learned parameters.
//

void rope_forward(float* q,                    // query [B*T, C], modified in-place
    float* k,                                  // key [B*T, kv_dim], modified in-place
    int B,                                     // batch size
    int T,                                     // sequence length
    int n_head,                                // query heads (12)
    int n_kv_head,                             // KV heads (4)
    int head_dim,                              // dims per head (64)
    float theta);                              // RoPE frequency base (10000.0)

// Backward: Inverse rotation (negate the angle):
//            dq[2i]   = dq_out[2i]*cos(angle) + dq_out[2i+1]*sin(angle)
//            dq[2i+1] = -dq_out[2i]*sin(angle) + dq_out[2i+1]*cos(angle)
//          Same for dk. (Rotation is orthogonal, so inverse = transpose = negate angle.)
void rope_backward(float* dq,                  // query gradient [B*T, C], modified in-place
    float* dk,                                // key gradient [B*T, kv_dim], modified in-place
    int B,                                    // batch size
    int T,                                    // sequence length
    int n_head,                               // query heads
    int n_kv_head,                            // KV heads
    int head_dim,                             // dims per head
    float theta);                             // RoPE frequency base

// --- attention.cu ---
// Forward (Grouped-Query Attention):
//   1. Q[B,T,C]         = inp @ W_q          (C = n_head * head_dim)
//   2. K[B,T,kv_dim]    = inp @ W_k          (kv_dim = n_kv_head * head_dim)
//   3. V[B,T,kv_dim]    = inp @ W_v
//   4. Apply RoPE to Q and K
//   5. Reshape Q to [B, n_head, T, hd], K and V to [B, n_kv_head, T, hd]
//   6. For GQA: broadcast each K,V head across (n_head/n_kv_head) query heads
//   7. scores = Q @ K^T / sqrt(head_dim)
//   8. Apply causal mask: scores[i][j] = -inf where j > i
//   9. probs = softmax(scores)
//  10. context = probs @ V
//  11. Reshape to [B, T, C]
//  12. out = context @ W_o
//
void attention_forward(float* out,              // final output [B*T, C]
    float* q,                              // query buffer [B*T, C]
    float* k,                              // key buffer [B*T, kv_dim]
    float* v,                              // value buffer [B*T, kv_dim]
    float* att_or_lse,                     // scores [B,nh,T,T] (standard) or lse [B,nh,T] (flash)
    const float* inp,                      // input from residual stream [B*T, C]
    const float* wq,                       // query weight [C, C]
    const float* wk,                       // key weight [C, kv_dim]
    const float* wv,                       // value weight [C, kv_dim]
    const float* wo,                       // output weight [C, C]
    int B,                                 // batch size
    int T,                                 // sequence length
    int C,                                 // embedding dim (768)
    int n_head,                            // query heads (12)
    int n_kv_head,                         // KV heads (4)
    float rope_theta,                      // RoPE base (10000.0)
    cublasHandle_t handle);                // cuBLAS handle
// Backward: reverse all 12 steps.
//   dcontext = dout @ W_o^T,  dW_o = context^T @ dout
//   dprobs = dcontext_heads @ V^T,  dV = probs^T @ dcontext_heads
//   dscores = probs * (dprobs - sum(dprobs * probs))   (softmax backward)
//   dscores masked (zero where causal mask was -inf)
//   dscores /= sqrt(head_dim)
//   dQ = dscores @ K,  dK = dscores^T @ Q
//   Backward through RoPE on dQ and dK
//   dinp += dQ @ W_q^T,  dW_q = inp^T @ dQ   (and same for K, V)
void attention_backward(float* dinp,            // input gradient [B*T, C]
    float* dwq,                           // query weight gradient [C, C]
    float* dwk,                           // key weight gradient [C, kv_dim]
    float* dwv,                           // value weight gradient [C, kv_dim]
    float* dwo,                           // output weight gradient [C, C]
    const float* dout,                    // upstream gradient [B*T, C]
    const float* inp,                     // saved input [B*T, C]
    const float* q,                       // saved query [B*T, C]
    const float* k,                       // saved key [B*T, kv_dim]
    const float* v,                       // saved value [B*T, kv_dim]
    const float* att_or_lse,              // probs [B,nh,T,T] (standard) or lse [B,nh,T] (flash)
    const float* wq,                      // query weight [C, C]
    const float* wk,                      // key weight [C, kv_dim]
    const float* wv,                      // value weight [C, kv_dim]
    const float* wo,                      // output weight [C, C]
    int B,                                // batch size
    int T,                                // sequence length
    int C,                                // embedding dim
    int n_head,                           // query heads
    int n_kv_head,                        // KV heads
    float rope_theta,                     // RoPE base
    cublasHandle_t handle);               // cuBLAS handle

// --- swiglu.cu ---
// Forward: gate   = inp @ W_gate            [B*T, ffn_hidden]
//          up     = inp @ W_up              [B*T, ffn_hidden]
//          hidden = SiLU(gate) * up         where SiLU(x) = x / (1 + exp(-x))
//          out    = hidden @ W_down         [B*T, C]
//
void swiglu_forward(float* out,                 // mlp output [B*T, C]
    float* gate,                              // gate projection [B*T, ffn_hidden]
    float* up,                                // up projection [B*T, ffn_hidden]
    float* hidden,                            // SiLU(gate)*up [B*T, ffn_hidden]
    const float* inp,                         // input from rmsnorm [B*T, C]
    const float* w_gate,                      // gate weight [C, ffn_hidden]
    const float* w_up,                        // up weight [C, ffn_hidden]
    const float* w_down,                      // down weight [ffn_hidden, C]
    int B_T,                                  // batch * seq_len
    int C,                                    // embedding dim (768)
    int ffn_hidden,                           // intermediate size (2048)
    cublasHandle_t handle);                   // cuBLAS handle
// Backward: dhidden = dout @ W_down^T,  dW_down = hidden^T @ dout
//           dgate = dhidden * up * SiLU'(gate)
//                   where SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//           dup   = dhidden * SiLU(gate)
//           dinp += dgate @ W_gate^T,  dW_gate = inp^T @ dgate
//           dinp += dup @ W_up^T,      dW_up   = inp^T @ dup
void swiglu_backward(float* dinp,               // input gradient [B*T, C]
    float* dw_gate,                          // gate weight gradient [C, ffn_hidden]
    float* dw_up,                            // up weight gradient [C, ffn_hidden]
    float* dw_down,                          // down weight gradient [ffn_hidden, C]
    const float* dout,                       // upstream gradient [B*T, C]
    const float* inp,                        // saved input [B*T, C]
    const float* gate,                       // saved gate [B*T, ffn_hidden]
    const float* up,                         // saved up [B*T, ffn_hidden]
    const float* w_gate,                     // gate weight [C, ffn_hidden]
    const float* w_up,                       // up weight [C, ffn_hidden]
    const float* w_down,                     // down weight [ffn_hidden, C]
    int B_T,                                 // batch * seq_len
    int C,                                   // embedding dim
    int ffn_hidden,                          // intermediate size
    cublasHandle_t handle);                  // cuBLAS handle
// --- softmax.cu ---
// Forward: For each row of length cols:
//            max_val = max(row)
//            out[i] = exp(row[i] - max_val) / sum(exp(row - max_val))
//          The max subtraction prevents overflow — mathematically identical to plain softmax.
//
// (Backward is typically fused into attention or cross_entropy backward.)
void softmax_forward(float* output,             // softmax output [rows, cols]
    const float* input,                         // raw scores [rows, cols]
    int rows,                                   // number of rows
    int cols);                                  // elements per row
// --- cross_entropy.cu ---
// Forward: probs = softmax(logits[b,t,:])       over vocab dimension V
//          loss[b,t] = -log(probs[target])
//          total_loss = mean over B*T positions
//
void crossentropy_forward(float* losses,        // per-position loss [B*T]
    const float* logits,                // raw model output [B*T, V]
    const int* targets,                 // correct token IDs [B*T]
    int B,                              // batch size
    int T,                              // sequence length
    int V);                             // vocab size (32000)

// Backward (fused softmax + cross-entropy):
//          dlogits[b,t,v] = probs[b,t,v] - (1 if v == target else 0)
//          Scale by 1/(B*T) to match the mean reduction.
void crossentropy_softmax_backward(float* dlogits, // logit gradients [B*T, V]
    const float* probs,        // softmax probs [B*T, V]
    const int* targets,        // correct token IDs [B*T]
    int B,                     // batch size
    int T,                     // sequence length
    int V);                    // vocab size

// --- residual.cu ---
// Forward: out[i] = a[i] + b[i]
//
void residual_forward(float* out,               // output [N]
    const float* a,                             // first input [N]
    const float* b,                             // second input [N]
    int N);                                     // total elements (B*T*C)// Backward: da[i] += dout[i]
//           db[i] += dout[i]
//           Gradient flows equally to both branches (addition duplicates the gradient).
void residual_backward(float* da,               // first input gradient [N], accumulated
    float* db,                                  // second input gradient [N], accumulated
    const float* dout,                          // upstream gradient [N]
    int N);                                     // total elements

// --- adamw.cu ---
// Per-parameter update:
//   m = beta1 * m + (1 - beta1) * grad              (first moment / momentum)
//   v = beta2 * v + (1 - beta2) * grad^2            (second moment / velocity)
//   m_hat = m / (1 - beta1^t)                        (bias correction)
//   v_hat = v / (1 - beta2^t)                        (bias correction)
//   param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
//   The weight_decay * param term is decoupled weight decay (AdamW, not Adam).
void adamw_update(float* params,                // model weights [N], updated in-place
    const float* grads,                         // gradients [N]
    float* m,                                   // first moment buffer [N]
    float* v,                                   // second moment buffer [N]
    float lr,                                   // learning rate (e.g. 3e-4)
    float beta1,                                // first moment decay (0.9)
    float beta2,                                // second moment decay (0.95)
    float eps,                                  // prevents div-by-zero (1e-8)
    float weight_decay,                         // decoupled weight decay (0.1)
    int t,                                      // current step (1-indexed, for bias correction)
    int N);                                     // total number of parameters


// --- flash_attention.cu ---
// Fused causal attention: QK^T / sqrt(d) → softmax → @V
// Q, K, V must already have RoPE applied.
// Does NOT materialize [B, n_head, T, T].
void flash_attention_forward(
    float* context,        // output context [B*T, C] (probs @ V, per query head)
    float* lse,            // log-sum-exp per row [B, n_head, T], saved for backward
    const float* q,        // query [B*T, C]            (C = n_head * head_dim)
    const float* k,        // key   [B*T, kv_dim]       (kv_dim = n_kv_head * head_dim)
    const float* v,        // value [B*T, kv_dim]
    int B,                 // batch size
    int T,                 // sequence length
    int n_head,            // query heads (12)
    int n_kv_head,         // KV heads (4, GQA)
    int head_dim);         // dims per head (64)

void flash_attention_backward(
    float* dq,             // query gradient [B*T, C]
    float* dk,             // key gradient   [B*T, kv_dim], must be zeroed before call (atomicAdd)
    float* dv,             // value gradient [B*T, kv_dim], must be zeroed before call (atomicAdd)
    const float* dcontext, // upstream gradient [B*T, C]
    const float* q,        // saved query [B*T, C]
    const float* k,        // saved key   [B*T, kv_dim]
    const float* v,        // saved value [B*T, kv_dim]
    const float* lse,      // saved log-sum-exp from forward [B, n_head, T]
    int B,                 // batch size
    int T,                 // sequence length
    int n_head,            // query heads
    int n_kv_head,         // KV heads
    int head_dim);         // dims per head

// ======================== fp16 overloads (mixed precision) ========================
// Same functions but with __nv_bfloat16* for storage. Internal math stays fp32.
// Compiler picks the right version based on whether you pass float* or __nv_bfloat16*.

// matmul — uses cublasGemmEx with tensor cores
void matmul_forward(cublasHandle_t handle, __nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
    int M, int N, int K);
void matmul_backward(cublasHandle_t handle, __nv_bfloat16* dA, __nv_bfloat16* dB,
    const __nv_bfloat16* dC, const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K);

// embedding
void embedding_forward(__nv_bfloat16* out, const __nv_bfloat16* wte, const int* input_ids, int B, int T, int C);
void embedding_backward(__nv_bfloat16* dwte, const __nv_bfloat16* dout, const int* input_ids, int B, int T, int C);
void embedding_backward(float* dwte, const __nv_bfloat16* dout, const int* input_ids, int B, int T, int C);

// rmsnorm — rrms stays float* (one scalar per row, needs precision)
void rmsnorm_forward(__nv_bfloat16* out, float* rrms, const __nv_bfloat16* inp, const __nv_bfloat16* weight,
    float eps, int B_T, int C);
void rmsnorm_backward(__nv_bfloat16* dinp, __nv_bfloat16* dweight, const __nv_bfloat16* dout, const __nv_bfloat16* inp,
    const __nv_bfloat16* weight, const float* rrms, int B_T, int C);

// rope
void rope_forward(__nv_bfloat16* q, __nv_bfloat16* k, int B, int T, int n_head, int n_kv_head,
    int head_dim, float theta);
void rope_backward(__nv_bfloat16* dq, __nv_bfloat16* dk, int B, int T, int n_head, int n_kv_head,
    int head_dim, float theta);

// attention
void attention_forward(__nv_bfloat16* out, __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v, 
    #ifdef USE_FLASH_ATTN
        float* lse, // flash: log-sum-exp [B, n_head, T]
    #else
        __nv_bfloat16* att,  // standard: scores [B, n_head, T, T]
    #endif
    const __nv_bfloat16* inp, const __nv_bfloat16* wq, const __nv_bfloat16* wk, const __nv_bfloat16* wv, const __nv_bfloat16* wo,
    int B, int T, int C, int n_head, int n_kv_head, float rope_theta, cublasHandle_t handle);
void attention_backward(__nv_bfloat16* dinp, __nv_bfloat16* dwq, __nv_bfloat16* dwk, __nv_bfloat16* dwv, __nv_bfloat16* dwo,
    const __nv_bfloat16* dout, const __nv_bfloat16* inp, const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    #ifdef USE_FLASH_ATTN
        const float* lse, // flash: saved log-sum-exp [B, n_head, T]
    #else
        const __nv_bfloat16* att,  // standard: saved probs [B, n_head, T, T]
    #endif
    const __nv_bfloat16* wq, const __nv_bfloat16* wk, const __nv_bfloat16* wv, const __nv_bfloat16* wo,
    int B, int T, int C, int n_head, int n_kv_head, float rope_theta, cublasHandle_t handle);

// swiglu
void swiglu_forward(__nv_bfloat16* out, __nv_bfloat16* gate, __nv_bfloat16* up, __nv_bfloat16* hidden,
    const __nv_bfloat16* inp, const __nv_bfloat16* w_gate, const __nv_bfloat16* w_up, const __nv_bfloat16* w_down,
    int B_T, int C, int ffn_hidden, cublasHandle_t handle);
void swiglu_backward(__nv_bfloat16* dinp, __nv_bfloat16* dw_gate, __nv_bfloat16* dw_up, __nv_bfloat16* dw_down,
    const __nv_bfloat16* dout, const __nv_bfloat16* inp, const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const __nv_bfloat16* w_gate, const __nv_bfloat16* w_up, const __nv_bfloat16* w_down,
    int B_T, int C, int ffn_hidden, cublasHandle_t handle);

// softmax
void softmax_forward(__nv_bfloat16* output, const __nv_bfloat16* input, int rows, int cols);

// crossentropy — losses stay float*, logits/probs are fp16
void crossentropy_forward(float* losses, const __nv_bfloat16* logits, const int* targets, int B, int T, int V);
void crossentropy_softmax_backward(__nv_bfloat16* dlogits, const __nv_bfloat16* probs, const int* targets,
    int B, int T, int V, float loss_scale = 1.0f);

// residual
void residual_forward(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int N);
void residual_backward(__nv_bfloat16* da, __nv_bfloat16* db, const __nv_bfloat16* dout, int N);

// adamw — mixed precision: fp16 weights/grads, fp32 master/m/v
void adamw_update(__nv_bfloat16* params, float* master_params, const __nv_bfloat16* grads,
    float* m, float* v, float lr, float beta1, float beta2, float eps,
    float weight_decay, float grad_scale, int t, int N);

float compute_grad_norm_bf16(float* dwte_fp32, int emb_N,
                             __nv_bfloat16* grads_bf16, int rest_N, float loss_scale);
void scale_grads(float* data, float scale, int N);
void fp32_to_bf16(const float* src, __nv_bfloat16* dst, int N);

void flash_attention_forward(
    __nv_bfloat16* context,
    float* lse, //keep lse as float* (one scalar per row)
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int B, int T, int n_head, int n_kv_head, int head_dim);

void flash_attention_backward(
    __nv_bfloat16* dq,
    __nv_bfloat16* dk,
    __nv_bfloat16* dv,    
    const __nv_bfloat16* dcontext,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,    
    const float* lse,
    int B, int T, int n_head, int n_kv_head, int head_dim);

