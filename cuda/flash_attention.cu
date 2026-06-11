#include "kernels.cuh"

// Flash-attention tile sizes (Ampere SM 8.6 / RTX 3060).
// SRAM per block = K_tile + V_tile = 2 * B_c * head_dim * sizeof(T):
//   bf16: 16 KB, fp32: 32 KB — both fit the 48 KB default (no opt-in).
constexpr int B_r = 64;   // query rows per block
constexpr int B_c = 64;   // key/value rows staged in SRAM per inner step

// WIP: fused forward. One block per (batch, query head, Q-tile); online softmax
// over KV tiles (running max/sum + acc), causal mask, GQA (kv_h = h / gqa_ratio).
// Writes context = acc / l and lse = m + log(l). bf16 path does all math in fp32.
template <typename Templ>
__global__ void flash_attention_forward_kernel(Templ* context, float* lse,
                                                const Templ* q, const Templ* k, const Templ* v,
                                                int B, int T, int n_head, int n_kv_head, int head_dim){

}

void flash_attention_forward(float* context,        // [B*T, C]
                            float* lse,            // [B, n_head, T], saved for backward
                            const float* q,        // [B*T, C]
                            const float* k,        // [B*T, kv_dim]
                            const float* v,        // [B*T, kv_dim]
                            int B,
                            int T,
                            int n_head,
                            int n_kv_head,
                            int head_dim){
    dim3 grid(B, n_head, ceil_div(T, B_r));
    dim3 block(B_r);
    size_t smem = 2 * B_c * head_dim * sizeof(float);
    flash_attention_forward_kernel<float><<<grid, block, smem>>>(
        context, lse, q, k, v, B, T, n_head, n_kv_head, head_dim);
}

void flash_attention_forward(__nv_bfloat16* context, float* lse,
                            const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                            int B, int T, int n_head, int n_kv_head, int head_dim) {
    dim3 grid(B, n_head, ceil_div(T, B_r));
    dim3 block(B_r);
    size_t smem = 2 * B_c * head_dim * sizeof(__nv_bfloat16);
    flash_attention_forward_kernel<__nv_bfloat16><<<grid, block, smem>>>(
        context, lse, q, k, v, B, T, n_head, n_kv_head, head_dim);
}


// WIP: fused backward. Recompute P_ij = exp(scale*dot(q_i,k_j) - lse_i) (no T×T saved).
// Pass A: D_i = sum_j P_ij * (dO_i·V_j). Pass B (causal j<=i): dS = P*(dP - D_i),
//   dQ_i += scale*dS*K_j (register, written once), atomicAdd dK += scale*dS*Q_i,
//   atomicAdd dV += P*dO_i (no scale). dK/dV use atomics (shared across GQA heads).
template <typename Templ>
__global__ void flash_attention_backward_kernel(
    Templ* dq, Templ* dk, Templ* dv, const Templ* dcontext,
    const Templ* q, const Templ* k, const Templ* v, const float* lse, int B, int T,
    int n_head, int n_kv_head, int head_dim){

}

void flash_attention_backward(float* dq,             // [B*T, C]
                                float* dk,             // [B*T, kv_dim], must be zeroed before call (atomicAdd)
                                float* dv,             // [B*T, kv_dim], must be zeroed before call (atomicAdd)
                                const float* dcontext, // [B*T, C]
                                const float* q,        // [B*T, C]
                                const float* k,        // [B*T, kv_dim]
                                const float* v,        // [B*T, kv_dim]
                                const float* lse,      // [B, n_head, T]
                                int B,
                                int T,
                                int n_head,
                                int n_kv_head,
                                int head_dim){
    dim3 grid(B, n_head, ceil_div(T, B_r));
    dim3 block(B_r);
    size_t smem = 2 * B_c * head_dim * sizeof(float);
    flash_attention_backward_kernel<float><<<grid, block, smem>>>(
        dq, dk, dv, dcontext, q, k, v, lse, B, T, n_head, n_kv_head, head_dim);
    }


void flash_attention_backward(__nv_bfloat16* dq,
                                __nv_bfloat16* dk,             // must be zeroed before call (atomicAdd)
                                __nv_bfloat16* dv,             // must be zeroed before call (atomicAdd)
                                const __nv_bfloat16* dcontext,
                                const __nv_bfloat16* q,
                                const __nv_bfloat16* k,
                                const __nv_bfloat16* v,
                                const float* lse,
                                int B,
                                int T,
                                int n_head,
                                int n_kv_head,
                                int head_dim){
    dim3 grid(B, n_head, ceil_div(T, B_r));
    dim3 block(B_r);
    size_t smem = 2 * B_c * head_dim * sizeof(__nv_bfloat16);
    flash_attention_backward_kernel<__nv_bfloat16><<<grid, block, smem>>>(
        dq, dk, dv, dcontext, q, k, v, lse, B, T, n_head, n_kv_head, head_dim);
    }
