#pragma once
#include "config.h"
#include "tensor.h"
#include <cublas_v2.h>

struct Model {
    ModelConfig config;
    cublasHandle_t cublas_handle;

    // --- All weights (one flat allocation) --- (params.storage.data)
    ParameterBlock params;
    float* wte;       // [vocab_size, C]        token embedding
    float* w_out;     // [vocab_size, C]        output projection
    float* rmsf_w;    // [C]                    final rmsnorm weight
    // per-layer (arrays of n_layer pointers, each into the flat buffer):
    float** rms1_w;   // [C] each               attention rmsnorm
    float** wq;       // [C, C] each            query projection
    float** wk;       // [C, kv_dim] each       key projection
    float** wv;       // [C, kv_dim] each       value projection
    float** wo;       // [C, C] each            output projection
    float** rms2_w;   // [C] each               mlp rmsnorm
    float** w_gate;   // [C, ffn_hidden] each   swiglu gate
    float** w_up;     // [C, ffn_hidden] each   swiglu up
    float** w_down;   // [ffn_hidden, C] each   swiglu down

    // --- All gradients (same layout as params) --- (grads.storage.data)
    ParameterBlock grads;
    float* dwte;
    float* dw_out;
    float* drmsf_w;
    float** drms1_w;
    float** dwq;
    float** dwk;
    float** dwv;
    float** dwo;
    float** drms2_w;
    float** dw_gate;
    float** dw_up;
    float** dw_down;

    // --- Activations (intermediates for forward/backward) --- (acts.storage.data)
    ParameterBlock acts;
    float* x;           // [B*T, C]              main hidden state
    float* ln1_out;     // [B*T, C]              output of first RMSNorm
    float* rrms1;       // [B*T]                 reciprocal RMS (saved for backward)
    float* q;           // [B*T, C]              query projection output
    float* k;           // [B*T, kv_dim]         key projection output
    float* v;           // [B*T, kv_dim]         value projection output
    float* lse;         // [B*n_head*T]          log-sum-exp per attention row (saved for flash backward)
    float* att;         // [B*n_head*T, T]       attention scores / probs
    float* attn_out;    // [B*T, C]              attention block output (after W_o)
    float* ln2_out;     // [B*T, C]              output of second RMSNorm
    float* rrms2;       // [B*T]                 reciprocal RMS (saved for backward)
    float* gate_buf;    // [B*T, ffn_hidden]     SwiGLU gate values
    float* up_buf;      // [B*T, ffn_hidden]     SwiGLU up values
    float* hidden_buf;  // [B*T, ffn_hidden]     SiLU(gate) * up
    float* mlp_out;     // [B*T, C]              MLP block output
    float* ln_final;    // [B*T, C]              output of final RMSNorm
    float* rrms_final;  // [B*T]                 reciprocal RMS (saved for backward)
    float* logits;      // [B*T, vocab_size]     final output before loss
    float* probs;       // [B*T, vocab_size]     softmax probs (written by crossentropy fwd)
    float* losses;      // [B*T]                 per-position losses

    // --- Gradient buffers (for backward pass intermediates) ---
    float* dx;          // [B*T, C]              gradient flowing back through residual stream
    float* dln1_out;    // [B*T, C]              gradient through first rmsnorm output
    float* dln2_out;    // [B*T, C]              gradient through second rmsnorm output
    float* dattn_out;   // [B*T, C]              gradient of attention output
    float* dmlp_out;    // [B*T, C]              gradient of MLP output
    float* dln_final;   // [B*T, C]              gradient through final rmsnorm output
    float* dlogits;     // [B*T, vocab_size]     gradient from loss
    float* dx_tmp;      // [B*T, C]              temp buffer for gradient accumulation

    // --- AdamW optimizer state ---
    Tensor adam_m;      // first moment  [n_params]
    Tensor adam_v;      // second moment [n_params]

    float loss;         // scalar loss for current batch
    const int* saved_targets;   // targets from forward(), needed by backward()

    void build(ModelConfig cfg, int B, int T);
    void load(const char* checkpoint_path);   // load weights from a saved .bin file
    void forward(const int* input_ids, const int* targets, int B, int T);
    void backward(int B, int T);
    void zero_grad();
    void update(float lr, TrainConfig tcfg, int step);
    void free();
};

// ======================== bf16 mixed precision model ========================
// Same architecture, but weights/activations/gradients stored as __nv_bfloat16 (bf16).
// fp32 master weights kept for optimizer precision. Tensor cores enabled.
struct ModelBF16 {
    ModelConfig config;
    cublasHandle_t cublas_handle;
    float loss_scale = 1.0f;       // no scaling needed — fp32 optimizer handles precision

    // --- fp16 weights (one flat allocation) ---
    HalfParameterBlock params;
    __nv_bfloat16* wte;
    __nv_bfloat16* w_out;
    __nv_bfloat16* rmsf_w;
    __nv_bfloat16** rms1_w;
    __nv_bfloat16** wq;
    __nv_bfloat16** wk;
    __nv_bfloat16** wv;
    __nv_bfloat16** wo;
    __nv_bfloat16** rms2_w;
    __nv_bfloat16** w_gate;
    __nv_bfloat16** w_up;
    __nv_bfloat16** w_down;

    // --- fp32 master weights (for optimizer precision) ---
    Tensor master_params;

    // --- fp16 gradients (same layout as params, except dwte is fp32) ---
    HalfParameterBlock grads;
    __nv_bfloat16* dwte;           // points into grads (fp16, placeholder — not used for accumulation)
    float* dwte_fp32;       // separate fp32 buffer for embedding grads (scatter-add can overflow fp16)
    __nv_bfloat16* dw_out;
    __nv_bfloat16* drmsf_w;
    __nv_bfloat16** drms1_w;
    __nv_bfloat16** dwq;
    __nv_bfloat16** dwk;
    __nv_bfloat16** dwv;
    __nv_bfloat16** dwo;
    __nv_bfloat16** drms2_w;
    __nv_bfloat16** dw_gate;
    __nv_bfloat16** dw_up;
    __nv_bfloat16** dw_down;

    // --- fp16 activations ---
    HalfParameterBlock acts;
    __nv_bfloat16* x;
    __nv_bfloat16* ln1_out;
    float* rrms1;       // stays fp32 (one scalar per row)
    __nv_bfloat16* q;
    __nv_bfloat16* k;
    __nv_bfloat16* v;
    __nv_bfloat16* lse; // [B, n_head, T]        log-sum-exp per attention row (fp32, saved for flash backward)
    __nv_bfloat16* att; 
    __nv_bfloat16* attn_out;
    __nv_bfloat16* ln2_out;
    float* rrms2;       // stays fp32
    __nv_bfloat16* gate_buf;
    __nv_bfloat16* up_buf;
    __nv_bfloat16* hidden_buf;
    __nv_bfloat16* mlp_out;
    __nv_bfloat16* ln_final;
    float* rrms_final;  // stays fp32
    __nv_bfloat16* logits;
    __nv_bfloat16* probs;
    float* losses;      // stays fp32 (loss needs precision)

    // --- fp16 gradient buffers ---
    __nv_bfloat16* dx;
    __nv_bfloat16* dln1_out;
    __nv_bfloat16* dln2_out;
    __nv_bfloat16* dattn_out;
    __nv_bfloat16* dmlp_out;
    __nv_bfloat16* dln_final;
    __nv_bfloat16* dlogits;
    __nv_bfloat16* dx_tmp;

    // --- AdamW optimizer state (fp32) ---
    Tensor adam_m;
    Tensor adam_v;

    float loss;
    const int* saved_targets;

    void build(ModelConfig cfg, int B, int T);
    void load(const char* checkpoint_path);
    void forward(const int* input_ids, const int* targets, int B, int T);
    void backward(int B, int T);
    void zero_grad();
    void update(float lr, TrainConfig tcfg, int step, float grad_clip_scale = 1.0f);
    void free();
};
